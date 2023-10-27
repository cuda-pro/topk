/*
 * SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <dirent.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/time.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include "threadpool.h"

// #define GPU
#define DEBUG
#define CPU
#define TOPK 100

#ifdef GPU
#include <cuda.h>

#include "topk.h"
#endif

template <typename T>
void print(std::vector<T> const& v) {
    for (auto i : v) {
        std::cout << i << ' ';
    }
    std::cout << std::endl;
}

template <typename T>
std::vector<T> sub_vec_v1(std::vector<T> const& v, int m, int n) {
    auto first = v.cbegin() + m;
    auto last = v.cbegin() + n + 1;

    std::vector<T> vec(first, last);
    return vec;
}
template <typename T>
std::vector<T> sub_vec(std::vector<T>& v, int m, int n) {
    std::vector<T> vec;
    std::copy(v.begin() + m, v.begin() + n + 1, std::back_inserter(vec));
    return vec;
}

std::vector<std::string> getFilesInDirectory(const std::string& directory) {
    std::vector<std::string> files;
    DIR* dirp = opendir(directory.c_str());
    struct dirent* dp;
    while ((dp = readdir(dirp)) != NULL) {
        struct stat path_stat;
        stat((directory + "/" + dp->d_name).c_str(), &path_stat);
        if (S_ISREG(path_stat.st_mode))  // Check if it's a regular file - not a directory
            files.push_back(dp->d_name);
    }
    closedir(dirp);
    std::sort(files.begin(), files.end());  // sort the files
    return files;
}

struct UserSpecifiedInput {
    int n_docs;
    std::vector<std::vector<uint16_t>> querys;
    std::vector<std::vector<uint16_t>> docs;
    std::vector<uint16_t> doc_lens;

    UserSpecifiedInput(std::string qf, std::string df) {
        load(qf, df);
    }

    void load(std::string query_file_dir, std::string docs_file_name) {
        std::stringstream ss;
        std::string tmp_str;
        std::string tmp_index_str;

        std::vector<std::string> files = getFilesInDirectory(query_file_dir);
        for (const auto& query_file_name : files) {
            std::vector<uint16_t> single_query;

            std::ifstream query_file(query_file_dir + "/" + query_file_name);
            while (std::getline(query_file, tmp_str)) {
                ss.clear();
                ss << tmp_str;
                std::cout << query_file_name << ":" << tmp_str << std::endl;
                while (std::getline(ss, tmp_index_str, ',')) {
                    single_query.emplace_back(std::stoi(tmp_index_str));
                }
            }
            query_file.close();
            ss.clear();
            std::sort(single_query.begin(), single_query.end());  // pre-sort the query
            querys.emplace_back(single_query);
        }
        std::cout << "query_size: " << querys.size() << std::endl;

        std::ifstream docs_file(docs_file_name);
        while (std::getline(docs_file, tmp_str)) {
            std::vector<uint16_t> next_doc;
            ss.clear();
            ss << tmp_str;
            while (std::getline(ss, tmp_index_str, ',')) {
                next_doc.emplace_back(std::stoi(tmp_index_str));
            }
            docs.emplace_back(next_doc);
            doc_lens.emplace_back(next_doc.size());
            // todo: send task to thread pool
        }
        docs_file.close();
        ss.clear();
        n_docs = docs.size();
        std::cout << "doc_size: " << docs.size() << std::endl;
    }
};

// 求query与doc全集内各数据交集个数平均分 topk (k=100); 这里定义item交集分数为：
// query[i] >= doc[j](0 <= i < query_size, 0 <= j < doc_size) 算一个交集, 平均分为 query与doc交集数目 / max(query_size, doc_size)
void doc_query_scoring_cpu(std::vector<std::vector<uint16_t>>& querys,
                           std::vector<std::vector<uint16_t>>& docs,
                           std::vector<uint16_t>& lens,
                           std::vector<std::vector<int>>& indices  // shape [querys.size(), TOPK]
) {
#ifdef DEBUG
    printf("doc_query_scoring_cpu query_size:%zu\t docs_size:%zu\t lens_size:%zu\n", querys.size(), docs.size(), lens.size());
    std::cout << "query:" << std::endl;
    for (auto query : querys) {
        print(query);
    }
    std::cout << "doc:" << std::endl;
    for (auto doc : docs) {
        print(doc);
    }
    std::cout << "len:" << std::endl;
    print(lens);
#endif

    std::vector<int> s_indices(docs.size());
    for (auto& query : querys) {
        // init indices (doc_id) for partial sort with score
        for (int id = 0; id < docs.size(); ++id) {
            s_indices[id] = id;
        }

        int i = 0;
        std::vector<float> scores(docs.size());
        for (int doc_id = 0; doc_id < docs.size(); doc_id++) {
            int tmp_score = 0;
            auto& doc = docs[doc_id];
            for (int j = 0; j < doc.size(); j++) {
                while (i < query.size() && query[i] < doc[j]) {
                    i++;
                }
                if (i < query.size()) {
                    tmp_score += (query[i] == doc[j]);
                }
            }
            scores[doc_id] = tmp_score / std::max(query.size(), doc.size());
        }
        std::cout << "query:" << std::endl;
        print(query);
        std::cout << "scores:" << std::endl;
        print(scores);

        // sort scores with min heap Heap-based sort
        int topk = docs.size() > TOPK ? TOPK : docs.size();
        std::partial_sort(s_indices.begin(), s_indices.begin() + topk, s_indices.end(),
                          [&scores](const int& a, const int& b) {
                              if (scores[a] != scores[b]) {
                                  return scores[a] > scores[b];  // 按照分数降序排序
                              }
                              return a < b;  // 如果分数相同，按索引从小到大排序
                          });
        std::vector<int> s_ans(s_indices.begin(), s_indices.begin() + topk);
        indices.push_back(s_ans);
    }
}

void doc_query_scoring(std::vector<std::vector<uint16_t>>& querys,
                       std::vector<std::vector<uint16_t>>& docs,
                       std::vector<uint16_t>& lens,
                       std::vector<std::vector<int>>& indices  // shape [querys.size(), TOPK]
) {
#ifdef GPU
    doc_query_scoring_gpu()
#endif

#ifdef CPU
        doc_query_scoring_cpu(querys, docs, lens, indices);
#endif
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cout << "Usage: query_doc_scoring.bin <doc_file_name> <query_file_name> <output_filename>" << std::endl;
        return -1;
    }
    std::string doc_file_name = argv[1];
    std::string query_file_dir = argv[2];
    std::string output_file = argv[3];

    std::cout << "start get topk" << std::endl;

    // read file
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    UserSpecifiedInput inputs(query_file_dir, doc_file_name);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::cout << "read file cost " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms " << std::endl;

    // score topk
#ifdef CPU_CONCURENCY
    int concurrency = std::thread::hardware_concurrency();
    std::cout << "hardware concurrency:" << concurrency << std::endl;
    ThreadPool pool(concurrency);
    std::vector<std::future<std::vector<std::vector<int>>>> results;
    int split_docs_size = int(inputs.docs.size() / concurrency);
    std::cout << "split_docs_size:" << split_docs_size << std::endl;
    for (int i = 0; i < concurrency; ++i) {
        int start = i * split_docs_size;
        int end = start + split_docs_size - 1;
        results.emplace_back(
            pool.enqueue([&inputs, start, end] {
                std::vector<std::vector<int>> indices;
                std::vector<std::vector<uint16_t>> sub_docs = sub_vec(inputs.docs, start, end);
                std::vector<uint16_t> sub_doc_lens = sub_vec(inputs.doc_lens, start, end);
                // printf("start:%d\tend:%d\t; sub_docs_size:%zu\t sub_doc_lens_size:%zu\n", start, end, sub_docs.size(), sub_doc_lens.size());
                doc_query_scoring(inputs.querys, sub_docs, sub_doc_lens, indices);
                return indices;
            }));
    }
    if (inputs.docs.size() % concurrency > 0) {
        int start = concurrency * split_docs_size;
        int end = inputs.docs.size() - 1;
        results.emplace_back(
            pool.enqueue([&inputs, start, end] {
                std::vector<std::vector<int>> sub_indices;
                std::vector<std::vector<uint16_t>> sub_docs = sub_vec(inputs.docs, start, end);
                std::vector<uint16_t> sub_doc_lens = sub_vec(inputs.doc_lens, start, end);
                // printf("start:%d\tend:%d\t; sub_docs_size:%zu\t sub_doc_lens_size:%zu\n", start, end, sub_docs.size(), sub_doc_lens.size());
                doc_query_scoring(inputs.querys, sub_docs, sub_doc_lens, sub_indices);
                return sub_indices;
            }));
    }

    std::vector<std::vector<int>> indices;
    for (auto&& result : results) {
        auto res = result.get();
        if (res.size() == 0) {
            continue;
        }
        std::cout << "reslut ";
        for (auto vec : res) {
            print(vec);
        }
        std::cout << std::endl;
    }
#else
    std::vector<std::vector<int>> indices;
    doc_query_scoring(inputs.querys, inputs.docs, inputs.doc_lens, indices);
#endif

    std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();
    std::cout << "topk cost " << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() << " ms " << std::endl;

    // get total time
    std::chrono::milliseconds total_time = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t1);
    // write result data
    std::ofstream ofs;
    ofs.open(output_file, std::ios::out);
    // first line topk cost time in ms
    ofs << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() << std::endl;
    // topk index
    for (auto& s_indices : indices) {  // makesure indices.size() == querys.size()
        for (size_t i = 0; i < s_indices.size(); ++i) {
            ofs << s_indices[i];
            if (i != s_indices.size() - 1)  // if not the last element
                ofs << "\t";
        }
        ofs << "\n";
    }
    ofs.close();

    std::cout << "all cost " << total_time.count() << " ms " << std::endl;
    std::cout << "end get topk" << std::endl;
    return 0;
}
