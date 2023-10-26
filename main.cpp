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
#define CPU

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
std::vector<T> vec_slice(std::vector<T> const& v, int m, int n) {
    auto first = v.cbegin() + m;
    auto last = v.cbegin() + n + 1;

    std::vector<T> vec(first, last);
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

void doc_query_scoring_cpu(std::vector<std::vector<uint16_t>>& querys,
                           std::vector<std::vector<uint16_t>>& docs,
                           std::vector<uint16_t>& lens,
                           std::vector<std::vector<int>>& indices  // shape [querys.size(), TOPK]
) {
    printf("query_size:%zu\t doc_size:%zu\t lens:%zu", querys.size(), docs.size(), lens.size());
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
    std::vector<std::vector<int>> indices;
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::cout << "read file cost " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms " << std::endl;

    // score topk
    int concurrency = std::thread::hardware_concurrency();
    std::cout << "hardware concurrency:" << concurrency << std::endl;
    ThreadPool pool(concurrency);
    std::vector<std::future<int>> results;
    for (int i = 0; i < concurrency; ++i) {
        std::vector<std::vector<uint16_t>> sub_vec = vec_slice(inputs.docs, i, int(inputs.docs.size() / concurrency));
        results.emplace_back(
            pool.enqueue([i] {
                // std::cout << "hello " << i << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(1));
                // std::cout << "world " << i << std::endl;
                return i * i;
            }));
    }
    for (auto&& result : results)
        printf("result:%d\n", result.get());

    doc_query_scoring(inputs.querys, inputs.docs, inputs.doc_lens, indices);

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
