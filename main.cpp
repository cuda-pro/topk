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
#include <unordered_map>
#include <vector>

#include "threadpool.h"

// golang/rust compile feature (GPU,DEBUG,CPU) like this define
// #define DEBUG
#define CPU
// #define CPU_CONCURENCY
// #define GPU

#ifdef GPU
#include <cuda.h>

#include "topk.h"
#endif

#define TOPK 100

size_t get_file_size(const char* fileName) {
    if (fileName == NULL) {
        return 0;
    }

    struct stat statbuf;
    stat(fileName, &statbuf);
    size_t filesize = statbuf.st_size;

    return filesize;
}

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
struct DocScores {
    std::vector<std::vector<int>> indices;   // shape [querys.size(), TOPK]
    std::vector<std::vector<float>> scores;  // shape [querys.size(), TOPK]

    DocScores(std::vector<std::vector<int>>& i, std::vector<std::vector<float>>& s) {
        indices = i;
        scores = s;
    }
};

struct UserTopkQueryDocsInputPipeline {
    int n_docs;
    std::vector<std::vector<uint16_t>> querys;
    std::vector<std::vector<uint16_t>> docs;
    std::vector<uint16_t> doc_lens;
    int worker_pool_size;
    std::vector<std::future<DocScores>> results;
    int split_docs_size;

    UserTopkQueryDocsInputPipeline(std::string qf, std::string df, int wk_size, int split_size) {
        worker_pool_size = wk_size;
        split_docs_size = split_size;
        topk_pipeline(qf, df);
    }

    void topk_pipeline(std::string query_file_dir, std::string docs_file_name) {
        ThreadPool pool(worker_pool_size);
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

        int line_cn = 0;
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
            line_cn++;
        }
        docs_file.close();
        ss.clear();
        n_docs = docs.size();
        std::cout << "doc_size: " << docs.size() << std::endl;
    }
};

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
        }
        docs_file.close();
        ss.clear();
        n_docs = docs.size();
        std::cout << "doc_size: " << docs.size() << std::endl;
    }
};

// intersection(query,doc): query[i] == doc[j](0 <= i < query_size, 0 <= j < doc_size)
// score = total_intersection(query,doc) / max(query_size, doc_size)
// note: query/doc vec must sorted by ASC
void doc_query_scoring_cpu(std::vector<std::vector<uint16_t>>& querys,
                           int start_doc_id,
                           std::vector<std::vector<uint16_t>>& docs,
                           std::vector<uint16_t>& lens,
                           std::vector<std::vector<int>>& indices,  // shape [querys.size(), topk]
                           std::vector<std::vector<float>>& scores  // shape [querys.size(), topk]
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
            s_indices[id] = id + start_doc_id;
        }

        // int query map <query_token, cn>
        std::unordered_map<uint16_t, int> map_query(query.size());
        for (auto q : query) {
            map_query[q]++;
        }

        std::vector<float> s_scores(docs.size());
        for (int id = 0; id < docs.size(); id++) {
            float tmp_score = 0;
            auto doc = docs[id];
            for (int j = 0; j < doc.size(); j++) {
                tmp_score += map_query[doc[j]];
            }
            s_scores[id] = tmp_score / std::max(query.size(), doc.size());
        }
#ifdef DEBUG
        std::cout << "query:" << std::endl;
        print(query);
        std::cout << "scores:" << std::endl;
        print(s_scores);
#endif
        int topk = docs.size() > TOPK ? TOPK : docs.size();
        // sort scores with Heap-based select topk sort
        std::partial_sort(s_indices.begin(), s_indices.begin() + topk, s_indices.end(),
                          [&s_scores, start_doc_id](const int& a, const int& b) {
                              if (s_scores[a - start_doc_id] != s_scores[b - start_doc_id]) {
                                  return s_scores[a - start_doc_id] > s_scores[b - start_doc_id];  // by score DESC
                              }
                              return a < b;  // the same score, by doc_id ASC
                          });

        std::vector<int> topk_doc_ids(s_indices.begin(), s_indices.begin() + topk);
        indices.push_back(topk_doc_ids);

        std::vector<float> topk_scores(topk_doc_ids.size());
        int i = 0;
        for (auto doc_id : topk_doc_ids) {
            topk_scores[i++] = s_scores[doc_id - start_doc_id];
        }
        scores.push_back(topk_scores);
    }
}

void doc_query_scoring(std::vector<std::vector<uint16_t>>& querys,
                       int start_doc_id,
                       std::vector<std::vector<uint16_t>>& docs,
                       std::vector<uint16_t>& lens,
                       std::vector<std::vector<int>>& indices,  // shape [querys.size(), TOPK]
                       std::vector<std::vector<float>>& scores  // shape [querys.size(), TOPK]
) {
#ifdef GPU
    doc_query_scoring_gpu(querys, start_doc_id, docs, lens, indices, scores);
#else
    doc_query_scoring_cpu(querys, start_doc_id, docs, lens, indices, scores);
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
    // note: big file need split some small file for map/reduce
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    UserSpecifiedInput inputs(query_file_dir, doc_file_name);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::cout << "read file cost " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms " << std::endl;

    // score topk, just instruction bound, no io
#ifdef CPU_CONCURENCY
    int concurrency = std::thread::hardware_concurrency();
    std::cout << "hardware concurrency:" << concurrency << std::endl;
    ThreadPool pool(concurrency);
    std::vector<std::future<DocScores>> results;
    int split_docs_size = int(inputs.docs.size() / concurrency);
    std::cout << "split_docs_size:" << split_docs_size << std::endl;
    for (int i = 0; i < concurrency && split_docs_size > 0; ++i) {
        int start = i * split_docs_size;
        int end = start + split_docs_size - 1;
        results.emplace_back(
            pool.enqueue([&inputs, start, end] {
                std::vector<std::vector<int>> sub_indices;
                std::vector<std::vector<float>> sub_scores;
                std::vector<std::vector<uint16_t>> sub_docs = sub_vec(inputs.docs, start, end);
                std::vector<uint16_t> sub_doc_lens = sub_vec(inputs.doc_lens, start, end);
                // printf("start:%d\tend:%d\t; sub_docs_size:%zu\t sub_doc_lens_size:%zu\n", start, end, sub_docs.size(), sub_doc_lens.size());
                doc_query_scoring(inputs.querys, start, sub_docs, sub_doc_lens, sub_indices, sub_scores);
                DocScores ds(sub_indices, sub_scores);
                return ds;
            }));
    }
    if (inputs.docs.size() % concurrency > 0) {
        int start = concurrency * split_docs_size;
        int end = inputs.docs.size() - 1;
        results.emplace_back(
            pool.enqueue([&inputs, start, end] {
                std::vector<std::vector<int>> sub_indices;
                std::vector<std::vector<float>> sub_scores;
                std::vector<std::vector<uint16_t>> sub_docs = sub_vec(inputs.docs, start, end);
                std::vector<uint16_t> sub_doc_lens = sub_vec(inputs.doc_lens, start, end);
                // printf("start:%d\tend:%d\t; sub_docs_size:%zu\t sub_doc_lens_size:%zu\n", start, end, sub_docs.size(), sub_doc_lens.size());
                doc_query_scoring(inputs.querys, start, sub_docs, sub_doc_lens, sub_indices, sub_scores);
                DocScores ds(sub_indices, sub_scores);
                return ds;
            }));
    }
    std::cout << std::endl;

    // reduce topk
    std::vector<std::vector<int>> s_indices(inputs.querys.size());
    std::vector<std::vector<float>> s_scores(inputs.querys.size());
    for (auto&& result : results) {
        auto res = result.get();
        if (res.indices.size() == 0) {
            continue;
        }

        int q_id = 0;
        for (auto& doc_ids : res.indices) {
            s_indices[q_id].reserve(s_indices[q_id].size() + doc_ids.size());
            s_indices[q_id].insert(s_indices[q_id].end(), doc_ids.begin(), doc_ids.end());
            q_id++;
        }

        q_id = 0;
        for (auto doc_scores : res.scores) {
            s_scores[q_id].reserve(s_scores[q_id].size() + res.scores[q_id].size());
            s_scores[q_id].insert(s_scores[q_id].end(), res.scores[q_id].begin(), res.scores[q_id].end());
            q_id++;
        }
    }

    std::cout << "after merge,doc_ids: ";
    for (auto& doc_ids : s_indices) {
        print(doc_ids);
    }
    std::cout << "after merge,scores: ";
    for (auto& doc_scores : s_scores) {
        print(doc_scores);
    }
    std::cout << std::endl;

    // reduce topk -> topk
    std::vector<std::vector<int>> indices;
    int topk = inputs.docs.size() > TOPK ? TOPK : inputs.docs.size();
    for (int q_id = 0; q_id < inputs.querys.size(); q_id++) {
        auto doc_ids = s_indices[q_id];
        std::vector<int> idx(doc_ids.size());
        // init index for partial sort with score
        for (int id = 0; id < doc_ids.size(); ++id) {
            idx[id] = id;
        }
        auto doc_scores = s_scores[q_id];
        // sort scores with Heap-based select topk sort
        std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(),
                          [&doc_scores, doc_ids](const int& a, const int& b) {
                              if (doc_scores[a] != doc_scores[b]) {
                                  return doc_scores[a] > doc_scores[b];  // by score DESC
                              }
                              return doc_ids[a] < doc_ids[b];  // the same score, by doc_id ASC
                          });
        std::vector<int> topk_idxes(idx.begin(), idx.begin() + topk);
        std::vector<int> topk_doc_ids(topk_idxes.size());
        int i = 0;
        for (auto index : topk_idxes) {
            topk_doc_ids[i++] = doc_ids[index];
        }
        indices.push_back(topk_doc_ids);
    }
#else
    std::vector<std::vector<int>> indices;
    std::vector<std::vector<float>> scores;
    doc_query_scoring(inputs.querys, 0, inputs.docs, inputs.doc_lens, indices, scores);
#endif
    std::cout << "indices: ";
    for (auto& ind : indices) {
        print(ind);
    }

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
