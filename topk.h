#pragma once
#include <cuda.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

#define MAX_DOC_SIZE 128
#define MAX_QUERY_SIZE 4096
#define N_THREADS_IN_ONE_BLOCK 512
#define TOPK 100

/*
// Amazing~ no last params, nvcc compile ok, run is ok....
void doc_query_scoring_gpu(std::vector<std::vector<uint16_t>> &query,
                           std::vector<std::vector<uint16_t>> &docs,
                           std::vector<uint16_t> &lens,
                           std::vector<std::vector<int>> &indices);
*/

void doc_query_scoring_gpu(std::vector<std::vector<uint16_t>> &query,
                           int start_doc_id,
                           std::vector<std::vector<uint16_t>> &docs,
                           std::vector<uint16_t> &lens,
                           std::vector<std::vector<int>> &indices,  // shape [querys.size(), TOPK]
                           std::vector<std::vector<float>> &scores  // shape [querys.size(), TOPK]
);
