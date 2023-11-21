#include "helper.h"
#include "topk.h"

// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#built-in-vector-types
typedef uint4 group_t;  // cuda uint4: 4 * uint (sizeof(uint4)=16 128bit)

void __global__ docQueryScoringCoalescedMemoryAccessSampleKernel(const __restrict__ uint16_t* docs,
                                                                 const int* doc_lens, const size_t n_docs,
                                                                 uint16_t* query, const int query_len,
                                                                 float* scores) {
    // each thread process one doc-query pair scoring task
    register auto tid = blockIdx.x * blockDim.x + threadIdx.x, tnum = gridDim.x * blockDim.x;

    if (tid >= n_docs) {
        return;
    }

    __shared__ uint16_t query_on_shm[MAX_QUERY_SIZE];
#pragma unroll
    for (auto i = threadIdx.x; i < query_len; i += blockDim.x) {
        query_on_shm[i] =
            query[i];  // not very efficient query loading temporally, as assuming its not hotspot
    }

    __syncthreads();

    for (auto doc_id = tid; doc_id < n_docs; doc_id += tnum) {
        register int query_idx = 0;

        register float tmp_score = 0.;

        register bool no_more_load = false;

        for (auto i = 0; i < MAX_DOC_SIZE / (sizeof(group_t) / sizeof(uint16_t)); i++) {
            if (no_more_load) {
                break;
            }
            register group_t loaded = ((group_t*)docs)[i * n_docs + doc_id];  // tid
            register uint16_t* doc_segment = (uint16_t*)(&loaded);
            for (auto j = 0; j < sizeof(group_t) / sizeof(uint16_t); j++) {
                if (doc_segment[j] == 0) {
                    no_more_load = true;
                    break;
                    // return;
                }
                while (query_idx < query_len && query_on_shm[query_idx] < doc_segment[j]) {
                    ++query_idx;
                }
                if (query_idx < query_len) {
                    tmp_score += (query_on_shm[query_idx] == doc_segment[j]);
                }
            }
            __syncwarp();
        }
        scores[doc_id] = tmp_score / max(query_len, doc_lens[doc_id]);  // tid
    }
}

void doc_query_scoring_gpu_function(
    std::vector<std::vector<uint16_t>>& querys,
    std::vector<std::vector<uint16_t>>& docs,
    std::vector<uint16_t>& lens,
    std::vector<std::vector<int>>& indices  // shape [querys.size(), TOPK]
) {
    auto n_docs = docs.size();
    auto n_querys = querys.size();
    std::vector<int> s_indices(n_docs);

    float* d_scores = nullptr;
    uint16_t *d_docs = nullptr, *d_query = nullptr;
    int* d_doc_lens = nullptr;

    // copy to device
    cudaMalloc(&d_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs);
    cudaMalloc(&d_scores, sizeof(float) * n_docs);
    cudaMalloc(&d_doc_lens, sizeof(int) * n_docs);

    uint16_t* h_docs = new uint16_t[MAX_DOC_SIZE * n_docs];
    memset(h_docs, 0, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs);
    std::vector<int> h_doc_lens_vec(n_docs);
    for (int i = 0; i < docs.size(); i++) {
        for (int j = 0; j < docs[i].size(); j++) {
            auto group_sz = sizeof(group_t) / sizeof(uint16_t);
            auto layer_0_offset = j / group_sz;
            auto layer_0_stride = n_docs * group_sz;
            auto layer_1_offset = i;
            auto layer_1_stride = group_sz;
            auto layer_2_offset = j % group_sz;
            auto final_offset =
                layer_0_offset * layer_0_stride + layer_1_offset * layer_1_stride + layer_2_offset;
            h_docs[final_offset] = docs[i][j];
        }
        h_doc_lens_vec[i] = docs[i].size();
    }

    cudaMemcpy(d_docs, h_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs, cudaMemcpyHostToDevice);
    cudaMemcpy(d_doc_lens, h_doc_lens_vec.data(), sizeof(int) * n_docs, cudaMemcpyHostToDevice);

    // device
    cudaDeviceProp device_props;
    cudaGetDeviceProperties(&device_props, 0);
    cudaSetDevice(0);

    // query documents scores
    std::vector<std::vector<float>> q_scores(n_querys);

    // query stream
    cudaStream_t* q_streams = (cudaStream_t*)malloc(n_querys * sizeof(cudaStream_t));

    for (int i = 0; i < n_querys; i++) {
        cudaStreamCreate(&q_streams[i]);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    int stream_id = 0;
    for (auto& query : querys) {
        const size_t query_len = query.size();
        cudaMalloc(&d_query, sizeof(uint16_t) * query_len);
        cudaMemcpyAsync(d_query,
                        query.data(),
                        sizeof(uint16_t) * query_len,
                        cudaMemcpyHostToDevice,
                        q_streams[stream_id]);

        // launch kernel
        int block = N_THREADS_IN_ONE_BLOCK;
        int grid = (n_docs + block - 1) / block;
        docQueryScoringCoalescedMemoryAccessSampleKernel<<<grid, block, 0, q_streams[stream_id]>>>(
            d_docs, d_doc_lens, n_docs, d_query, query_len, d_scores);
        std::vector<float> s_scores(n_docs);
        cudaMemcpyAsync(s_scores.data(),
                        d_scores,
                        sizeof(float) * n_docs,
                        cudaMemcpyDeviceToHost,
                        q_streams[stream_id]);
        cudaStreamSynchronize(q_streams[stream_id]);
        q_scores[stream_id] = s_scores;
        std::cout << "stream_id:  " << stream_id << " scores size:" << q_scores[stream_id].size()
                  << std::endl;

        stream_id++;
        cudaFree(d_query);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("elapsed time:%f ms\n", elapsed_time);

    std::chrono::high_resolution_clock::time_point t = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_querys; i++) {
        // init indices
        for (int j = 0; j < n_docs; ++j) {
            s_indices[j] = j;
        }

        auto scores = q_scores[i];
        std::cout << "scores size:" << scores.size() << std::endl;
        int topk = scores.size() > TOPK ? TOPK : scores.size();
        // sort scores
        std::chrono::high_resolution_clock::time_point tt = std::chrono::high_resolution_clock::now();
        std::partial_sort(s_indices.begin(),
                          s_indices.begin() + topk,
                          s_indices.end(),
                          [&scores](const int& a, const int& b) {
                              if (scores[a] != scores[b]) {
                                  return scores[a] > scores[b];  // 按照分数降序排序
                              }
                              return a < b;  // 如果分数相同，按索引从小到大排序
                          });
        std::chrono::high_resolution_clock::time_point tt1 = std::chrono::high_resolution_clock::now();
        std::cout << "partial_sort cost "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(tt1 - tt).count() << " ms "
                  << std::endl;

        std::vector<int> s_ans(s_indices.begin(), s_indices.begin() + topk);
        indices.push_back(s_ans);
    }
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    std::cout << "total partial_sort cost "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t).count() << " ms "
              << std::endl;

    for (int i = 0; i < n_querys; i++) {
        cudaStreamDestroy(q_streams[i]);
    }

    cudaFree(d_docs);
    cudaFree(d_scores);
    cudaFree(d_doc_lens);
    delete[] h_docs;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(q_streams);
}
