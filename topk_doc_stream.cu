#include "topk.h"

typedef uint4 group_t;  // uint32_t

void __global__ docQueryScoringCoalescedMemoryAccessSampleKernel(
    const __restrict__ uint16_t *docs,
    const int *doc_lens, const size_t n_docs,
    uint16_t *query, const int query_len, float *scores) {
    // each thread process one doc-query pair scoring task
    register auto tid = blockIdx.x * blockDim.x + threadIdx.x, tnum = gridDim.x * blockDim.x;

    if (tid >= n_docs) {
        return;
    }

    __shared__ uint16_t query_on_shm[MAX_QUERY_SIZE];
#pragma unroll
    for (auto i = threadIdx.x; i < query_len; i += blockDim.x) {
        query_on_shm[i] = query[i];  // not very efficient query loading temporally, as assuming its not hotspot
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
            register group_t loaded = ((group_t *)docs)[i * n_docs + doc_id];  // tid
            register uint16_t *doc_segment = (uint16_t *)(&loaded);
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

void doc_query_scoring_gpu(std::vector<std::vector<uint16_t>> &querys,
                           int start_doc_id,
                           std::vector<std::vector<uint16_t>> &docs,
                           std::vector<uint16_t> &lens,
                           std::vector<std::vector<int>> &indices,  // shape [querys.size(), TOPK]
                           std::vector<std::vector<float>> &scores  // shape [querys.size(), TOPK]
) {
    auto n_docs = docs.size();
    std::vector<float> s_scores(n_docs);
    std::vector<int> s_indices(n_docs);

    float *d_scores = nullptr;
    uint16_t *d_docs = nullptr, *d_query = nullptr;
    int *d_doc_lens = nullptr;

    // global memory alloc
    cudaMalloc(&d_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs);
    cudaMalloc(&d_scores, sizeof(float) * n_docs);
    cudaMalloc(&d_doc_lens, sizeof(int) * n_docs);

    uint16_t *h_docs = new uint16_t[MAX_DOC_SIZE * n_docs];
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
            auto final_offset = layer_0_offset * layer_0_stride + layer_1_offset * layer_1_stride + layer_2_offset;
            h_docs[final_offset] = docs[i][j];
        }
        h_doc_lens_vec[i] = docs[i].size();
    }

    // device, todo: rr workload by user query (userId)
    cudaDeviceProp device_props;
    cudaGetDeviceProperties(&device_props, 0);
    cudaSetDevice(0);

    for (auto &query : querys) {
        // init indices
        for (int i = 0; i < n_docs; ++i) {
            s_indices[i] = i + start_doc_id;
        }

        // doc stream
        const int N_STREAM = 8;
        cudaStream_t doc_streams[N_STREAM];
        std::cout << " Creating " << N_STREAM << " CUDA streams." << std::endl;
        for (int i = 0; i < N_STREAM; i++) {
            CUDA_CALL(cudaStreamCreate(&doc_streams[i]));
        }

        // init query global memory
        const size_t query_len = query.size();
        cudaMalloc(&d_query, sizeof(uint16_t) * query_len);
        cudaMemcpy(d_query, query.data(), sizeof(uint16_t) * query_len, cudaMemcpyHostToDevice);

        const int docs_chunk_size = n_docs / N_STREAM;
        int docs_offset = 0;
        int docs_len = docs_chunk_size;
        for (int i = 0; i < N_STREAM; i++) {
            // last docs_chunk_size
            if (n_docs % N_STREAM > 0 && i == N_STREAM - 1) {
                docs_len = docs_chunk_size + n_docs % N_STREAM;
            }
            docs_offset = i * docs_chunk_size;
            CUDA_CALL(cudaMemcpyAsync(d_docs + MAX_DOC_SIZE * docs_offset,
                                      h_docs + MAX_DOC_SIZE * docs_offset,
                                      sizeof(uint16_t) * MAX_DOC_SIZE * docs_len,
                                      cudaMemcpyHostToDevice, doc_streams[i]));
            CUDA_CALL(cudaMemcpyAsync(d_doc_lens + docs_offset,
                                      h_doc_lens_vec.data() + docs_offset,
                                      sizeof(int) * docs_len,
                                      cudaMemcpyHostToDevice, doc_streams[i]));
            // launch kernel
            int block = N_THREADS_IN_ONE_BLOCK;
            int grid = (docs_len + block - 1) / block;
            docQueryScoringCoalescedMemoryAccessSampleKernel<<<grid, block, 0, doc_streams[i]>>>(
                d_docs + MAX_DOC_SIZE * docs_offset,
                d_doc_lens + docs_offset,
                docs_len, d_query, query_len,
                d_scores + docs_offset);

            CUDA_CALL(cudaMemcpyAsync(s_scores.data() + docs_offset,
                                      d_scores + docs_offset,
                                      sizeof(float) * docs_len, cudaMemcpyDeviceToHost, doc_streams[i]));
            CUDA_CALL(cudaStreamSynchronize(doc_streams[i]));
            std::cout << "stream_id:  " << i << " docs_len:" << docs_len << std::endl;
        }
        CUDA_CHECK();

        int topk = s_scores.size() > TOPK ? TOPK : s_scores.size();
        // sort scores with Heap-based sort
        std::partial_sort(s_indices.begin(), s_indices.begin() + topk, s_indices.end(),
                          [&s_scores, start_doc_id](const int &a, const int &b) {
                              if (s_scores[a - start_doc_id] != s_scores[b - start_doc_id]) {
                                  return s_scores[a - start_doc_id] > s_scores[b - start_doc_id];  // by score DESC
                              }
                              return a < b;  // the same score, by index ASC
                          });

        std::vector<int> topk_doc_ids(s_indices.begin(), s_indices.begin() + topk);
        indices.push_back(topk_doc_ids);

        std::vector<float> topk_scores(topk_doc_ids.size());
        int i = 0;
        for (auto doc_id : topk_doc_ids) {
            topk_scores[i++] = s_scores[doc_id - start_doc_id];
        }
        scores.push_back(topk_scores);

        for (int i = 0; i < N_STREAM; i++) {
            cudaStreamDestroy(doc_streams[i]);
        }

        cudaFree(d_query);
    }

    // deallocation
    cudaFree(d_docs);
    // cudaFree(d_query);
    cudaFree(d_scores);
    cudaFree(d_doc_lens);
    free(h_docs);

    // free(doc_streams);
}