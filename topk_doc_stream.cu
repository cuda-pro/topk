#include "helper.h"
#include "topk.h"

// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#built-in-vector-types
typedef uint4 group_t;  // cuda uint4: 4 * uint (sizeof(uint4)=16 128bit)

void __global__ docQueryScoringCoalescedMemoryAccessSampleKernel(const __restrict__ uint16_t *docs,
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
    // device
    cudaDeviceProp device_props;
    cudaGetDeviceProperties(&device_props, 0);
    cudaSetDevice(0);
    // check deviceOverlap
    if (!device_props.deviceOverlap) {
        printf("device don't support deviceOverlap,so can't speed up the process from streams\n");
        return;
    }

    auto n_docs = docs.size();
    std::vector<int> s_indices(n_docs);
    std::chrono::high_resolution_clock::time_point it = std::chrono::high_resolution_clock::now();
    // std::iota(s_indices.begin(), s_indices.end(), start_doc_id);
    // #pragma omp parallel for schedule(static)
    for (int j = 0; j < n_docs; ++j) {
        s_indices[j] = j + start_doc_id;
    }
    std::chrono::high_resolution_clock::time_point it1 = std::chrono::high_resolution_clock::now();
    std::cout << "iota indeices cost " << std::chrono::duration_cast<std::chrono::milliseconds>(it1 - it).count() << " ms " << std::endl;

    std::chrono::high_resolution_clock::time_point dgt = std::chrono::high_resolution_clock::now();
    uint16_t *h_docs = nullptr;
    int *h_doc_lens_vec = nullptr;
    float *s_scores = nullptr;
#ifdef PINNED_MEMORY
    // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gb65da58f444e7230d3322b6126bb4902
    cudaMallocHost(&h_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs);  // cudaHostAllocDefault
    // cudaHostAlloc(&h_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs, cudaHostAllocDefault);
    cudaHostAlloc(&h_doc_lens_vec, sizeof(int) * n_docs, cudaHostAllocDefault);
    cudaHostAlloc(&s_scores, sizeof(float) * n_docs, cudaHostAllocDefault);
#else
    h_docs = new uint16_t[MAX_DOC_SIZE * n_docs];
    h_doc_lens_vec = new int[n_docs];
    s_scores = new float[n_docs];
#endif
    memset(h_docs, 0, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs);
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
    std::chrono::high_resolution_clock::time_point dgt1 = std::chrono::high_resolution_clock::now();
    std::cout << "align group docs cost " << std::chrono::duration_cast<std::chrono::milliseconds>(dgt1 - dgt).count() << " ms " << std::endl;

    // doc stream
    const int N_STREAM = 1;
    const int docs_chunk_size = n_docs / N_STREAM;
    int docs_offset = 0;
    int docs_len = docs_chunk_size;
    cudaStream_t doc_streams[N_STREAM];
    for (int i = 0; i < N_STREAM; i++) {
        CUDA_CALL(cudaStreamCreateWithFlags(&doc_streams[i], cudaStreamNonBlocking));
    }

    // global memory alloc
    uint16_t *d_docs[N_STREAM];
    int *d_doc_lens[N_STREAM];
    float *d_scores[N_STREAM];
    for (int i = 0; i < N_STREAM; i++) {
        // last docs_chunk_size
        if (n_docs % N_STREAM > 0 && i == N_STREAM - 1) {
            docs_len = docs_chunk_size + n_docs % N_STREAM;
        }
        cudaMalloc(&d_docs[i], sizeof(uint16_t) * MAX_DOC_SIZE * docs_len);
        cudaMalloc(&d_doc_lens[i], sizeof(int) * docs_len);
        cudaMalloc(&d_scores[i], sizeof(float) * docs_len);
    }

    std::chrono::high_resolution_clock::time_point ddt = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_STREAM; i++) {
        // last docs_chunk_size
        if (n_docs % N_STREAM > 0 && i == N_STREAM - 1) {
            docs_len = docs_chunk_size + n_docs % N_STREAM;
        }
        docs_offset = i * docs_chunk_size;
        std::chrono::high_resolution_clock::time_point dt = std::chrono::high_resolution_clock::now();
        CUDA_CALL(cudaMemcpyAsync(d_docs[i],
                                  h_docs + MAX_DOC_SIZE * docs_offset,
                                  sizeof(uint16_t) * MAX_DOC_SIZE * docs_len,
                                  cudaMemcpyHostToDevice, doc_streams[i]));
        std::chrono::high_resolution_clock::time_point dt1 = std::chrono::high_resolution_clock::now();
        std::cout << "cudaMemcpy H2D docs cost " << std::chrono::duration_cast<std::chrono::milliseconds>(dt1 - dt).count() << " ms " << std::endl;
    }
    std::chrono::high_resolution_clock::time_point ddt1 = std::chrono::high_resolution_clock::now();
    std::cout << "total cudaMemcpy H2D doc cost " << std::chrono::duration_cast<std::chrono::milliseconds>(ddt1 - ddt).count() << " ms " << std::endl;

    std::chrono::high_resolution_clock::time_point ddlt = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_STREAM; i++) {
        // last docs_chunk_size
        if (n_docs % N_STREAM > 0 && i == N_STREAM - 1) {
            docs_len = docs_chunk_size + n_docs % N_STREAM;
        }
        docs_offset = i * docs_chunk_size;
        std::chrono::high_resolution_clock::time_point dlt = std::chrono::high_resolution_clock::now();
        CUDA_CALL(cudaMemcpyAsync(d_doc_lens[i],
                                  h_doc_lens_vec + docs_offset,
                                  sizeof(int) * docs_len,
                                  cudaMemcpyHostToDevice, doc_streams[i]));
        std::chrono::high_resolution_clock::time_point dlt1 = std::chrono::high_resolution_clock::now();
        std::cout << "cudaMemcpy H2D doc_lens cost " << std::chrono::duration_cast<std::chrono::milliseconds>(dlt1 - dlt).count() << " ms " << std::endl;
    }
    std::chrono::high_resolution_clock::time_point ddlt1 = std::chrono::high_resolution_clock::now();
    std::cout << "total cudaMemcpy H2D doc_lens cost " << std::chrono::duration_cast<std::chrono::milliseconds>(ddlt1 - ddlt).count() << " ms " << std::endl;

    for (auto &query : querys) {
        const size_t query_len = query.size();
        // init query global memory
        uint16_t *d_query = nullptr;
        cudaMalloc(&d_query, sizeof(uint16_t) * query_len);
        cudaMemcpy(d_query, query.data(), sizeof(uint16_t) * query_len, cudaMemcpyHostToDevice);

        std::chrono::high_resolution_clock::time_point lt = std::chrono::high_resolution_clock::now();
        docs_len = docs_chunk_size;
        for (int i = 0; i < N_STREAM; i++) {
            // last docs_chunk_size
            if (n_docs % N_STREAM > 0 && i == N_STREAM - 1) {
                docs_len = docs_chunk_size + n_docs % N_STREAM;
            }
            docs_offset = i * docs_chunk_size;

            std::chrono::high_resolution_clock::time_point tt = std::chrono::high_resolution_clock::now();
            // launch kernel
            int block = N_THREADS_IN_ONE_BLOCK;
            int grid = (docs_len + block - 1) / block;
            docQueryScoringCoalescedMemoryAccessSampleKernel<<<grid, block, 0, doc_streams[i]>>>(
                d_docs[i], d_doc_lens[i], docs_len,
                d_query, query_len,
                d_scores[i]);

            std::chrono::high_resolution_clock::time_point tt1 = std::chrono::high_resolution_clock::now();
            std::cout << "docQueryScoringCoalescedMemoryAccessSampleKernel cost " << std::chrono::duration_cast<std::chrono::milliseconds>(tt1 - tt).count() << " ms " << std::endl;
        }
        std::chrono::high_resolution_clock::time_point lt2 = std::chrono::high_resolution_clock::now();
        std::cout << "total docQueryScoringCoalescedMemoryAccessSampleKernel cost " << std::chrono::duration_cast<std::chrono::milliseconds>(lt2 - lt).count() << " ms " << std::endl;

        std::chrono::high_resolution_clock::time_point st = std::chrono::high_resolution_clock::now();
        docs_len = docs_chunk_size;
        for (int i = 0; i < N_STREAM; i++) {
            // last docs_chunk_size
            if (n_docs % N_STREAM > 0 && i == N_STREAM - 1) {
                docs_len = docs_chunk_size + n_docs % N_STREAM;
            }
            docs_offset = i * docs_chunk_size;
            std::chrono::high_resolution_clock::time_point stt = std::chrono::high_resolution_clock::now();
            CUDA_CALL(cudaMemcpyAsync(s_scores + docs_offset,
                                      d_scores[i],
                                      sizeof(float) * docs_len, cudaMemcpyDeviceToHost, doc_streams[i]));
            std::chrono::high_resolution_clock::time_point stt1 = std::chrono::high_resolution_clock::now();
            std::cout << "cudaMemcpy D2H scores cost " << std::chrono::duration_cast<std::chrono::milliseconds>(stt1 - stt).count() << " ms " << std::endl;
        }
        std::chrono::high_resolution_clock::time_point st1 = std::chrono::high_resolution_clock::now();
        std::cout << "total cudaMemcpy D2H scores cost " << std::chrono::duration_cast<std::chrono::milliseconds>(st1 - st).count() << " ms " << std::endl;

        for (int i = 0; i < N_STREAM; i++) {
            CUDA_CALL(cudaStreamSynchronize(doc_streams[i]));
        }
        std::chrono::high_resolution_clock::time_point lt1 = std::chrono::high_resolution_clock::now();
        std::cout << "doc stream sync launch kernel cost "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(lt1 - lt).count() << " ms "
                  << std::endl;

        std::chrono::high_resolution_clock::time_point t = std::chrono::high_resolution_clock::now();
        int topk = n_docs > TOPK ? TOPK : n_docs;
        // sort scores with Heap-based sort
        std::partial_sort(s_indices.begin(), s_indices.begin() + topk, s_indices.end(),
                          [s_scores, start_doc_id](const int &a, const int &b) {
                              if (s_scores[a - start_doc_id] != s_scores[b - start_doc_id]) {
                                  return s_scores[a - start_doc_id] > s_scores[b - start_doc_id];  // by score DESC
                              }
                              return a < b;  // the same score, by index ASC
                          });
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
        std::cout << "heap partial_sort cost " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t).count() << " ms " << std::endl;

        std::vector<int> topk_doc_ids(s_indices.begin(), s_indices.begin() + topk);
        indices.push_back(topk_doc_ids);

        std::vector<float> topk_scores(topk_doc_ids.size());
        int i = 0;
        for (auto doc_id : topk_doc_ids) {
            topk_scores[i++] = s_scores[doc_id - start_doc_id];
        }
        scores.push_back(topk_scores);

        cudaFree(d_query);
    }

    cudaFree(d_docs);
    cudaFree(d_scores);
    cudaFree(d_doc_lens);
#ifdef PINNED_MEMORY
    cudaFreeHost(h_docs);
    cudaFreeHost(h_doc_lens_vec);
    cudaFreeHost(s_scores);
#else
    delete[] h_docs;
    delete[] h_doc_lens_vec;
    delete[] s_scores;
#endif

    for (int i = 0; i < N_STREAM; i++) {
        cudaStreamDestroy(doc_streams[i]);
    }
}
