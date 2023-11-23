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

/*
Note:
The host is automatically asynchronous with kernel launches
 Use streams to control asynchronous behavior
— Ordered within a stream (FIFO)
— Unordered with other streams
— Default stream is synchronous with all streams.

 Memory copies can execute concurrently if (and only if)
— The memory copy is in a different non-default stream
— The copy uses pinned memory on the host
— The asynchronous API is called
— There is not another memory copy occurring in the same direction at
the same time.

 Synchronization with the host can be accomplished via
— cudaDeviceSynchronize()
— cudaStreamSynchronize(stream)
— cudaEventSynchronize(event)
 Synchronization between streams can be accomplished with
— cudaStreamWaitEvent(stream,event)
 Use CUDA_LAUNCH_BLOCKING to identify race conditions
*/
void doc_query_scoring_gpu(std::vector<std::vector<uint16_t>>& querys,
                           int start_doc_id,
                           std::vector<std::vector<uint16_t>>& docs,
                           std::vector<uint16_t>& lens,
                           std::vector<std::vector<int>>& indices,  // shape [querys.size(), TOPK]
                           std::vector<std::vector<float>>& scores  // shape [querys.size(), TOPK]
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
    auto n_querys = querys.size();
    std::vector<int> s_scores(n_docs);
    std::vector<int> s_indices(n_docs);
    std::chrono::high_resolution_clock::time_point it = std::chrono::high_resolution_clock::now();
    // std::iota(s_indices.begin(), s_indices.end(), start_doc_id);
    // #pragma omp parallel for schedule(static)
    for (int j = 0; j < n_docs; ++j) {
        s_indices[j] = j + start_doc_id;
    }
    std::chrono::high_resolution_clock::time_point it1 = std::chrono::high_resolution_clock::now();
    std::cout << "iota indeices cost " << std::chrono::duration_cast<std::chrono::milliseconds>(it1 - it).count() << " ms " << std::endl;

    int block = N_THREADS_IN_ONE_BLOCK;
    int grid = (n_docs + block - 1) / block;

    float* d_querys_scores = nullptr;
    uint16_t *d_docs = nullptr, *d_querys = nullptr;
    int* d_doc_lens = nullptr;

    // init device global memory
    std::chrono::high_resolution_clock::time_point dat = std::chrono::high_resolution_clock::now();
    cudaMalloc(&d_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs);
    cudaMalloc(&d_doc_lens, sizeof(int) * n_docs);
    cudaMalloc(&d_querys, sizeof(uint16_t) * MAX_DOC_SIZE * n_querys);
    cudaMalloc(&d_querys_scores, sizeof(float) * n_docs * n_querys);
    std::chrono::high_resolution_clock::time_point dat1 = std::chrono::high_resolution_clock::now();
    std::cout << "cudaMalloc docs cost " << std::chrono::duration_cast<std::chrono::milliseconds>(dat1 - dat).count() << " ms " << std::endl;

    // pre align docs -> h_docs [n_docs,MAX_DOC_SIZE], h_doc_lens_vec[n_docs]
    // std::chrono::high_resolution_clock::time_point dgt = std::chrono::high_resolution_clock::now();
    // uint16_t* h_docs = new uint16_t[MAX_DOC_SIZE * n_docs];
    // memset(h_docs, 0, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs);
    std::chrono::high_resolution_clock::time_point dgt = std::chrono::high_resolution_clock::now();
    uint16_t* h_docs = nullptr;
    // h_docs= new uint16_t[MAX_DOC_SIZE * n_docs];
    // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gb65da58f444e7230d3322b6126bb4902
    // cudaMallocHost(&h_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs);  // cudaHostAllocDefault
    cudaHostAlloc(&h_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs, cudaHostAllocDefault);
    cudaMemset(h_docs, 0, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs);
    int* h_doc_lens_vec = nullptr;
    cudaHostAlloc(&h_doc_lens_vec, sizeof(int) * n_docs, cudaHostAllocDefault);
    // cudaMemset(h_doc_lens, 0, sizeof(int) * n_docs);
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

    std::chrono::high_resolution_clock::time_point dt = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_docs, h_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs, cudaMemcpyHostToDevice);
    std::chrono::high_resolution_clock::time_point dt1 = std::chrono::high_resolution_clock::now();
    std::cout << "cudaMemcpy H2D docs cost " << std::chrono::duration_cast<std::chrono::milliseconds>(dt1 - dt).count() << " ms " << std::endl;

    std::chrono::high_resolution_clock::time_point dlt = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_doc_lens, h_doc_lens_vec, sizeof(int) * n_docs, cudaMemcpyHostToDevice);
    std::chrono::high_resolution_clock::time_point dlt1 = std::chrono::high_resolution_clock::now();
    std::cout << "cudaMemcpy H2D doc_lens cost " << std::chrono::duration_cast<std::chrono::milliseconds>(dlt1 - dlt).count() << " ms " << std::endl;

    // query stream
    cudaStream_t* q_streams = (cudaStream_t*)malloc(n_querys * sizeof(cudaStream_t));
    for (int i = 0; i < n_querys; i++) {
        cudaStreamCreateWithFlags(&q_streams[i], cudaStreamNonBlocking);
    }

    std::chrono::high_resolution_clock::time_point lt = std::chrono::high_resolution_clock::now();
    // query documents scores
    // std::vector<std::vector<float>> q_scores(n_querys);
    // use cudaMallocHost to allocate pinned memory
    float* q_scores = nullptr;
    cudaHostAlloc(&q_scores, sizeof(float) * n_docs * n_querys, cudaHostAllocDefault);
    for (int stream_id = 0; stream_id < n_querys; stream_id++) {
        const size_t query_len = querys[stream_id].size();
        cudaMemcpyAsync(d_querys + stream_id * MAX_DOC_SIZE, querys[stream_id].data(), sizeof(uint16_t) * query_len, cudaMemcpyHostToDevice, q_streams[stream_id]);
        // launch kernel
        docQueryScoringCoalescedMemoryAccessSampleKernel<<<grid, block, 0, q_streams[stream_id]>>>(
            d_docs, d_doc_lens, n_docs, d_querys + stream_id * MAX_DOC_SIZE, query_len, d_querys_scores + stream_id * n_docs);
        // std::vector<float> s_scores(n_docs);
        cudaMemcpyAsync(q_scores + stream_id * n_docs, d_querys_scores + stream_id * n_docs, sizeof(float) * n_docs, cudaMemcpyDeviceToHost, q_streams[stream_id]);
        // q_scores[stream_id] = std::move(s_scores);

        // std::cout << "stream_id:  " << stream_id << " scores size:" << q_scores[stream_id].size() << std::endl;
    }
    std::chrono::high_resolution_clock::time_point lt2 = std::chrono::high_resolution_clock::now();
    std::cout << "async cost "
              << std::chrono::duration_cast<std::chrono::milliseconds>(lt2 - lt).count() << " ms "
              << std::endl;

    for (int i = 0; i < n_querys; i++) {
        cudaStreamSynchronize(q_streams[i]);
    }
    std::chrono::high_resolution_clock::time_point lt1 = std::chrono::high_resolution_clock::now();
    std::cout << "stream sync launch kernel cost "
              << std::chrono::duration_cast<std::chrono::milliseconds>(lt1 - lt).count() << " ms "
              << std::endl;

    std::chrono::high_resolution_clock::time_point t = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_querys; i++) {
        std::chrono::high_resolution_clock::time_point tt = std::chrono::high_resolution_clock::now();
        // auto s_scores = std::move(q_scores[i]);
        // std::cout << "scores size:" << q_scores[i].size() << std::endl;
        int topk = n_docs > TOPK ? TOPK : n_docs;
        // sort scores
        std::partial_sort(s_indices.begin(),
                          s_indices.begin() + topk,
                          s_indices.end(),
                          [&q_scores, i, start_doc_id](const int& a, const int& b) {
                              if (q_scores[i][a - start_doc_id] != q_scores[i][b - start_doc_id]) {
                                  return q_scores[i][a - start_doc_id] > q_scores[i][b - start_doc_id];
                              }
                              return a < b;
                          });

        std::vector<int> topk_doc_ids(s_indices.begin(), s_indices.begin() + topk);
        indices.push_back(topk_doc_ids);
        std::chrono::high_resolution_clock::time_point tt1 = std::chrono::high_resolution_clock::now();
        std::cout << "partial_sort cost "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(tt1 - tt).count() << " ms "
                  << std::endl;

        std::vector<float> topk_scores(topk_doc_ids.size());
        int i = 0;
        for (auto doc_id : topk_doc_ids) {
            topk_scores[i++] = q_scores[i][doc_id - start_doc_id];
        }
        scores.emplace_back(topk_scores);
    }
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    std::cout << "total partial_sort cost "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t).count() << " ms "
              << std::endl;

    cudaFree(d_docs);
    cudaFree(d_doc_lens);
    cudaFree(d_querys);
    cudaFree(d_querys_scores);
    // delete[] h_docs;
    cudaFreeHost(h_docs);
    cudaFreeHost(h_doc_lens_vec);
    cudaFreeHost(q_scores);

    for (int i = 0; i < n_querys; i++) {
        cudaStreamDestroy(q_streams[i]);
    }
    free(q_streams);
}
