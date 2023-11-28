#include "helper.h"
#include "readfile.h"
#include "topk.h"

__global__ void docsKernel(cudf::column_device_view const d_docs, const size_t n_docs, uint16_t *out_docs, uint16_t *out_doc_len) {
    // each thread process one doc-query pair scoring task
    register auto tid = blockIdx.x * blockDim.x + threadIdx.x, tnum = gridDim.x * blockDim.x;

#ifdef DEBUG
    printf("tid:%d tnum:%d GPU from block(%d, %d, %d), thread(%d, %d, %d)\n ",
           tid, tnum,
           blockIdx.x,
           blockIdx.y, blockIdx.z,
           threadIdx.x, threadIdx.y, threadIdx.z);
#endif
    if (tid >= n_docs) {
        return;
    }
    auto docs = cudf::detail::lists_column_device_view(d_docs);
    for (auto doc_id = tid; doc_id < n_docs; doc_id += tnum) {
        auto offset_s = docs.offset_at(doc_id);
        auto offset_e = docs.offset_at(doc_id + 1);
        auto sub_view = docs.child().slice(offset_s, offset_e - offset_s);
        // printf("\ntid:%d docid:%d s:%d e:%d sub_view_size:%d\n", tid, doc_id, offset_s, offset_e, sub_view.size());
        // if (doc_id == 2){
        for (auto i = 0; i < sub_view.size(); i++) {
            auto const item = sub_view.element<cudf::string_view>(i);
            int num = h_atoi(item.data());
            // printf("%d,", num);
            out_docs[doc_id * MAX_DOC_SIZE + i] = num;
        }
        //}
        out_doc_len[doc_id] = sub_view.size();
    }
}

void load_file_cudf_chunk(std::string docs_file_name, std::vector<std::vector<uint16_t>> &docs, std::vector<uint16_t> &doc_lens) {
    unsigned int buffsize = CHUNK_SIZE;
    int count = 0;
    int readcnt = 0;
    unsigned int doccnt = 0;
    char *buff = new char[buffsize];

    FILE *fd = fopen(docs_file_name.c_str(), "rb");
    // fseek(fd, 0, SEEK_END);
    // std::cout << "file size: " << ftell(fd) << std::endl;
    // fseek(fd, 0, SEEK_SET);
    std::cout << "chunk size: " << buffsize << std::endl;

    while (!feof(fd)) {
        memset(buff, 0, buffsize);
        count = fread(buff, sizeof(char), buffsize, fd);
        auto cur_pos = ftell(fd);
        std::string chunk_buff(buff);
        auto offset = chunk_buff.find_last_of("\n");
        if (!feof(fd) && offset != std::string::npos) {
            chunk_buff.erase(offset + 1);
            fseek(fd, cur_pos - (buffsize - offset) + 1, SEEK_SET);
        }
        std::cout << " fread size: " << count << std::endl;
        // std::cout << " buffer: " << chunk_buff << std::endl;

        cudaStream_t stream;
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        rmm::cuda_stream_view stream_view(stream);
        // cudf multibyte_split
        auto delimiter = "\n";
        cudf::io::text::parse_options options;
        options.strip_delimiters = false;
        auto source = cudf::io::text::make_source(chunk_buff);
        auto lines = cudf::io::text::multibyte_split(*source, delimiter, options, stream_view);
        auto vec_lines = cudf::strings::split_record(lines->view(), cudf::string_scalar(","), -1, stream_view);
        auto const d_col = cudf::column_device_view::create(vec_lines->view());

        auto n_docs = lines->size();
        uint16_t *d_docs = nullptr;
        uint16_t *d_doc_lens = nullptr;
        cudaMalloc(&d_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs);
        cudaMemset(d_docs, 0, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs);
        cudaMalloc(&d_doc_lens, sizeof(uint16_t) * n_docs);
        // cudaMemset(d_doc_lens, 0, sizeof(uint16_t) * n_docs);

        int block = N_THREADS_IN_ONE_BLOCK;
        int grid = (n_docs + block - 1) / block;
        docsKernel<<<grid, block, 0, stream_view.value()>>>(*d_col, n_docs, d_docs, d_doc_lens);

        uint16_t *h_docs = nullptr;
        uint16_t *h_doc_lens = nullptr;
#ifdef PINNED_MEMORY
        // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gb65da58f444e7230d3322b6126bb4902
        cudaMallocHost(&h_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs);  // cudaHostAllocDefault
        // cudaHostAlloc(&h_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs, cudaHostAllocDefault);
        cudaHostAlloc(&h_doc_lens, sizeof(int) * n_docs, cudaHostAllocDefault);
        // cudaHostAlloc(&h_doc_offsets_vec, sizeof(int) * n_docs, cudaHostAllocDefault);
#else
        h_docs = new uint16_t[MAX_DOC_SIZE * n_docs];
        h_doc_lens = new uint16_t[n_docs];
#endif
        // memset(h_docs, 0, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs);
        // memset(h_doc_lens, 0, sizeof(uint16_t) * n_docs);
        cudaMemcpyAsync(h_docs, d_docs, sizeof(uint16_t) * n_docs * MAX_DOC_SIZE, cudaMemcpyDeviceToHost, stream_view.value());
        cudaMemcpyAsync(h_doc_lens, d_doc_lens, sizeof(uint16_t) * n_docs, cudaMemcpyDeviceToHost, stream_view.value());
        cudaStreamSynchronize(stream_view.value());

#ifdef DEBUG
        std::cout << "h_docs:" << std::endl;
        print2d_uint16(h_docs, n_docs, MAX_DOC_SIZE);
        std::cout << "h_doc_lens:" << std::endl;
        print1d_uint16(h_doc_lens, n_docs);
#endif

        for (int i = 0; i < n_docs; i++) {
            std::vector<uint16_t> vec_docs;
            vec_docs.reserve(h_doc_lens[i]);
            vec_docs.insert(vec_docs.end(), h_docs + i * MAX_DOC_SIZE, h_docs + i * MAX_DOC_SIZE + h_doc_lens[i]);
            docs.emplace_back(vec_docs);
        }
        doc_lens.insert(doc_lens.end(), h_doc_lens, h_doc_lens + n_docs);

        cudaStreamDestroy(stream_view.value());
        cudaFree(d_docs);
        cudaFree(d_doc_lens);
#ifdef PINNED_MEMORY
        cudaFreeHost(h_docs);
        cudaFreeHost(h_doc_lens);
#else
        delete[] h_docs;
        delete[] h_doc_lens;
#endif

        doccnt += n_docs;
        readcnt++;
    }
    std::cout << "readcnt: " << readcnt << std::endl;
    std::cout << "doccnt: " << doccnt << std::endl;
#ifdef DEBUG
    std::cout << "docs:" << std::endl;
    for (auto doc : docs) {
        print(doc);
    }
    std::cout << "doc_lens:" << std::endl;
    print(doc_lens);
#endif

    free(buff);
    fclose(fd);
}

void load_file_stream_cudf_chunk(std::string docs_file_name, std::vector<std::vector<uint16_t>> &docs, std::vector<uint16_t> &doc_lens) {
    unsigned int buffsize = CHUNK_SIZE;
    int count = 0;
    int readcnt = 0;
    unsigned int doccnt = 0;
    char *buff = new char[buffsize];
    FILE *fd = fopen(docs_file_name.c_str(), "rb");
    fseek(fd, 0, SEEK_END);
    auto file_size = ftell(fd);
    std::cout << "file size: " << file_size << std::endl;
    std::cout << "chunk size: " << buffsize << std::endl;
    fseek(fd, 0, SEEK_SET);

    auto split_cn = (file_size + buffsize - 1) / buffsize;
    cudaStream_t *doc_streams = new cudaStream_t[split_cn];
    std::cout << " Creating " << split_cn << " CUDA streams." << std::endl;
    for (int i = 0; i < split_cn; i++) {
        CUDA_CALL(cudaStreamCreate(&doc_streams[i]));
    }

    while (!feof(fd)) {
        memset(buff, 0, buffsize);
        count = fread(buff, sizeof(char), buffsize, fd);
        auto cur_pos = ftell(fd);
        std::string chunk_buff(buff);
        auto offset = chunk_buff.find_last_of("\n");
        if (!feof(fd) && offset != std::string::npos) {
            chunk_buff.erase(offset + 1);
            fseek(fd, cur_pos - (buffsize - offset) + 1, SEEK_SET);
        }
        std::cout << " fread size: " << count << std::endl;
        // std::cout << " buffer: " << chunk_buff << std::endl;

        // cudf multibyte_split
        auto delimiter = "\n";
        cudf::io::text::parse_options options;
        options.strip_delimiters = false;
        auto source = cudf::io::text::make_source(chunk_buff);
        // todo:: need multibyte_split support stream
        auto lines = cudf::io::text::multibyte_split(*source, delimiter, options, cudf::get_default_stream());
        auto vec_lines = cudf::strings::split_record(lines->view(), cudf::string_scalar(","));
        auto const d_col = cudf::column_device_view::create(vec_lines->view());
        // todo: launch docsKernel with stream

        doccnt += lines->size();
        readcnt++;
    }
    std::cout << "readcnt: " << readcnt << std::endl;
    std::cout << "doccnt: " << doccnt << std::endl;

    free(doc_streams);
    free(buff);
    fclose(fd);
}

#ifdef PIO_TOPK
typedef std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<float>>> tupleIdScores;

void load_file_cudf_chunk_topk(const std::string docs_file_name,
                               std::vector<std::vector<uint16_t>> &queries,
                               std::vector<std::vector<int>> &indices,
                               std::vector<std::vector<float>> &scores) {
    std::vector<std::vector<int>> q_indices(queries.size());
    std::vector<std::vector<float>> q_scores(queries.size());

    unsigned int buffsize = CHUNK_SIZE;
    int count = 0;
    int readcnt = 0;
    unsigned int doccnt = 0;
    char *buff = new char[buffsize];

    FILE *fd = fopen(docs_file_name.c_str(), "rb");
    // fseek(fd, 0, SEEK_END);
    // std::cout << "file size: " << ftell(fd) << std::endl;
    // fseek(fd, 0, SEEK_SET);
    std::cout << "chunk size: " << buffsize << std::endl;

#ifdef PIO_CPU_CONCURRENCY
    int concurrency = std::thread::hardware_concurrency();
    std::cout << "hardware concurrency:" << concurrency << std::endl;
    ThreadPool pool(concurrency);
    std::vector<std::future<tupleIdScores>> results;
#endif

    while (!feof(fd)) {
        memset(buff, 0, buffsize);
        count = fread(buff, sizeof(char), buffsize, fd);
        auto cur_pos = ftell(fd);
        std::string chunk_buff(buff);
        auto offset = chunk_buff.find_last_of("\n");
        if (!feof(fd) && offset != std::string::npos) {
            chunk_buff.erase(offset + 1);
            fseek(fd, cur_pos - (buffsize - offset) + 1, SEEK_SET);
        }
        std::cout << " fread size: " << count << std::endl;
        // std::cout << " buffer: " << chunk_buff << std::endl;

        cudaStream_t stream;
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        rmm::cuda_stream_view stream_view(stream);
        // cudf multibyte_split
        auto delimiter = "\n";
        cudf::io::text::parse_options options;
        options.strip_delimiters = false;
        auto source = cudf::io::text::make_source(chunk_buff);
        auto lines = cudf::io::text::multibyte_split(*source, delimiter, options, stream_view);
        auto vec_lines = cudf::strings::split_record(lines->view(), cudf::string_scalar(","), -1, stream_view);
        auto d_col = cudf::column_device_view::create(vec_lines->view());
        auto n_docs = lines->size();

#ifdef PIO_CPU_CONCURRENCY
        // std::unique_ptr https://github.com/progschj/ThreadPool/issues/93 lambda maybe always upgrade
        auto f = [&queries, doccnt, n_docs, col = std::move(d_col), &stream_view]() mutable {
            std::vector<std::vector<int>> sub_topk_indices;
            std::vector<std::vector<float>> sub_topk_scores;
            doc_query_scoring_gpu(queries, doccnt, n_docs, std::move(*col), sub_topk_indices, sub_topk_scores, stream_view);
            cudaStreamDestroy(stream_view.value());
            return std::make_tuple(sub_topk_indices, sub_topk_scores);
        };
        results.emplace_back(pool.enqueue(std::move(f)));
#else
        std::vector<std::vector<int>> sub_topk_indices;
        std::vector<std::vector<float>> sub_topk_scores;
        doc_query_scoring_gpu(queries, doccnt, n_docs, *d_col, sub_topk_indices, sub_topk_scores, stream_view);
        for (auto i = 0; i < queries.size(); i++) {
            q_indices[i].insert(q_indices[i].end(), sub_topk_indices[i].begin(), sub_topk_indices[i].end());
            q_scores[i].insert(q_scores[i].end(), sub_topk_scores[i].begin(), sub_topk_scores[i].end());
        }
        cudaStreamDestroy(stream_view.value());
#endif

        doccnt += n_docs;
        readcnt++;
    }
    std::cout << "readcnt: " << readcnt << std::endl;
    std::cout << "doccnt: " << doccnt << std::endl;

#ifdef PIO_CPU_CONCURRENCY
    for (auto &&result : results) {
        auto res = result.get();
        for (auto i = 0; i < queries.size(); i++) {
            q_indices[i].insert(q_indices[i].end(), std::get<0>(res)[i].begin(), std::get<0>(res)[i].end());
            q_scores[i].insert(q_scores[i].end(), std::get<1>(res)[i].begin(), std::get<1>(res)[i].end());
        }
    }
#endif

    // sort topk
    for (auto i = 0; i < queries.size(); i++) {
        std::unordered_map<int, int> indices_map;
        for (auto j = 0; j < q_indices[i].size(); j++) {
            indices_map[q_indices[i][j]] = j;
        }
        int topk = q_indices[i].size() > TOPK ? TOPK : q_indices[i].size();
        std::partial_sort(q_indices[i].begin(), q_indices[i].begin() + topk, q_indices[i].end(),
                          [&q_scores, i, &indices_map](const int &a, const int &b) {
                              if (q_scores[i][indices_map[a]] != q_scores[i][indices_map[b]]) {
                                  return q_scores[i][indices_map[a]] > q_scores[i][indices_map[b]];  // by score DESC
                              }
                              return a < b;  // the same score, by index ASC
                          });

        std::vector<int> topk_doc_ids(q_indices[i].begin(), q_indices[i].begin() + topk);
        indices.emplace_back(topk_doc_ids);

        std::vector<float> topk_scores(topk_doc_ids.size());
        int id = 0;
        for (auto doc_id : topk_doc_ids) {
            topk_scores[id++] = q_scores[i][indices_map[doc_id]];
        }
        scores.emplace_back(topk_scores);
    }

    free(buff);
    fclose(fd);
}
#endif