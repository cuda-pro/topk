#include "helper.h"
#include "readfile.h"

__global__ void docsKernel(cudf::column_device_view const d_docs, const size_t n_docs) {
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
        // auto const item = d_docs.element<cudf::string_view>(doc_id);
        auto offset_s = docs.offset_at(doc_id);
        auto offset_e = docs.offset_at(doc_id + 1);
        auto sub_view = docs.child().slice(offset_s, offset_e - offset_s);
        printf("\ntid:%d docid:%d s:%d e:%d sub_view_size:%d\n", tid, doc_id, offset_s, offset_e, sub_view.size());
        // if (doc_id == 0){
        for (auto i = 0; i < sub_view.size(); i++) {
            auto const item = sub_view.element<cudf::string_view>(i);
            int num = h_atoi(item.data());
            printf("%d,", num);
        }
        //}
    }
}

void get_file_cudf_chunk(std::string docs_file_name, std::vector<std::vector<uint16_t>> &docs, std::vector<uint16_t> &doc_lens) {
    unsigned int buffsize = CHUNK_SIZE;
    int count = 0;
    int readcnt = 0;
    unsigned int doccnt = 0;
    char *buff = new char[buffsize];

    FILE *fd = fopen(docs_file_name.c_str(), "rb");

    fseek(fd, 0, SEEK_END);
    std::cout << "file size: " << ftell(fd) << std::endl;
    std::cout << "chunk size: " << buffsize << std::endl;
    fseek(fd, 0, SEEK_SET);

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
        std::cout << " buffer: " << chunk_buff << std::endl;

        // cudf multibyte_split
        auto delimiter = "\n";
        cudf::io::text::parse_options options;
        options.strip_delimiters = true;
        auto source = cudf::io::text::make_source(chunk_buff);
        auto lines = cudf::io::text::multibyte_split(*source, delimiter, options);
        auto vec_lines = cudf::strings::split_record(lines->view(), cudf::string_scalar(","));
        auto const d_col = cudf::column_device_view::create(vec_lines->view());

        int block = N_THREADS_IN_ONE_BLOCK;
        int grid = (lines->size() + block - 1) / block;
        docsKernel<<<grid, block>>>(*d_col, lines->size());

        /*
        auto iter= vec_lines->view().child_begin();
        for (;iter!=vec_lines->view().child_end();iter++){

          std::vector<uint16_t> next_doc;
          auto iter_item =iter->child_begin();
          for(;iter_item!=iter->child_end();iter_item++){
            next_doc.emplace_back(1);
          }
          docs.emplace_back(next_doc);
          doc_lens.emplace_back(next_doc.size());

        }
         */
        doccnt += lines->size();

        readcnt++;
    }
    std::cout << "readcnt: " << readcnt << std::endl;
    std::cout << "doccnt: " << doccnt << std::endl;

    free(buff);
    fclose(fd);
}

void get_file_stream_cudf_chunk(std::string docs_file_name, std::vector<std::vector<uint16_t>> &docs, std::vector<uint16_t> &doc_lens) {
    unsigned int buffsize = CHUNK_SIZE;
    int count = 0;
    int readcnt = 0;
    char *buff = new char[buffsize];
    FILE *fd = fopen(docs_file_name.c_str(), "rb");

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

        // cudf multibyte_split
        cudf::io::text::parse_options options;
        options.strip_delimiters = true;
        auto source = cudf::io::text::make_source(chunk_buff);
        auto delimiter = "\n";
        auto doc_lines = cudf::io::text::multibyte_split(*source, delimiter, options);
    }

    free(buff);
    fclose(fd);
}
