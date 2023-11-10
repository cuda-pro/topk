// g++ readfile.cpp -o bin/readfile --std=c++11 -O3
// nvcc readfile.cpp -o readfile -O3 --std=c++17 -I./ -I/include -L/lib -lcudf -DGPU -DFMT_HEADER_ONLY

#include <string.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef GPU
#include "readfile.h"
#endif

#define BUFFER_SIZE 1024 * 1024 * 512

void load_file_line(std::string docs_file_name, std::vector<std::vector<uint16_t>> &docs, std::vector<uint16_t> &doc_lens) {
    std::stringstream ss;
    std::string tmp_str;
    std::string tmp_index_str;
    std::ifstream docs_file(docs_file_name);
    while (std::getline(docs_file, tmp_str)) {
        std::vector<uint16_t> next_doc;
        ss.clear();
        ss << tmp_str;
        while (std::getline(ss, tmp_index_str, ',')) {
            next_doc.emplace_back(std::stoi(tmp_index_str));
        }
        if (next_doc.size() > 0) {
            docs.emplace_back(next_doc);
            doc_lens.emplace_back(next_doc.size());
        }
    }
    docs_file.close();
    ss.clear();
}

void load_file_buffer(std::string docs_file_name, std::vector<std::vector<uint16_t>> &docs, std::vector<uint16_t> &doc_lens) {
    std::stringstream sl;
    std::stringstream ss;
    std::string tmp_str;
    std::string tmp_index_str;
    unsigned int buffsize = BUFFER_SIZE;
    unsigned int count = 0;
    int readcnt = 0;
    char *buff = new char[buffsize];
    FILE *fd = fopen(docs_file_name.c_str(), "rb");

    while (!feof(fd)) {
        memset(buff, 0, buffsize);
        count += fread(buff, sizeof(char), buffsize, fd);
        auto cur_pos = ftell(fd);
        std::string tmp(buff);
        auto offset = tmp.find_last_of("\n");
        if (!feof(fd) && offset != std::string::npos) {
            tmp.erase(offset + 1);
            fseek(fd, cur_pos - (buffsize - offset) + 1, SEEK_SET);
        }
        sl.clear();
        sl << tmp;
        while (std::getline(sl, tmp_str)) {
            std::vector<uint16_t> next_doc;
            ss.clear();
            ss << tmp_str;
            while (std::getline(ss, tmp_index_str, ',')) {
                next_doc.emplace_back(std::stoi(tmp_index_str));
            }
            if (next_doc.size() > 0) {
                docs.emplace_back(next_doc);
                doc_lens.emplace_back(next_doc.size());
            }
        }
        readcnt++;
    }
    std::cout << "readcnt: " << readcnt << " fread size: " << count << std::endl;

    free(buff);
    fclose(fd);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cout << "params need <docs_file_path> <use_method [line|buffer|chunk]>" << std::endl;
        return -1;
    }
    std::string file(argv[1]);
    std::string method(argv[2]);
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<uint16_t>> docs;
    std::vector<uint16_t> doc_lens;
    if (method == "line") {
        load_file_line(file, docs, doc_lens);
    } else if (method == "buffer") {
        load_file_buffer(file, docs, doc_lens);
    } else {
#ifdef GPU
        load_file_cudf_chunk(file, docs, doc_lens);
#else
        load_file_buffer(file, docs, doc_lens);
#endif
    }
    std::cout << "docs_size:" << docs.size() << " doc_lens_size:" << doc_lens.size() << std::endl;
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::cout << "read file cost " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms " << std::endl;
    return 0;
}