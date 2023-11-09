#pragma once

#include <cuda.h>
#include <string.h>

#include <chrono>
#include <cudf/column/column_device_view.cuh>
#include <cudf/io/text/data_chunk_source_factories.hpp>
#include <cudf/io/text/multibyte_split.hpp>
#include <cudf/strings/split/split.hpp>
#include <cudf/types.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define CHUNK_SIZE 1024 * 1024 * 256
#define N_THREADS_IN_ONE_BLOCK 256

void get_file_cudf_chunk(std::string docs_file_name, std::vector<std::vector<uint16_t>> &docs, std::vector<uint16_t> &doc_lens);

void get_file_stream_cudf_chunk(std::string docs_file_name, std::vector<std::vector<uint16_t>> &docs, std::vector<uint16_t> &doc_lens);