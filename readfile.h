#pragma once

#include <cuda.h>
#include <string.h>

#include <chrono>
#include <cudf/column/column_device_view.cuh>
#include <cudf/io/text/data_chunk_source_factories.hpp>
#include <cudf/io/text/multibyte_split.hpp>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/strings/split/split.hpp>
#include <cudf/types.hpp>
#include <fstream>
#include <iostream>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
#ifdef PIO_CPU_CONCURRENCY
#include "threadpool.h"
#endif

#define CHUNK_SIZE 1024 * 1024 * 256

void load_file_cudf_chunk(std::string docs_file_name, std::vector<std::vector<uint16_t>> &docs, std::vector<uint16_t> &doc_lens);

void load_file_stream_cudf_chunk(std::string docs_file_name, std::vector<std::vector<uint16_t>> &docs, std::vector<uint16_t> &doc_lens);

void load_file_cudf_chunk_topk(const std::string docs_file_name,
                               std::vector<std::vector<uint16_t>> &queries,
                               std::vector<std::vector<int>> &indices,
                               std::vector<std::vector<float>> &scores);