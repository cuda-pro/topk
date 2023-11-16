#pragma once

#include <ctype.h>
#include <dirent.h>
#include <omp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/stat.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#ifdef __CUDACC__
#include <cuda.h>
#define CUDF_HOST_DEVICE __host__ __device__
#else
#define CUDF_HOST_DEVICE
#endif

#ifdef __CUDACC__
#define CUDA_CALL(F)                                                          \
    if ((F) != cudaSuccess) {                                                 \
        printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
               __FILE__, __LINE__);                                           \
        exit(-1);                                                             \
    }
#define CUDA_CHECK()                                                          \
    if ((cudaPeekAtLastError()) != cudaSuccess) {                             \
        printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
               __FILE__, __LINE__ - 1);                                       \
        exit(-1);                                                             \
    }
#else
#define CUDA_CALL(F) (F)
#define CUDA_CHECK()
#endif

CUDF_HOST_DEVICE static int h_atoi(const char* src) {
    int s = 0;
    bool isMinus = false;

    while (*src == ' ') {
        src++;
    }

    if (*src == '+' || *src == '-') {
        if (*src == '-') {
            isMinus = true;
        }
        src++;
    } else if (*src < '0' || *src > '9') {
        s = 2147483647;
        return s;
    }

    while (*src != '\0' && *src >= '0' && *src <= '9') {
        s = s * 10 + *src - '0';
        src++;
    }
    return s * (isMinus ? -1 : 1);
}

CUDF_HOST_DEVICE static int h_itoa(int n, char s[]) {
    int i, j, sign;
    sign = n;
    if (sign < 0) {
        n = -n;
    }
    i = 0;
    do {
        s[i++] = n % 10 + '0';
    } while ((n /= 10) > 0);
    if (sign < 0) {
        s[i] = '-';
    }
    for (j = 0; j < (i + 1) / 2; j++) {
        char tmp = s[j];
        s[j] = s[i - j];
        s[i - j] = tmp;
    }
    s[i + 1] = '\0';
    return 0;
}

static size_t get_file_size(const char* fileName) {
    if (fileName == NULL) {
        return 0;
    }

    struct stat statbuf;
    stat(fileName, &statbuf);
    size_t filesize = statbuf.st_size;

    return filesize;
}
static std::vector<std::string> getFilesInDirectory(const std::string& directory) {
    std::vector<std::string> files;
    DIR* dirp = opendir(directory.c_str());
    struct dirent* dp;
    while ((dp = readdir(dirp)) != NULL) {
        struct stat path_stat;
        stat((directory + "/" + dp->d_name).c_str(), &path_stat);
        if (S_ISREG(path_stat.st_mode))  // Check if it's a regular file - not a directory
            files.push_back(dp->d_name);
    }
    closedir(dirp);
    std::sort(files.begin(), files.end());  // sort the files
    return files;
}

static void print1d_uint16(uint16_t* data, size_t col_cn) {
    for (int i = 0; i < col_cn; i++) {
        printf("%d ", data[i]);
        // printf("%d,", *(data+ i));
    }
    printf("\n");
}

static void print2d_uint16(uint16_t* data, size_t row_cn, size_t col_cn) {
    for (int i = 0; i < row_cn; i++) {
        for (int j = 0; j < col_cn; j++) {
            // data[i][j];
            printf("%d,", *(data + i * col_cn + j));
        }
        printf("\n");
    }
}

template <typename T>
static void print(std::vector<T> const& v) {
    for (auto i : v) {
        std::cout << i << ' ';
    }
    std::cout << std::endl;
}

template <typename T>
static std::vector<T> sub_vec_v1(std::vector<T> const& v, int m, int n) {
    auto first = v.cbegin() + m;
    auto last = v.cbegin() + n + 1;

    std::vector<T> vec(first, last);
    return vec;
}
template <typename T>
static std::vector<T> sub_vec(std::vector<T>& v, int m, int n) {
    std::vector<T> vec;
    std::copy(v.begin() + m, v.begin() + n + 1, std::back_inserter(vec));
    return vec;
}

template <typename KeyT, typename IdxT>
static auto topk_sort_permutation(const std::vector<KeyT>& vec,
                                  const std::vector<IdxT>& inds,
                                  uint32_t k,
                                  bool select_min) -> std::vector<IdxT> {
    std::vector<IdxT> p(vec.size());
    std::iota(p.begin(), p.end(), 0);
    if (select_min) {
        std::sort(p.begin(), p.end(), [&vec, &inds, k](IdxT i, IdxT j) {
            const IdxT ik = i / k;
            const IdxT jk = j / k;
            if (ik == jk) {
                if (vec[i] == vec[j]) {
                    return inds[i] < inds[j];
                }
                return vec[i] < vec[j];
            }
            return ik < jk;
        });
    } else {
        std::sort(p.begin(), p.end(), [&vec, &inds, k](IdxT i, IdxT j) {
            const IdxT ik = i / k;
            const IdxT jk = j / k;
            if (ik == jk) {
                if (vec[i] == vec[j]) {
                    return inds[i] < inds[j];
                }
                return vec[i] > vec[j];
            }
            return ik < jk;
        });
    }
    return p;
}

template <typename KeyT, typename IdxT>
static void apply_permutation(std::vector<KeyT>& vec, const std::vector<IdxT>& p) {
    for (auto i = IdxT(vec.size()) - 1; i > 0; i--) {
        auto j = p[i];
        while (j > i)
            j = p[j];
        std::swap(vec[j], vec[i]);
    }
}

#ifdef __CUDACC__
static int show_mem_usage() {
    // show memory usage of GPU
    size_t free_byte;
    size_t total_byte;
    CUDA_CALL(cudaMemGetInfo(&free_byte, &total_byte));
    size_t used_byte = total_byte - free_byte;
    printf("GPU memory usage: used = %4.2lf MB, free = %4.2lf MB, total = %4.2lf MB\n",
           used_byte / 1024.0 / 1024.0, free_byte / 1024.0 / 1024.0, total_byte / 1024.0 / 1024.0);
    return cudaSuccess;
}

static int getThreadNum() {
    cudaDeviceProp prop;
    int count;

    CUDA_CALL(cudaGetDeviceCount(&count));
    printf("gpu num %d\n", count);
    CUDA_CALL(cudaGetDeviceProperties(&prop, 0));
    printf("max thread num: %d\n", prop.maxThreadsPerBlock);
    printf("max grid dimensions: %d, %d, %d)\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    return prop.maxThreadsPerBlock;
}
#endif

static int g_ncore = omp_get_num_procs();

static int dtn(int n, int min_n) {
    int max_tn = n / min_n;
    int tn = max_tn > g_ncore ? g_ncore : max_tn;
    if (tn < 1) {
        tn = 1;
    }
    return tn;
}