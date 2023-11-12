#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#include "cuda_runtime.h"

/** Reserved value for indicating "empty". */
#define EMPTY_CELL (0)
/** CUDA naive thread block size. */
#define BLOCK_SIZE (256)

__inline__ __device__ int8_t atomicCAS(int8_t* address, int8_t compare, int8_t val) {
    int32_t* base_address = (int32_t*)((char*)address - ((size_t)address & 3));
    int32_t int_val = (int32_t)val << (((size_t)address & 3) * 8);
    int32_t int_comp = (int32_t)compare << (((size_t)address & 3) * 8);
    return (int8_t)atomicCAS(base_address, int_comp, int_val);
}

// TODO: can we do this more efficiently?
__inline__ __device__ int16_t atomicCAS(int16_t* address, int16_t compare, int16_t val) {
    int32_t* base_address = (int32_t*)((char*)address - ((size_t)address & 2));
    int32_t int_val = (int32_t)val << (((size_t)address & 2) * 8);
    int32_t int_comp = (int32_t)compare << (((size_t)address & 2) * 8);
    return (int16_t)atomicCAS(base_address, int_comp, int_val);
}

__inline__ __device__ int64_t atomicCAS(int64_t* address, int64_t compare, int64_t val) {
    return (int64_t)atomicCAS((unsigned long long*)address, (unsigned long long)compare,
                              (unsigned long long)val);
}

/*
__inline__ __device__ uint32_t atomicCAS(uint32_t* address, uint32_t compare, uint32_t val) {
    return (uint32_t)atomicCAS((unsigned int*)address, (unsigned int)compare,
                               (unsigned int)val);
}

// -arch=sm_70 below is for 16-bit atomicCAS
__inline__ __device__ uint16_t atomicCAS(uint16_t* address, uint16_t compare, uint16_t val) {
    return (uint16_t)atomicCAS((unsigned short*)address, (unsigned short)compare,
                               (unsigned short)val);
}
*/

template <typename dtype = int>
__device__ uint64_t hash_func_64b(dtype* data) {
    uint64_t hash = 14695981039346656037UL;
    for (int j = 0; j < 4; j++) {
        hash ^= (unsigned int)data[j];
        hash *= 1099511628211UL;
    }
    // hash = (hash >> 60) ^ (hash & 0xFFFFFFFFFFFFFFF);
    return hash;
}

template <typename key_type>
__device__ int hash(key_type key, int _capacity) {
    return (uint64_t)key % _capacity;
}

template <typename key_type>
__device__ int hash_murmur3(key_type key, int _capacity) {
    // use the murmur3 hash function for int32
    int64_t k = (int64_t)key;
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k % _capacity;
}

// val: cn (int)
// hashtable on gpu global memory off-chip
// -> shared memory on-chip???
template <typename key_type, typename val_type>
class GPUHashTable {
   private:
    bool free_pointers;
    const int _capacity;
    key_type* table_keys;
    val_type* table_vals;

   public:
    GPUHashTable(const int capacity)
        : _capacity(capacity), free_pointers(true) {
        srand(time(NULL));
        cudaMalloc((void**)&table_keys, _capacity * sizeof(key_type));
        cudaMemset(table_keys, 0, sizeof(key_type) * _capacity);
        cudaMalloc((void**)&table_vals, _capacity * sizeof(val_type));
        cudaMemset(table_vals, 0, sizeof(val_type) * _capacity);
    };
    ~GPUHashTable() {
        if (free_pointers) {
            cudaFree(table_keys);
            cudaFree(table_vals);
        }
    };
    void insert_unique_cn_many(const key_type* keys, const int n);
    void lookup_many(const key_type* keys, val_type* results, const int n);
    int get_capacity() { return _capacity; }
    class device_view {
       private:
        int _capacity;
        key_type* _table_keys;
        val_type* _table_vals;

       public:
        __host__ __device__ device_view(
            int capacity, key_type* table_keys, val_type* table_vals) : _capacity(capacity), _table_keys(table_keys), _table_vals(table_vals) {}
        __device__ val_type lookup(const key_type key) const;
        __device__ void insert(const key_type key, const val_type val);
    };
    __host__ __device__ device_view get_device_view() {
        return device_view(_capacity, table_keys, table_vals);
    }
};

using hashtable = GPUHashTable<int64_t, int>;
using hashtable32 = GPUHashTable<int, int>;
using hashtable16 = GPUHashTable<int16_t, int>;
using hashtable_u16 = GPUHashTable<uint16_t, int>;

// Insert into hashmap
template <typename key_type, typename val_type = int>
__global__ void insert_unique_cn_kernel(key_type* table_keys, val_type* table_vals, const key_type* keys, int n, int _capacity) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        key_type key = keys[idx];
        int slot = hash(key, _capacity);
        while (true) {
            key_type prev = atomicCAS(&table_keys[slot], EMPTY_CELL, key);
            if (prev == EMPTY_CELL) {
                // printf("insert_unique_cn_kernel idx:%d slot:%d key:%d \n",idx,slot,key);
                atomicAdd(&table_vals[slot], 1);
                return;
            } else if (prev == key) {
                // printf("insert_unique_cn_kernel idx:%d slot:%d key:%d more!\n",idx,slot,key);
                atomicAdd(&table_vals[slot], 1);
                return;
            }

            slot = (slot + 1) % _capacity;
        }
    }
}

// lookup from hashmap
template <typename key_type, typename val_type>
__global__ void lookup_kernel(key_type* table_keys, val_type* table_vals, const key_type* keys, val_type* vals, int n, int _capacity) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        key_type key = keys[idx];
        int slot = hash(key, _capacity);
        while (true) {
            key_type cur_key = table_keys[slot];
            if (key == cur_key) {
                vals[idx] = table_vals[slot];
                // printf("lookup_kernel idx:%d slot:%d key:%d val:%d \n",idx,slot,key,table_vals[slot]);
                return;
            }
            if (table_keys[slot] == EMPTY_CELL) {
                return;
            }
            slot = (slot + 1) % _capacity;
        }
    }
}

template <typename key_type, typename val_type>
void GPUHashTable<key_type, val_type>::insert_unique_cn_many(const key_type* keys, const int n) {
    insert_unique_cn_kernel<key_type, val_type><<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(table_keys, table_vals, keys, n, _capacity);
}

template <typename key_type, typename val_type>
void GPUHashTable<key_type, val_type>::lookup_many(const key_type* keys, val_type* results, const int n) {
    lookup_kernel<key_type, val_type><<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(table_keys, table_vals, keys, results, n, _capacity);
}

template <typename key_type, typename val_type>
__device__ void GPUHashTable<key_type, val_type>::device_view::insert(const key_type key, const val_type val) {
    int slot = hash(key, _capacity);
    while (true) {
        key_type prev = atomicCAS(&_table_keys[slot], EMPTY_CELL, key);
        if (prev == EMPTY_CELL || prev == key) {
            _table_vals[slot] = val;
            return;
        }
        slot = (slot + 1) % _capacity;
    }
}

template <typename key_type, typename val_type>
__device__ val_type GPUHashTable<key_type, val_type>::device_view::lookup(const key_type key) const {
    int slot = hash(key, _capacity);
    while (true) {
        key_type cur_key = _table_keys[slot];

        if (key == cur_key) {
            // printf("lookup ok slot:%d key:%d cur_key:%d val:%d \n", slot, key,cur_key,_table_vals[slot]);
            return _table_vals[slot];
        }
        if (_table_keys[slot] == EMPTY_CELL) {
            // printf("lookup empty slot:%d key:%d cur_key:%d \n", slot, key,cur_key);
            return EMPTY_CELL;
        }
        slot = (slot + 1) % _capacity;
    }
}