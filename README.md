# task
Given 8.5 million big data files, each data is an integer id vector of up to 128 dimensions (called doc), and the id value range is 0-50000. 
Given a integer id vector of up to 128 dimensions (called query), the data set can be spread for optimization

```shell
# Generate test data, has been sorted in ascending order, the default docs file counts one document per line,10 documents; 10 query files
make gen
```
Find the average score topk (k=100) of the number of data intersections in query and doc; Here we define the intersection fraction of item as:
query[i] == doc[j] (0<=i<query_size, 0<=j<doc_size) calculates an intersection, the average number of query and doc intersections /max(query_size,doc_size)

``` shell
./bin/query_doc_scoring <doc_file_name> <query_file_name> <output_filename>
```

# optimize
note: just optimize stand-alone, for dist m/r(fan-out/in) arch to schedule those instances.

0. gpu device RR balance by user request
1. concurrency(cpu thread pool) + parallel(cpu openMP + gpu warp threads): cpu(baseline) -> cpu thread concurrency -> cpu + gpu -> cpu thread concurrency/parallel + gpu stream concurrency/warp thread parallel => dist
2. find or filter: use hashmap/bitmap(bloom) on cpu/gpu global memory or gpu shared memory
3. topk sort: heap sort (partial_sort) on cpu -> bitonic/radix sort on gpu parallel topk,then reduce topk to cpu
4. search: need build index (list(IVF,skip),tree, graph), orderly struct/model
5. SIMD: for cpu arch instruction set (intel cpu sse,avx2,avx512 etc..)
6. sequential IO stream pipeline: for r query/docs file, (batch per thread, multibyte_split parallel Accelerators) , w res file
7. resources pool

# result
add read file chunk topk on gpu, run on google colab A100

## cpu_readfile -> vec docs -> cpu_topk (cpu_baseline)

1. read file cost from 33054 ms(line/per) 
2. topk cost 87230 ms 
3. all cost 120284 ms 

---

## cpu_readfile -> vec docs split -> cpu_concurrency_topk 
use thread_pool thread num: cpu core num a100 (12 cores)

1. read file cost from 33054 ms(line/per) 
2. topk cost 14206 ms, reduce: (87230-14206)/87230=**83.71%** compare with `cpu_baseline`  
3. all cost 47654 ms, reduce: (120284-47654)/120284=**60.38%** compare with `cpu_baseline`  

---

## cpu_readfile -> vec docs -> gpu_cpu_topk (gpu_baseline)

1. read file cost from 33054 ms(line/per) 
2. topk cost 2504 ms, reduce: (87230-2504)/87230=**97.13%** compare with `cpu_baseline`  ;  (14206-2504)/14206=**97.13%** compare with `cpu_concurrency`  
3. all cost 36026 ms, reduce: (120284-36026)/120284=**70.05%** compare with `cpu_baseline`  ; (47654-36026)/47654=**24.40%** compare with `cpu_concurrency`  

---

## cpu_readfile -> vec docs split -> cpu_concurency_gpu_topk  : (

1. read file cost from 33054 ms(line/per) 
2. topk cost 2915 ms, increase: (2915-2504)/2915=**14.10%** compare with `gpu_baseline` ;
3. all cost 36230 ms, increase: (36230-36026)/36230=**00.56%** compare with `gpu_baseline` ; 

increase **cpu context switch cost** 

---


## gpu_readfile -> vec docs -> gpu_cpu_topk

1. read file cost from 34274 ms(line/per) to 9196 ms(gpu chunk multi_split), cost reduce (34274-9196)/34274 = **73.17%**
2. total cost reduce (35551 - 11589)/35551 = **67.40%**

---

## gpu_readfile -> gpu_chunk_topk -> gpu_cpu_topk

1. read file chunk pipeline to rank topk on gpu
2. total cost reduce (35551 - 7021)/35551 = **80.25%** compare with `gpu baseline`
3. total cost reduce (11589 - 7021)/11589 = **39.42%** compare with `gpu read file chunk to cpu vec docs then load to gpu rank topk`

---

##  query hashtable + doc hashtable score
(todo)

---

## (gpu_readfile -> gpu_chunk_topk -> gpu_cpu_topk) + stream pool + rmm 
(todo)

---

## use auto adpter select k -> sort -> top k. gpu accelerate
 (todo)

---

detail see: [my_colab_gpu_topk.ipynb](https://github.com/weedge/doraemon-nb/blob/main/my_colab_gpu_topk.ipynb)


# [reference](./docs/reference.md)
## view paper
1. [Fast Segmented Sort on GPUs.](https://raw.github.com/weedge/learn/main/gpu/Fast%20Segmented%20Sort%20on%20GPUs.pdf)
2. [Efficient Top-K query processing on massively parallel hardware](https://raw.githubusercontent.com/weedge/learn/main/gpu/Efficient%20Top-K%20Query%20Processing%20on%20Massively%20Parallel%20Hardware.pdf)
3. [stdgpu: Efficient STL-like Data Structures on the GPU](https://www.researchgate.net/publication/335233070_stdgpu_Efficient_STL-like_Data_Structures_on_the_GPU)
4. [Parallel Top-K Algorithms on GPU: A Comprehensive Study and New Methods](https://sc23.supercomputing.org/presentation/?id=pap294&sess=sess156)

## view code
1. https://github.com/rapidsai/cudf/pull/8702 , https://github.com/rapidsai/cudf/blob/branch-23.12/cpp/tests/io/text/multibyte_split_test.cpp
2. https://github.com/vtsynergy/bb_segsort (k/v), https://github.com/Funatiq/bb_segsort (k,k/v)
3. https://github.com/anilshanbhag/gpu-topk
4. https://github.com/heavyai/heavydb/blob/master/QueryEngine/TopKSort.cu
5. https://github.com/rapidsai/raft/blob/branch-23.12/cpp/include/raft/neighbors/detail/cagra/topk_for_cagra/topk_core.cuh
6. https://github.com/rapidsai/raft/blob/branch-23.12/cpp/include/raft/matrix/select_k.cuh , https://github.com/rapidsai/raft/blob/branch-23.12/cpp/test/matrix/select_k.cuh
