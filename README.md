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
note: just optimize stand-alone, for dist m/r(fan-out/in) arch to schedule those instances
1. concurrency(cpu thread pool) + parallel(cpu openMP + gpu stream warp pool): cpu(baseline) -> cpu thread concurrency -> cpu + gpu -> cpu thread concurrency/parallel + gpu => dist
2. find or filter: use hashmap/bitmap(bloom) on cpu/gpu
3. topk sort: heap sort (partial_sort) on cpu -> bitonic sort on gpu parallel to select topk
4. search: need build index (list(IVF,skip),tree or graph), orderly struct/model
5. SIMD: for cpu arch instruction set (intel cpu sse,avx2,avx512 etc..)
6. IO stream pipeline: for r query/docs file, (batch per thread, parallel Accelerators) , w res file

# [reference](./docs/reference.md)
## paper
1. [Fast Segmented Sort on GPUs.](https://raw.github.com/weedge/learn/main/gpu/Fast%20Segmented%20Sort%20on%20GPUs.pdf)
2. [Efficient Top-K query processing on massively parallel hardware](https://raw.githubusercontent.com/weedge/learn/main/gpu/Efficient%20Top-K%20Query%20Processing%20on%20Massively%20Parallel%20Hardware.pdf)
3. [stdgpu: Efficient STL-like Data Structures on the GPU](https://www.researchgate.net/publication/335233070_stdgpu_Efficient_STL-like_Data_Structures_on_the_GPU)
  
## code
1. https://github.com/vtsynergy/bb_segsort (k/v), https://github.com/Funatiq/bb_segsort (k,k/v)
2. https://github.com/anilshanbhag/gpu-topk
3. https://github.com/heavyai/heavydb/blob/master/QueryEngine/TopKSort.cu
4. https://github.com/rapidsai/raft/blob/branch-23.12/cpp/include/raft/neighbors/detail/cagra/topk_for_cagra/topk_core.cuh
