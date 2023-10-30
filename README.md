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
note: just optimize stand-alone, for dist m/r arch to schedule those instances
1. currency(cpu thread pool) + parallel(cpu openMP + gpu warp pool): cpu(baseline) -> cpu thread currency -> cpu + gpu -> cpu thread currency + gpu => dist
2. find or filter: use hash/bitmap(bloom)
3. topk sort: heap sort (partial_sort) -> bitonic sort (gpu parallel)
4. search: need build index (list(IVF,skip),tree or graph), orderly struct/model
5. SIMD: for cpu arch instruction set (intel cpu sse,avx2,avx512 etc..)
6. IO stream pipeline: for r query/docs file, (batch per thread, parallel Accelerators) , w res file

# reference
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
- https://docs.nvidia.com/cuda/cuda-runtime-api/index.html
- https://docs.nvidia.com/cuda/thrust/index.html
- https://nvlabs.github.io/cub/index.html
- https://stotko.github.io/stdgpu/api/memory.html
- https://www.youtube.com/watch?v=cOBtkPsgkus
- https://www.csd.uwo.ca/~mmorenom/HPC-Slides/Many_core_computing_with_CUDA.pdf
- [Exploring Performance Portability for Accelerators via High-level Parallel Patterns](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=4Ab_NBkAAAAJ&citation_for_view=4Ab_NBkAAAAJ:hqOjcs7Dif8C), [PPT](https://pdfs.semanticscholar.org/b34a/f7c4739d622379fa31a1e88155335061c1b1.pdf)
- https://zhuanlan.zhihu.com/p/52344300
- 
- https://passlab.github.io/OpenMPProgrammingBook/cover.html
  


## code
1. https://github.com/Funatiq/bb_segsort
2. https://github.com/anilshanbhag/gpu-topk
3. https://github.com/heavyai/heavydb/blob/master/QueryEngine/TopKSort.cu

## paper
1. [Fast Segmented Sort on GPUs.](https://raw.github.com/weedge/learn/main/gpu/Fast%20Segmented%20Sort%20on%20GPUs.pdf)
2. [Efficient Top-K query processing on massively parallel hardware](https://raw.githubusercontent.com/weedge/learn/main/gpu/Efficient%20Top-K%20Query%20Processing%20on%20Massively%20Parallel%20Hardware.pdf)