# task
给定850万条规模的数据文件，每条数据是最大128维度的整型id向量 （称为doc），id取值范围是0-50000，给定一个最大128维的整型id向量（称为query），数据集可以扩散进行优化
```shell
# 生成测试数据,已升序排序,默认docs文件每行算一个文档,10个文档; 10个query文件
make gen
```
求query与doc全集内各数据交集个数平均分 topk (k=100); 这里定义item交集分数为：
query[i] == doc[j] (0<=i<query_size, 0<=j<doc_size) 算一个交集, 平均分为 query与doc交集数目/max(query_size,doc_size)
``` shell
./bin/query_doc_scoring <doc_file_name> <query_file_name> <output_filename>
```

# optimize
1. currency(cpu thread pool) + parallel(gpu warp pool): cpu -> cpu thread currency -> cpu + gpu -> cpu thread currency + gpu
2. find or filter: use hash/bitmap(bloom)
3. topk sort: heap sort (partial_sort) -> bitonic sort
4. search: need build index (list(ivf,skip),tree or graph), orderly struct

# reference
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
- https://docs.nvidia.com/cuda/cuda-runtime-api/index.html
- https://www.youtube.com/watch?v=cOBtkPsgkus



