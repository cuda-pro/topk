# task
给定850万条规模的数据文件，每条数据是最大128维度的整型id向量 （称为doc），id取值范围是0-50000，给定一个最大128维的整型id向量（称为query），数据集可以扩散进行优化
```shell
# 生成测试数据
# query/query*.txt
bash -x gen.sh 3 10 query/query0.txt
# docs.txt
bash -x gen.sh 10
```
求query与doc全集内各数据交集个数平均分 topk (k=100); 这里定义item交集分数为：
query[i] >= doc[j] (0<=i<query_size, 0<=j<doc_size) 算一个交集, 平均分为 交集数目/max(query_size,doc_size)
```
./bin/query_doc_scoring.bin <doc_file_name> <query_file_name> <output_filename>
```


# reference
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
- https://docs.nvidia.com/cuda/cuda-runtime-api/index.html
- https://www.youtube.com/watch?v=cOBtkPsgkus



