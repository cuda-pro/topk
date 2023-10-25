# task
【任务描述】
给定850万条规模的数据文件，每条数据是最大128维度的整型id向量 （称为doc），id取值范围是0-50000，给定一个最大128维的整型id向量（称为query），求query与doc全集内各数据的交集个数topk（k=100）

【评分标准（初赛）】机测
功能正确性（测试集准确率100%）为前置条件；
显存占用分在8GB内为50分，8-16GB内为25分，高于16GB为0分；
时延计算从读数据文件开始，精确到ms，分数按照排名，从第一名50分开始依次递减3分（第一名为50分，第二名为47分，如有平分则共享排名分，后续分数按照位次递延）。

# reference
1. https://www.youtube.com/watch?v=cOBtkPsgkus
2. https://raw.githubusercontent.com/cteqeu/HAC/main/GPU/cuda_profile.cu
3. https://docs.nvidia.com/cuda/cuda-runtime-api/index.html
4. https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
5. https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html

