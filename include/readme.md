from: https://github.com/ZhangJingrong/gpu_topK_benchmark

data_t: float, index_t: int
```c++
using topk_func_t = std::function<void(void* buf,
                        size_t& buf_size,
                        const T* in,
                        int batch_size,
                        idxT len,
                        idxT k,
                        T* out,
                        idxT* out_idx,
                        bool greater,
                        cudaStream_t stream)>;

std::string algo = "grid_select";
Factory<float, int> factory;
Factory<float, int>::topk_func_t topk_func = factory.create(algo);

// warnup
void* d_buf = nullptr;
size_t buf_size;
topk_func(nullptr,
          buf_size,
          nullptr,
          batch_size,
          len,
          k,
          nullptr,
          nullptr,
          greater,
          stream);
assert(buf_size);
TOPK_CUDA_CHECK(cudaMalloc((void**)&d_buf, buf_size));
```