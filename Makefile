CXX ?= g++
CXXSTD ?= c++11
CXXFLAGS ?= -std=$(CXXSTD) -march=native
#CXXFLAGS ?= -std=$(CXXSTD) -Wall -march=native -pthread -fopenmp
#CXXFLAGS ?= -std=$(CXXSTD) -march=native -pthread -fopenmp

ARCH ?= 70
ARCH_CODE ?= -arch=sm_${ARCH} -gencode=arch=compute_${ARCH},code=sm_${ARCH} 
CUDA_PATH ?= /usr/local/cuda
NVCC ?=$(CUDA_PATH)/bin/nvcc
NVCCSTD ?= c++11
#NVCCFLAGS ?= -std=$(NVCCSTD) --expt-relaxed-constexpr --extended-lambda $(ARCH_CODE) 
NVCCFLAGS ?= -std=$(NVCCSTD) -Xcompiler="-fopenmp" --expt-relaxed-constexpr --extended-lambda $(ARCH_CODE)
#NVCCFLAGS ?= -std=$(NVCCSTD) -Xcompiler="-Wall -Wextra" --expt-relaxed-constexpr $(ARCH_CODE)
RAPIDSAI_DIR ?= 
NVCCLIB_CUDA ?= -L$(CUDA_PATH)/lib64 -lcudart -lcuda
NVCCLIB_CUDF ?= -L$(RAPIDSAI_DIR)/lib -lcudf -I$(RAPIDSAI_DIR)/include 
NVCCLIB_RAFT ?= -L$(RAPIDSAI_DIR)/lib -lraft -I$(RAPIDSAI_DIR)/include
#NVCCLIB_CCCL ?= -I$(RAPIDSAI_DIR)/cccl/thrust -I$(RAPIDSAI_DIR)/cccl/libcudacxx/include -I$(RAPIDSAI_DIR)/cccl/cub
NVCCLIB_LINKER ?=
#NVCCLIB_LINKER ?= -Xlinker="-rpath,$(RAPIDSAI_DIR)/lib"
NVCC_STREAM_FLAGS ?= --default-stream per-thread

BUILD_TYPE ?= Debug
OPTIMIZE_CFLAGS?=-O2
ifeq ($(BUILD_TYPE),Debug)
    OPTIMIZE_CFLAGS=-O0
endif

DOC_CN?=10
QUERY_CN?=10
TARGETS?=build_cpu build_cpu_concurrency 

all: $(TARGETS)

init:
	mkdir -p bin

# https://www.scivision.dev/install-nvidia-hpc-free-compiler/
# https://www.server-world.info/en/note?os=Ubuntu_22.04&p=nvidia&f=3
install_ubuntu_hpc_compiler:
	@curl https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg
	@echo 'deb [signed-by=/usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /' | sudo tee /etc/apt/sources.list.d/nvhpc.list
	@sudo apt-get update -y
	@sudo apt-get install -y nvhpc-22-11
	@source ./nvidia_hpc_compiler.sh

gen:
	@bash gen.sh $(DOC_CN)
	@bash gen_querys.sh $(QUERY_CN)

build_cpu: init
	$(CXX) ./main.cpp -o ./bin/query_doc_scoring_cpu  \
		-I./ \
		$(CXXFLAGS) \
		$(OPTIMIZE_CFLAGS) \
		-g 

build_cpu_concurrency: init
	$(CXX) ./main.cpp -o ./bin/query_doc_scoring_cpu_concurrency  \
		-I./ \
		$(CXXFLAGS) \
		$(OPTIMIZE_CFLAGS) \
		-DCPU_CONCURRENCY \
		-g 

build_cpu_gpu: init
	$(NVCC) ./main.cpp ./topk.cu -o ./bin/query_doc_scoring_cpu_gpu  \
		-I./ \
		$(NVCCLIB_CUDA) \
		$(NVCCFLAGS) \
		$(OPTIMIZE_CFLAGS) \
		-DGPU \
		-g

build_cpu_concurrency_gpu: init
	$(NVCC) ./main.cpp ./topk.cu -o ./bin/query_doc_scoring_cpu_concurrency_gpu  \
		-I./ \
		$(NVCCLIB_CUDA) \
		$(NVCCFLAGS) \
		$(OPTIMIZE_CFLAGS) \
		-DCPU_CONCURRENCY \
		-DGPU \
		-g

build_cpu_gpu_query_stream: init
	$(NVCC) ./main.cpp ./topk_query_stream.cu -o ./bin/query_doc_scoring_cpu_gpu_query_stream  \
		-I./ \
		$(NVCCLIB_CUDA) \
		$(NVCCFLAGS) \
		$(OPTIMIZE_CFLAGS) \
		$(NVCC_STREAM_FLAGS) \
		-DGPU \
		-g

build_cpu_gpu_doc_stream: init
	$(NVCC) ./main.cpp ./topk_doc_stream.cu -o ./bin/query_doc_scoring_cpu_gpu_doc_stream  \
		-I./ \
		$(NVCCLIB_CUDA) \
		$(NVCCFLAGS) \
		$(OPTIMIZE_CFLAGS) \
		$(NVCC_STREAM_FLAGS) \
		-DGPU \
		-g
build_cpu_gpu_pinned_doc_stream: init
	$(NVCC) ./main.cpp ./topk_doc_stream.cu -o ./bin/query_doc_scoring_cpu_gpu_pinned_doc_stream  \
		-I./ \
		$(NVCCLIB_CUDA) \
		$(NVCCFLAGS) \
		$(OPTIMIZE_CFLAGS) \
		$(NVCC_STREAM_FLAGS) \
		-DGPU -DPINNED_MEMORY \
		-g

build_cpu_gpu_hashtable: init
	$(NVCC) ./main.cpp ./topk_hashtable.cu -o ./bin/query_doc_scoring_cpu_gpu_hashtable \
		-I./ \
		$(NVCCLIB_CUDA) \
		$(NVCCFLAGS) \
		$(OPTIMIZE_CFLAGS) \
		-DGPU \
		-g

build_cpu_gpu_align_locality: init
	$(NVCC) ./main.cpp ./topk_doc_align_locality.cu -o ./bin/query_doc_scoring_align_locality \
		-I./ \
		$(NVCCLIB_CUDA) \
		$(NVCCFLAGS) \
		$(OPTIMIZE_CFLAGS) \
		$(NVCC_STREAM_FLAGS) \
		-DGPU \
		-g
build_cpu_gpu_pinned_align_locality: init
	$(NVCC) ./main.cpp ./topk_doc_align_locality.cu -o ./bin/query_doc_scoring_pinned_align_locality \
		-I./ \
		$(NVCCLIB_CUDA) \
		$(NVCCFLAGS) \
		$(OPTIMIZE_CFLAGS) \
		$(NVCC_STREAM_FLAGS) \
		-DGPU -DPINNED_MEMORY \
		-g

build_cpu_gpu_pinned_memory: init
	$(NVCC) ./main.cpp ./topk_pinned_memory.cu -o ./bin/query_doc_scoring_pinned_memory \
		-I./ \
		$(NVCCLIB_CUDA) \
		$(NVCCFLAGS) \
		$(OPTIMIZE_CFLAGS) \
		-DGPU \
		-g
build_cpu_gpu_pinned_map_memory: init
	$(NVCC) ./main.cpp ./topk_pinned_memory.cu -o ./bin/query_doc_scoring_pinned_map_memory \
		-I./ \
		$(NVCCLIB_CUDA) \
		$(NVCCFLAGS) \
		$(OPTIMIZE_CFLAGS) \
		-DGPU -DMAP_HOST_MEMORY \
		-g

build_cpu_gpu_readfile: init
	$(NVCC) ./main.cpp ./readfile.cu ./topk.cu -o ./bin/query_doc_scoring_cpu_gpu_readfile \
		-I./ \
		$(NVCCFLAGS) \
		$(NVCCLIB_CUDA) $(NVCCLIB_LINKER) \
		$(NVCCLIB_CUDF) \
		$(OPTIMIZE_CFLAGS) \
		-DGPU -DFMT_HEADER_ONLY -DPIO \
		-g

# make -C topk build_gpu_cudf_strings BUILD_TYPE=Release  RAPIDSAI_DIR=$HOME/rapidsai NVCCSTD=c++17
build_gpu_cudf_strings: init
	$(NVCC) ./main.cpp ./readfile.cu ./topk_doc_cudf_strings.cu -o ./bin/query_doc_scoring_gpu_cudf_strings \
		-I./ \
		$(NVCCFLAGS) \
		$(NVCCLIB_CUDA) $(NVCCLIB_LINKER) \
		$(NVCCLIB_CUDF) \
		$(OPTIMIZE_CFLAGS) \
		$(NVCC_STREAM_FLAGS) \
		-DFMT_HEADER_ONLY -DGPU -DPIO_TOPK \
		-g

build_gpu_raft_selectk: init
	$(NVCC) ./main.cpp ./topk_raft_selectk.cu -o ./bin/query_doc_scoring_gpu_raft_selectk \
		-I./ \
		$(NVCCFLAGS) \
		$(NVCCLIB_CUDA) $(NVCCLIB_LINKER) \
		$(NVCCLIB_RAFT) \
		$(OPTIMIZE_CFLAGS) \
		$(NVCC_STREAM_FLAGS) \
		-DGPU -DFMT_HEADER_ONLY \
		-g

build_gpu_doc_align_locality_query_stream_raft_selectk: init
	$(NVCC) ./main.cpp ./topk_doc_align_locality_query_stream_raft_selectk.cu -o ./bin/query_doc_scoring_gpu_doc_align_locality_query_stream_raft_selectk \
		-I./ \
		$(NVCCFLAGS) \
		$(NVCCLIB_CUDA) $(NVCCLIB_LINKER) \
		$(NVCCLIB_RAFT) \
		$(OPTIMIZE_CFLAGS) \
		$(NVCC_STREAM_FLAGS) \
		-DGPU -DFMT_HEADER_ONLY \
		-g
build_gpu_pinned_doc_align_locality_query_stream_raft_selectk: init
	$(NVCC) ./main.cpp ./topk_doc_align_locality_query_stream_raft_selectk.cu -o ./bin/query_doc_scoring_gpu_pinned_doc_align_locality_query_stream_raft_selectk \
		-I./ \
		$(NVCCFLAGS) \
		$(NVCCLIB_CUDA) $(NVCCLIB_LINKER) \
		$(NVCCLIB_RAFT) \
		$(OPTIMIZE_CFLAGS) \
		$(NVCC_STREAM_FLAGS) \
		-DGPU -DFMT_HEADER_ONLY -DPINNED_MEMORY \
		-g
build_gpu_readfile_doc_align_locality_query_stream_raft_selectk: init
	$(NVCC) ./main.cpp ./readfile.cu ./topk_doc_align_locality_query_stream_raft_selectk.cu \
		-o ./bin/query_doc_scoring_gpu_readfile_doc_align_locality_query_stream_raft_selectk \
		-I./ \
		$(NVCCFLAGS) \
		$(NVCCLIB_CUDA) $(NVCCLIB_LINKER) \
		$(NVCCLIB_CUDF) \
		$(NVCCLIB_RAFT) \
		$(OPTIMIZE_CFLAGS) \
		$(NVCC_STREAM_FLAGS) \
		-DGPU -DFMT_HEADER_ONLY -DPIO \
		-g

build_gpu_cudf_strings_raft_selectk: init
	$(NVCC) ./main.cpp ./readfile.cu ./topk_doc_cudf_strings_raft_selectk.cu -o ./bin/query_doc_scoring_gpu_cudf_strings_raft_selectk \
		-I./ \
		$(NVCCFLAGS) \
		$(NVCCLIB_CUDA) $(NVCCLIB_LINKER) \
		$(NVCCLIB_CUDF) \
		$(NVCCLIB_RAFT) \
		$(OPTIMIZE_CFLAGS) \
		$(NVCC_STREAM_FLAGS) \
		-DFMT_HEADER_ONLY -DGPU -DPIO_TOPK \
		-g

build_examples: init build_cpu_examples build_cpu_gpu_examples
build_cpu_examples: init build_example_threadpool build_example_readfile_cpu
build_cpu_gpu_examples: init build_example_readfile_gpu build_example_raft_selectk build_example_raft_selectk_null_option

build_example_threadpool: init
	$(CXX) -o bin/example_threadpool example_threadpool.cpp \
		-I./ \
		$(CXXFLAGS) \
		$(OPTIMIZE_CFLAGS) \
		-g

build_example_readfile_cpu: init
	$(NVCC) -o bin/example_readfile_cpu example_readfile.cpp -DFMT_HEADER_ONLY \
		-I./ \
		$(CXXFLAGS) \
		$(OPTIMIZE_CFLAGS) \
		-g

build_example_readfile_gpu: init
	$(NVCC) -o bin/example_readfile_gpu example_readfile.cpp readfile.cu -DGPU -DFMT_HEADER_ONLY \
		-I./ \
		$(NVCCFLAGS) \
		$(NVCCLIB_CUDA) $(NVCCLIB_LINKER) \
		$(NVCCLIB_CUDF) \
		$(OPTIMIZE_CFLAGS) \
		-g

build_example_raft_selectk: init
	$(NVCC) -o bin/example_raft_selectk example_raft_selectk.cu -DFMT_HEADER_ONLY \
		-I./ \
		$(NVCCFLAGS) \
		$(NVCCLIB_CUDA) $(NVCCLIB_LINKER) \
		$(NVCCLIB_RAFT) \
		$(OPTIMIZE_CFLAGS) \
		-g

build_example_raft_selectk_null_option: init
	$(NVCC) -o bin/example_raft_selectk_null_option example_raft_selectk.cu -DNULL_OPTIONAL -DFMT_HEADER_ONLY \
		-I./ \
		$(NVCCFLAGS) \
		$(NVCCLIB_CUDA) $(NVCCLIB_LINKER) \
		$(NVCCLIB_RAFT) \
		$(OPTIMIZE_CFLAGS) \
		-g

run:
	bin/query_doc_scoring_cpu testdata/docs.txt testdata/query testdata/res_cpu.txt
	bin/query_doc_scoring_cpu_concurrency testdata/docs.txt testdata/query testdata/res_cpu_concurrency.txt

diff:
	diff testdata/res_cpu.txt  testdata/res_cpu_concurrency.txt

archive_cpu_gpu:
	rm -rf archive/cpu_gpu
	mkdir -p archive/cpu_gpu/{src,bin}
	cp build_cpu_gpu.sh archive/cpu_gpu/build.sh
	cp run.sh archive/cpu_gpu/
	cp helper.h main.cpp topk.h topk.cu archive/cpu_gpu/src
	rm -f archive/cpu_gpu_topk.zip
	cd archive/cpu_gpu/ && zip -v -r cpu_gpu_topk.zip \
		build.sh run.sh src \
		&& zip -sf cpu_gpu_topk.zip

archive_gpu_cudf_strings:
	rm -rf archive/gpu_cudf_strings
	mkdir -p archive/gpu_cudf_strings/{src,bin}
	cp build_gpu_cudf_strings.sh archive/gpu_cudf_strings/build.sh
	cp run.sh archive/gpu_cudf_strings/
	cp helper.h main.cpp readfile.cu readfile.h topk.h topk_doc_cudf_strings.cu archive/gpu_cudf_strings/src
	rm -f archive/gpu_cudf_strings_topk.zip
	cd archive/gpu_cudf_strings/ && zip -v -r gpu_cudf_strings_topk.zip \
		build.sh run.sh src \
		&& zip -sf gpu_cudf_strings_topk.zip

archive_gpu_raft_selectk:
	rm -rf archive/gpu_raft_selectk
	mkdir -p archive/gpu_raft_selectk/{src,bin}
	cp build_gpu_raft_selectk.sh archive/gpu_raft_selectk/build.sh
	cp run.sh archive/gpu_raft_selectk/
	cp helper.h main.cpp topk.h topk_raft_selectk.cu archive/gpu_raft_selectk/src
	rm -f archive/gpu_raft_selectk_topk.zip
	cd archive/gpu_raft_selectk/ && zip -v -r gpu_raft_selectk_topk.zip \
		build.sh run.sh src \
		&& zip -sf gpu_raft_selectk_topk.zip

# ping bj.bcebos.com
get_gpu_baseline:
	@wget "https://bj.bcebos.com/v1/ai-studio-online/9805dd2d2e8e472693efac637628e16b9f9c5be0fe30438bb4a80de3b386781a?responseContentDisposition=attachment%3B%20filename%3DSTI2_1017.zip&authorization=bce-auth-v1%2F5cfe9a5e1454405eb2a975c43eace6ec%2F2023-10-18T12%3A42%3A27Z%2F-1%2F%2F6b5388dcd9013bc9b340bb1806476afa938ce0c65f2f595e1a75f529e90e4187" -O STI2_1017.zip
	@rm -rf STI2 && unzip STI2_1017.zip && mv STI2\ 2 STI2

install_ubuntu_profiler:
	@wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nsight-systems-2023.2.3_2023.2.3.1001-1_amd64.deb
	@sudo apt update
	@sudo apt install ./nsight-systems-2023.2.3_2023.2.3.1001-1_amd64.deb
	@sudo apt --fix-broken install

profile_cpu_gpu:
#nvprof --print-gpu-trace bin/query_doc_scoring_cpu_gpu STI2/translate/docs.txt STI2/translate/querys ./cpu_gpu_res.txt
	@nsys profile --force-overwrite true -o report_cpu_gpu.nsys-rep \
		bin/query_doc_scoring_cpu_gpu STI2/translate/docs.txt STI2/translate/querys ./cpu_gpu_res.txt
	@ncu --set full --call-stack --nvtx -o report_cpu_gpu \
		bin/query_doc_scoring_cpu_gpu STI2/translate/docs.txt STI2/translate/querys ./cpu_gpu_res.txt

profile_cpu_concurency_gpu:
#nvprof --print-gpu-trace bin/query_doc_scoring_cpu_concurency_gpu STI2/translate/docs.txt STI2/translate/querys ./cpu_concurency_gpu_res.txt
	@nsys profile --force-overwrite true -o report_cpu_concurrency_gpu.nsys-rep \
		bin/query_doc_scoring_cpu_concurrency_gpu STI2/translate/docs.txt STI2/translate/querys ./cpu_concurency_gpu_res.txt
	@ncu --set full --call-stack --nvtx -o report_cpu_concurrency_gpu \
		bin/query_doc_scoring_cpu_concurrency_gpu STI2/translate/docs.txt STI2/translate/querys ./cpu_concurency_gpu_res.txt

#nvprof --profile-from-start off --profile-child-processes --csv bin/query_doc_scoring_cpu_gpu testdata/docs.txt testdata/query testdata/res_gpu.txt
#nvprof --profile-from-start off --profile-child-processes --csv bin/query_doc_scoring_cpu_gpu_doc_stream testdata/docs.txt testdata/query test

.PHONY: clean
clean:
	rm -rf bin/*

clean_testdata:
	rm -rf testdata/*


build_3d_gpu_selection:
	@cd third_party/gpu_selection && cmake -DTARGET_ARCH=$(ARCH) -B build -S . && make -C build && cd -
	@mv third_party/gpu_selection/build/lib/libgpu_selection.so lib/libgpu_selection.so

clean_3d_gpu_selection:
	@rm -f lib/libgpu_selection.so
	@rm -rf third_party/gpu_selection/build


# make build_3d_faiss NVCCSTD=c++14 CXXFLAGS="-std=c++14 -fPIC" BUILD_TYPE=Release TARGET=build_3d_faiss
ifeq ($(TARGET),build_3d_faiss)
  CXXFLAGS += -fPIC 
  NVCCFLAGS += -Xcompiler "-fPIC"
  CPPFLAGS = $(OPTIMIZE_CFLAGS) -g -I./third_party/ -I$(CUDA_PATH)/include
endif
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -c $< -o $@

faiss_gpu_objs := third_party/faiss/gpu/GpuResources.o \
third_party/faiss/gpu/utils/DeviceUtils.o \
third_party/faiss/gpu/utils/BlockSelectFloat.o \
third_party/faiss/gpu/utils/WarpSelectFloat.o \
third_party/faiss/gpu/utils/blockselect/BlockSelectFloat128.o \
third_party/faiss/gpu/utils/blockselect/BlockSelectFloat1.o \
third_party/faiss/gpu/utils/blockselect/BlockSelectFloat256.o \
third_party/faiss/gpu/utils/blockselect/BlockSelectFloat32.o \
third_party/faiss/gpu/utils/blockselect/BlockSelectFloat64.o \
third_party/faiss/gpu/utils/blockselect/BlockSelectFloatF1024.o \
third_party/faiss/gpu/utils/blockselect/BlockSelectFloatF2048.o \
third_party/faiss/gpu/utils/blockselect/BlockSelectFloatF512.o \
third_party/faiss/gpu/utils/blockselect/BlockSelectFloatT1024.o \
third_party/faiss/gpu/utils/blockselect/BlockSelectFloatT2048.o \
third_party/faiss/gpu/utils/blockselect/BlockSelectFloatT512.o \
third_party/faiss/gpu/utils/warpselect/WarpSelectFloat128.o \
third_party/faiss/gpu/utils/warpselect/WarpSelectFloat1.o \
third_party/faiss/gpu/utils/warpselect/WarpSelectFloat256.o \
third_party/faiss/gpu/utils/warpselect/WarpSelectFloat32.o \
third_party/faiss/gpu/utils/warpselect/WarpSelectFloat64.o \
third_party/faiss/gpu/utils/warpselect/WarpSelectFloatF1024.o \
third_party/faiss/gpu/utils/warpselect/WarpSelectFloatF2048.o \
third_party/faiss/gpu/utils/warpselect/WarpSelectFloatF512.o \
third_party/faiss/gpu/utils/warpselect/WarpSelectFloatT1024.o \
third_party/faiss/gpu/utils/warpselect/WarpSelectFloatT2048.o \
third_party/faiss/gpu/utils/warpselect/WarpSelectFloatT512.o \

libfaiss.so: $(faiss_gpu_objs)
	$(NVCC) -o $@ $(NVCCFLAGS) $(OPTIMIZE_CFLAGS) -g \
    -Xcompiler "-fPIC" -Xcompiler "-shared" $^ -lcublas

build_3d_faiss: libfaiss.so 
	@mv ./libfaiss.so ./lib/

clean_3d_faiss:
	@rm -f $(faiss_gpu_objs) lib/libfaiss.so
