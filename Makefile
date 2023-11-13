CXX ?= g++
CXXSTD ?= c++11
CXXFLAGS ?= -std=$(CXXSTD) -Wall -march=native -pthread

ARCH ?= 70
ARCH_CODE ?= -arch=sm_${ARCH}
#ARCH_CODE ?= -gencode arch=compute_${ARCH},code=sm_${ARCH} 
NVCC ?= nvcc
NVCCSTD ?= c++11
NVCCFLAGS ?= -std=$(NVCCSTD) --expt-relaxed-constexpr --extended-lambda $(ARCH_CODE)
#NVCCFLAGS ?= -std=$(NVCCSTD) -Xcompiler="-Wall -Wextra" --expt-relaxed-constexpr $(ARCH_CODE)
NVCCLIB_CUDA ?= -L/usr/local/cuda/lib64 -lcudart -lcuda
NVCCLIB_CUDF ?= -L/lib -lcudf -I/include 
NVCCLIB_RAFT ?= -L/lib -lraft -I/include 

BUILD_TYPE ?= Debug
OPTIMIZE_CFLAGS?=-O3
ifeq ($(BUILD_TYPE),Debug)
    OPTIMIZE_CFLAGS=-O0
endif

DOC_CN?=10
QUERY_CN?=10
TARGETS?=build_cpu build_cpu_concurrency 

all: $(TARGETS)

init:
	mkdir -p bin

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
		-DGPU \
		-g

build_cpu_gpu_doc_stream: init
	$(NVCC) ./main.cpp ./topk_doc_stream.cu -o ./bin/query_doc_scoring_cpu_gpu_doc_stream  \
		-I./ \
		$(NVCCLIB_CUDA) \
		$(NVCCFLAGS) \
		$(OPTIMIZE_CFLAGS) \
		-DGPU \
		-g

build_cpu_gpu_hashtable: init
	$(NVCC) ./main.cpp ./topk_hashtable.cu -o ./bin/query_doc_scoring_cpu_gpu_hashtable \
		-I./ \
		$(NVCCLIB_CUDA) \
		$(NVCCFLAGS) \
		$(OPTIMIZE_CFLAGS) \
		-DGPU \
		-g

build_cpu_gpu_sort: init
	$(NVCC) ./main.cpp ./topk_sort.cu -o ./bin/query_doc_scoring_cpu_gpu_sort \
		-I./ \
		$(NVCCFLAGS) \
		$(NVCCLIB_CUDA) \
		$(NVCCLIB_RAFT) \
		$(OPTIMIZE_CFLAGS) \
		-DGPU -DFMT_HEADER_ONLY \
		-g

build_cpu_gpu_readfile: init
	$(NVCC) ./main.cpp ./readfile.cu ./topk.cu -o ./bin/query_doc_scoring_cpu_gpu_readfile \
		-I./ \
		$(NVCCFLAGS) \
		$(NVCCLIB_CUDA) \
		$(NVCCLIB_CUDF) \
		$(OPTIMIZE_CFLAGS) \
		-DGPU -DFMT_HEADER_ONLY -DPIO \
		-g

build_gpu_cudf_strings: init
	$(NVCC) ./main.cpp ./readfile.cu ./topk_doc_cudf_strings.cu -o ./bin/query_doc_scoring_gpu_cudf_strings \
		-I./ \
		$(NVCCFLAGS) \
		$(NVCCLIB_CUDA) \
		$(NVCCLIB_CUDF) \
		$(OPTIMIZE_CFLAGS) \
		-DFMT_HEADER_ONLY -DGPU -DPIO_TOPK \
		-g

build_examples: init build_example_threadpool build_example_readfile_cpu build_example_readfile_gpu

build_example_threadpool:
	$(CXX) -o bin/example_threadpool example_threadpool.cpp \
		-I./ \
		$(CXXFLAGS) \
		$(OPTIMIZE_CFLAGS) \
		-g

build_example_readfile_cpu:
	$(NVCC) -o bin/example_readfile_cpu example_readfile.cpp -DFMT_HEADER_ONLY \
		-I./ \
		$(CXXFLAGS) \
		$(OPTIMIZE_CFLAGS) \
		-g

build_example_readfile_gpu:
	$(NVCC) -o bin/example_readfile_gpu example_readfile.cpp readfile.cu -DGPU -DFMT_HEADER_ONLY \
		-I./ \
		$(NVCCFLAGS) \
		$(NVCCLIB_CUDA) \
		$(NVCCLIB_CUDF) \
		$(OPTIMIZE_CFLAGS) \
		-g

run:
	bin/query_doc_scoring_cpu testdata/docs.txt testdata/query testdata/res_cpu.txt
	bin/query_doc_scoring_cpu_concurrency testdata/docs.txt testdata/query testdata/res_cpu_concurrency.txt

diff:
	diff testdata/res_cpu.txt  testdata/res_cpu_concurrency.txt

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


get_gpu_baseline:
	wget "https://bj.bcebos.com/v1/ai-studio-online/9805dd2d2e8e472693efac637628e16b9f9c5be0fe30438bb4a80de3b386781a?responseContentDisposition=attachment%3B%20filename%3DSTI2_1017.zip&authorization=bce-auth-v1%2F5cfe9a5e1454405eb2a975c43eace6ec%2F2023-10-18T12%3A42%3A27Z%2F-1%2F%2F6b5388dcd9013bc9b340bb1806476afa938ce0c65f2f595e1a75f529e90e4187" -O STI2_1017.zip
	rm -rf STI2 && unzip STI2_1017.zip && mv STI2\ 2 STI2


profile_cpu_gpu:
#nvprof --print-gpu-trace bin/query_doc_scoring_cpu_gpu STI2/translate/docs.txt STI2/translate/querys ./cpu_gpu_res.txt
	nsys profile  -o report_cpu_gpu.nsys-rep bin/query_doc_scoring_cpu_gpu STI2/translate/docs.txt STI2/translate/querys ./cpu_gpu_res.txt
	ncu --set full --call-stack --nvtx -o report_cpu_gpu bin/query_doc_scoring_cpu_gpu STI2/translate/docs.txt STI2/translate/querys ./cpu_gpu_res.txt

profile_cpu_concurency_gpu:
#nvprof --print-gpu-trace bin/query_doc_scoring_cpu_concurency_gpu STI2/translate/docs.txt STI2/translate/querys ./cpu_concurency_gpu_res.txt
	nsys profile  -o report_cpu_concurrency_gpu.nsys-rep bin/query_doc_scoring_cpu_concurrency_gpu STI2/translate/docs.txt STI2/translate/querys ./cpu_concurency_gpu_res.txt
	ncu --set full --call-stack --nvtx -o report_cpu_concurrency_gpu bin/query_doc_scoring_cpu_concurrency_gpu STI2/translate/docs.txt STI2/translate/querys ./cpu_concurency_gpu_res.txt

#nvprof --profile-from-start off --profile-child-processes --csv bin/query_doc_scoring_cpu_gpu testdata/docs.txt testdata/query testdata/res_gpu.txt
#nvprof --profile-from-start off --profile-child-processes --csv bin/query_doc_scoring_cpu_gpu_doc_stream testdata/docs.txt testdata/query test

clean:
	rm -rf bin/*

clean_testdata:
	rm -rf testdata/*
