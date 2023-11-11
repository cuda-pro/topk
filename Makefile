CXX ?= g++
CXXSTD ?= c++11
CXXFLAGS ?= -std=$(CXXSTD) -Wall -march=native -pthread

ARCH ?= 70
ARCH_CODE ?= -gencode arch=compute_${ARCH},code=sm_${ARCH} 
NVCC ?= nvcc
NVCCSTD ?= c++11
NVCCFLAGS ?= -std=$(NVCCSTD) -Xcompiler="-Wall -Wextra" --expt-relaxed-constexpr $(ARCH_CODE)
NVCCLIB_CUDA ?= -L/usr/local/cuda/lib64 -lcudart -lcuda
NVCCLIB_CUDF ?= -L/lib -lcudf -I/include 

BUILD_TYPE ?= Debug
OPTIMIZE_CFLAGS?=-O3
ifeq ($(BUILD_TYPE),Debug)
    OPTIMIZE_CFLAGS=-O0
endif

DOC_CN?=10
QUERY_CN?=10
TARGETS = build_cpu build_cpu_concurrency 

all: $(TARGETS)

init:
	mkdir -p bin
	mkdir -p achive

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

build_cpu_gpu_doc_stream: init
	$(NVCC) ./main.cpp ./topk_doc_stream.cu -o ./bin/query_doc_scoring_cpu_gpu_doc_stream  \
		-I./ \
		$(NVCCLIB_CUDA) \
		$(NVCCFLAGS) \
		$(OPTIMIZE_CFLAGS) \
		-DGPU \
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

achive_gpu_cudf_strings: init
	rm -f achive/gpu_cudf_strings_topk.zip
	zip -v achive/gpu_cudf_strings_topk.zip \
		build.sh run.sh \
		helper.h main.cpp readfile.cu readfile.h topk.h topk_doc_cudf_strings.cu \
		&& zip -sf achive/gpu_cudf_strings_topk.zip

clean:
	rm -rf bin/*

clean_testdata:
	rm -rf testdata/*