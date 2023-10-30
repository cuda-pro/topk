CXX ?= g++
CXXSTD ?= c++11
CXXFLAGS ?= -std=$(CXXSTD) -Wall -march=native -pthread

ARCH ?= 70
NVCC ?= nvcc
NVCCSTD ?= c++11
NVCCFLAGS ?= -std=$(NVCCSTD) -Xcompiler="-Wall -Wextra" -gencode arch=compute_${ARCH},code=sm_${ARCH} --expt-relaxed-constexpr
NVCCLIB_CUDA ?= -L/usr/local/cuda/lib64 -lcudart -lcuda

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

build_examples: init
	$(CXX) -o bin/example_threadpool example_threadpool.cpp \
		-I./ \
		$(CXXFLAGS) \
		$(OPTIMIZE_CFLAGS) \
		-g

run:
	bin/query_doc_scoring_cpu testdata/docs.txt testdata/query testdata/res_cpu.txt
	bin/query_doc_scoring_cpu_concurrency testdata/docs.txt testdata/query testdata/res_cpu_concurrency.txt

diff:
	diff testdata/res_cpu.txt  testdata/res_cpu_concurrency.txt


clean:
	rm -rf bin/*

clean_testdata:
	rm -rf testdata/*