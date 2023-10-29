CXX ?= g++
CXXSTD ?= c++11
CXXFLAGS ?= -std=$(CXXSTD) -Wall -march=native -pthread

NVCC ?= nvcc
NVCCSTD ?= c++11
NVCCFLAGS ?= -std=$(NVCCSTD)
NVCCLIB_CUDA ?= -L/usr/local/cuda/lib64 -lcudart -lcuda

BUILD_TYPE ?= Debug
OPTIMIZE_CFLAGS?=-O3
ifeq ($(BUILD_TYPE),Debug)
    OPTIMIZE_CFLAGS=-O0
endif

DOC_CN?=10
QUERY_CN?=10
TARGETS = build_cpu build_cpu_concurency 

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

build_cpu_concurency: init
	$(CXX) ./main.cpp -o ./bin/query_doc_scoring_cpu_concurency  \
		-I./ \
		$(CXXFLAGS) \
		$(OPTIMIZE_CFLAGS) \
		-DCPU_CONCURENCY \
		-g 

build_cpu_gpu: init
	$(NVCC) ./main.cpp ./topk.cu -o ./bin/query_doc_scoring_cpu_gpu  \
		-I./ \
		$(NVCCLIB_CUDA) \
		$(NVCCFLAGS) \
		$(OPTIMIZE_CFLAGS) \
		-g

build_cpu_concurency_gpu: init
	$(NVCC) ./main.cpp ./topk.cu -o ./bin/query_doc_scoring_cpu_concurency_gpu  \
		-I./ \
		$(NVCCLIB_CUDA) \
		$(NVCCFLAGS) \
		$(OPTIMIZE_CFLAGS) \
		-DCPU_CONCURENCY \
		-g

build_examples: init
	$(CXX) -o bin/example_threadpool example_threadpool.cpp \
		-I./ \
		$(CXXFLAGS) \
		$(OPTIMIZE_CFLAGS) \
		-g

run:
	bin/query_doc_scoring_cpu testdata/docs.txt testdata/query testdata/res_cpu.txt
	bin/query_doc_scoring_cpu_concurency testdata/docs.txt testdata/query testdata/res_cpu_concurency.txt

diff:
	diff testdata/res_cpu.txt  testdata/res_cpu_concurency.txt


clean:
	rm -rf bin/*

clean_testdata:
	rm -rf testdata/*