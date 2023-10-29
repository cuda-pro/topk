CC=gcc
BUILD_TYPE ?= Debug
OPTIMIZE_CFLAGS?=-O3
ifeq ($(BUILD_TYPE),Debug)
    OPTIMIZE_CFLAGS=-O0
endif

DOC_CN?=10
QUERY_CN?=10

init:
	mkdir -p bin

gen:
	@bash gen.sh $(DOC_CN)
	@bash gen_querys.sh $(QUERY_CN)

build_cpu: init
	g++ ./main.cpp -o ./bin/query_doc_scoring_cpu  -I./ -std=c++11 -pthread \
		$(OPTIMIZE_CFLAGS) \
		-g 

build_cpu_concurency: init
	g++ ./main.cpp -o ./bin/query_doc_scoring_cpu_concurency  -I./ -std=c++11 -pthread -DCPU_CONCURENCY \
		$(OPTIMIZE_CFLAGS) \
		-g 

build_cpu_gpu: init
	nvcc ./main.cpp ./topk.cu -o ./bin/query_doc_scoring_cpu_gpu  \
		-I./ -L/usr/local/cuda/lib64 -lcudart -lcuda \
		$(OPTIMIZE_CFLAGS) \
		-g

build_cpu_concurency_gpu: init
	nvcc ./main.cpp ./topk.cu -o ./bin/query_doc_scoring_cpu_concurency_gpu  \
		-I./ -L/usr/local/cuda/lib64 -lcudart -lcuda \
		-DCPU_CONCURENCY \
		$(OPTIMIZE_CFLAGS) \
		-g

build_examples: init
	g++ -o bin/example_threadpool example_threadpool.cpp -std=c++11 -pthread \
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