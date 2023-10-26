
init:
	mkdir -p bin
	mkdir -p testdata/query

gen: init
	@bash -x gen.sh 3 10
	@bash -x gen.sh 9 "" 

build_cpu: init
	g++ ./main.cpp -o ./bin/query_doc_scoring_cpu  -I./ -std=c++11 -pthread -O3 -g 

build_gpu:
	nvcc ./main.cpp ./topk.cu -o ./bin/query_doc_scoring_gpu  -I./ -L/usr/local/cuda/lib64 -lcudart -lcuda -O3 -g