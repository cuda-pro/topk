ROOT_DIR=$1
if [ -z "$ROOT_DIR" ]; then
        ROOT_DIR=$(cd $(dirname $0); pwd)
fi
echo $ROOT_DIR

mkdir -p bin
nvcc $ROOT_DIR/src/main.cpp $ROOT_DIR/src/topk.cu -o $ROOT_DIR/bin/query_doc_scoring \
    -I$ROOT_DIR/src/ \
	-std=c++11 --expt-relaxed-constexpr \
	-fopenmp \
	-L/usr/local/cuda/lib64 -lcudart -lcuda \
	-O3 \
	-DGPU \
	-g

if [ $? -eq 0 ]; then
  echo "build success"
else
  echo "build fail"
fi
