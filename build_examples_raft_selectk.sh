ROOT_DIR=$(cd $(dirname $0); pwd)
cd $ROOT_DIR

#sh build_deps_rapidsai.sh

RAPIDSAI_DIR=$HOME/rapidsai
mkdir -p bin

nvcc ./src/example_raft_selectk.cu -o ./bin/example_raft_selectk_null \
    -I./src/ \
	-std=c++17 --expt-relaxed-constexpr --extended-lambda -arch=sm_70 \
	-L/usr/local/cuda/lib64 -lcudart -lcuda \
	-L$RAPIDSAI_DIR/lib -lraft -I$RAPIDSAI_DIR/include \
	-O3 \
	-DNULL_OPTIONAL -DFMT_HEADER_ONLY \
	-g
   
if [ $? -eq 0 ]; then
  echo "build success"
  bin/example_raft_selectk_null 110
else
  echo "build fail"
fi

nvcc ./src/example_raft_selectk.cu -o ./bin/example_raft_selectk \
    -I./src/ \
	-std=c++17 --expt-relaxed-constexpr --extended-lambda -arch=sm_70 \
	-L/usr/local/cuda/lib64 -lcudart -lcuda \
	-L$RAPIDSAI_DIR/lib -lraft -I$RAPIDSAI_DIR/include \
	-O3 \
	-DFMT_HEADER_ONLY \
	-g
if [ $? -eq 0 ]; then
  echo "build success"
  bin/example_raft_selectk 110
else
  echo "build fail"
  exit 1
fi
