ROOT_DIR=$(cd $(dirname $0); pwd)
cd $ROOT_DIR

rm -rf ./raft
git clone https://github.com/rapidsai/raft.git
cd raft && ./build.sh libraft
cd -

rm -rf ./spdlog
git clone https://github.com/gabime/spdlog.git
cp -r ./spdlog/include/spdlog/fmt/bundled /include/spdlog/fmt/

mkdir -p bin
nvcc ./src/main.cpp ./src/topk_raft_selectk.cu -o ./bin/query_doc_scoring \
    -I./src/ \
	-std=c++17 --expt-relaxed-constexpr --extended-lambda -arch=sm_70 \
	-L/usr/local/cuda/lib64 -lcudart -lcuda \
	-L/lib -lraft -I/include  \
	-O3 \
	-DGPU -DFMT_HEADER_ONLY \
	-g

if [ $? -eq 0 ]; then
  echo "build success"
else
  echo "build fail"
fi
