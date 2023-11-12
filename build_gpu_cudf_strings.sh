ROOT_DIR=$(cd $(dirname $0); pwd)
cd $ROOT_DIR

rm -rf ./cudf
git clone https://github.com/rapidsai/cudf.git
cd cudf && ./build.sh libcudf
cd -

rm -rf ./spdlog
git clone https://github.com/gabime/spdlog.git
cp -r ./spdlog/include/spdlog/fmt/bundled /include/spdlog/fmt/

mkdir -p bin
nvcc ./src/main.cpp ./src/readfile.cu ./src/topk_doc_cudf_strings.cu -o ./bin/query_doc_scoring \
        -I./src/ \
	-std=c++17 --expt-relaxed-constexpr \
	-L/usr/local/cuda/lib64 -lcudart -lcuda \
	-L/lib -lcudf -I/include  \
	-O3 \
	-DFMT_HEADER_ONLY -DGPU -DPIO_TOPK \
	-g

if [ $? -eq 0 ]; then
  echo "build success"
else
  echo "build fail"
fi
