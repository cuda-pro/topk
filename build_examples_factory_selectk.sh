set -e

ROOT_DIR=$(cd $(dirname $0); pwd)
cd $ROOT_DIR

mkdir -p bin
ARCH=70
[ -n "$1" ] && ARCH=$1
[ ! -f "./third_party/done.txt" ] && bash -x download_third_party.sh
[ ! -f "./lib/libfaiss.so" ] && make build_3d_faiss NVCCSTD=c++14 ARCH=${ARCH} CXXFLAGS="-std=c++14 -fPIC" BUILD_TYPE=Release
[ ! -f "./lib/libgpu_selection.so" ] && make build_3d_gpu_selection ARCH=${ARCH}

nvcc -o bin/example_factory_selectk example_factory_selectk.cu \
	-O2 -std=c++17 -Xcompiler "-Wall -Wextra -Wno-unused-parameter" \
	--expt-relaxed-constexpr --extended-lambda \
	-arch=sm_${ARCH} -gencode=arch=compute_${ARCH},code=sm_${ARCH} \
	-I./include -I./third_party \
	-isystem ./third_party/DrTopKSC/bitonic/LargerKVersions/largerK/ \
	-I./third_party/DrTopKSC/baseline+filter+beta+shuffle/ \
	-I./third_party/gpu_selection/include -I./third_party/gpu_selection/lib \
	-L/usr/local/cuda/lib64 -lcudart -lcuda \
	-L./lib -lfaiss -Xlinker -rpath=./lib \
	-L./lib -lgpu_selection -Xlinker -rpath=./lib \
	-L./lib -lgridselect -Xlinker -rpath=./lib \
 	-g

