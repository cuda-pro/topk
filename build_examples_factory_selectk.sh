ROOT_DIR=$(cd $(dirname $0); pwd)
cd $ROOT_DIR

mkdir -p bin

bash -x download.sh
make build_3d_faiss
make build_3d_gpu_selectk

nvcc -o example_factory_selectk example_factory_selectk.cu \
	-O2 -std=c++17 -Xcompiler "-Wall -Wextra -Wno-unused-parameter" \
	--expt-relaxed-constexpr --extended-lambda -arch=sm_80 \
	-I./include -I./third_party/faiss \
	-isystem ./third_party/DrTopKSC/bitonic/LargerKVersions/largerK/ \
	-I./third_party/DrTopKSC/baseline+filter+beta+shuffle/ \
	-I./third_party/gpu_selection/include -I./third_party/gpu_selection/lib \
	-L/usr/local/cuda/lib64 -lcudart -lcuda -lcurand \
	-L./third_party/faiss -lfaiss -Xlinker -rpath=./lib \
	-L./third_party/gpu_selection -lgpu_selection -Xlinker -rpath=./lib \
	-L./third_party -lgridselect -Xlinker -rpath=./lib \
 	-g

