#!/bin/sh

ROOT_DIR=$(cd $(dirname $0); pwd)
cd $ROOT_DIR

rm -rf third_party
mkdir -p third_party 
cd third_party

# faiss block select topk source code
faiss_version=1.7.3
wget https://github.com/facebookresearch/faiss/archive/refs/tags/v${faiss_version}.zip -O faiss-${faiss_version}.zip
unzip faiss-${faiss_version}.zip
mv faiss-${faiss_version}/faiss ./
rm -r faiss-${faiss_version}
rm faiss-${faiss_version}.zip

# bucket-based select topk source code
# fork from https://github.com/upsj/gpu_selection.git 
# change uint32 -> int
git clone https://github.com/weedge/gpu_selection.git

# Dr.topk source code
# fork from https://github.com/Anil-Gaihre/DrTopKSC.git
# change radix/bitonic select
git clone https://github.com/weedge/DrTopKSC.git


echo "third party download success" > done.txt