set -e

ROOT_DIR=$(cd $(dirname $0); pwd)
cd $ROOT_DIR

# git for GFW
#git config --global url."https://mirror.ghproxy.com/https://".insteadOf https://
#git config --global url."https://hub.fastgit.org".insteadOf https://github.com
#git config --global url."https://gitclone.com/".insteadOf https://

RAPIDSAI_DIR=$HOME/rapidsai

if [ ! -f "$HOME/rapidsai/lib/libcudf.a" ]; then
    #rm -rf ./cudf-23.10.00
    #wget https://github.com/rapidsai/cudf/archive/refs/tags/v23.10.00.tar.gz -O cudf.tar.gz
    #tar -zxvf cudf.tar.gz
    #cd cudf-23.10.00 && INSTALL_PREFIX=$RAPIDSAI_DIR ./build.sh libcudf --cmake-args=\"-DBUILD_SHARED_LIBS=OFF\"
    #cd -

    rm -rf cudf && git clone -b branch-23.10 https://github.com/weedge/cudf.git
    cd cudf && INSTALL_PREFIX=$RAPIDSAI_DIR ./build.sh libcudf --cmake-args=\"-DBUILD_SHARED_LIBS=OFF\"
    cd -

    # use v23.10 rmm
    rm -rf ./rmm-23.10.00
    wget https://github.com/rapidsai/rmm/archive/refs/tags/v23.10.00.tar.gz
    tar -zxvf rmm.tar.gz
    cp -r rmm/include/rmm /root/rapidsai/include/
fi

if [ ! -f "$HOME/rapidsai/lib/libraft.a" ]; then
    apt install ninja-build -y
    rm -rf ./raft-23.10.00
    wget https://github.com/rapidsai/raft/archive/refs/tags/v23.10.00.tar.gz -O raft.tar.gz
    tar -zxvf raft.tar.gz
    cd raft-23.10.00 && ./build.sh libraft --compile-static-lib
    cp -r cpp/build/install/include/* $RAPIDSAI_DIR/include/
    cp -r cpp/build/install/lib/* $RAPIDSAI_DIR/lib/
    cd -
fi

if [ ! -d "$RAPIDSAI_DIR/include/spdlog/fmt/bundled" ]; then
    rm -rf ./spdlog-1.11.0
    wget https://github.com/gabime/spdlog/archive/refs/tags/v1.11.0.tar.gz -O spdlog.tar.gz
    tar -zxvf spdlog.tar.gz 
    cp -r ./spdlog-1.11.0/include/spdlog/fmt/bundled $RAPIDSAI_DIR/include/spdlog/fmt/
fi

# cuda-toolkit include this
#if [ ! -d "$RAPIDSAI_DIR/cccl" ]; then
#    wget https://github.com/NVIDIA/cccl/archive/refs/tags/v2.2.0.tar.gz -O cccl.tar.gz
#    tar zxvf cccl.tar.gz 
#    mkdir -p $RAPIDSAI_DIR/cccl && cp -r cccl-2.2.0/* $RAPIDSAI_DIR/cccl/
#fi

if [ $? -eq 0 ]; then
    cd $RAPIDSAI_DIR && zip -r -v rapidsai.zip  ./include ./lib
    echo "build ok"
else
    echo "build fail"
fi
