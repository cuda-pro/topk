ROOT_DIR=$(cd $(dirname $0); pwd)
cd $ROOT_DIR

# for GFW
#git config --global url."https://mirror.ghproxy.com/https://".insteadOf https://
#git config --global url."https://hub.fastgit.org".insteadOf https://github.com
#git config --global url."https://gitclone.com/".insteadOf https://

RAPIDSAI_DIR=$HOME/rapidsai

rm -rf ./cudf-23.10.00
wget https://github.com/rapidsai/cudf/archive/refs/tags/v23.10.00.tar.gz -O cudf.tar.gz
tar -zxvf cudf.tar.gz
cd cudf-23.10.00 && INSTALL_PREFIX=$RAPIDSAI_DIR ./build.sh libcudf --cmake-args=\"-DBUILD_SHARED_LIBS=OFF\"
cd -

apt install ninja-build -y
rm -rf ./raft-23.10.00
wget https://github.com/rapidsai/raft/archive/refs/tags/v23.10.00.tar.gz -O raft.tar.gz
tar -zxvf raft.tar.gz
cd raft-23.10.00 && ./build.sh libraft --compile-static-lib
cp -r build/install/include/* $RAPIDSAI_DIR/include/
cp -r build/install/lib/* $RAPIDSAI_DIR/lib/
cd -

rm -rf ./spdlog-1.11.0
wget https://github.com/gabime/spdlog/archive/refs/tags/v1.11.0.tar.gz -O spdlog.tar.gz
tar -zxvf spdlog.tar.gz 
cp -r ./spdlog-1.11.0/include/spdlog/fmt/bundled $RAPIDSAI_DIR/include/spdlog/fmt/


if [ $? -eq 0 ]; then
    cd $RAPIDSAI_DIR zip -r -v rapidsai.zip  ./include ./lib
    echo "build ok"
else
    echo "build fail"
fi
