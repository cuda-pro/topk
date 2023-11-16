# https://www.scivision.dev/install-nvidia-hpc-free-compiler/
# hpc-compiler

nvroot=/opt/nvidia/hpc_sdk/Linux_x86_64/2022/

[[ ! -d ${nvroot} ]] && echo "ERROR: ${nvroot} not found" && exit 1

export PATH=$PATH:${nvroot}/compilers/bin/

export CC=nvc CXX=nvc++ FC=nvfortran

export MPI_ROOT=${nvroot}/comm_libs/openmpi4/

nvfortran --version
nvc --version
nvc++ --version
