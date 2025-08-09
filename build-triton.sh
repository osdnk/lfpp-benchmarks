#!/bin/bash -l
#SBATCH --mem=10G
#SBATCH --cpus-per-task 4
#SBATCH --time=0:01:20
#SBATCH --partition=gpu-h100-80g
module load gcc cmake
make hexl-triton
make wrapper
export LD_LIBRARY_PATH=./hexl-bindings/hexl/build/hexl/lib:$(pwd)
export RUSTFLAGS="-C linker=gcc"
cargo run
cargo test
cargo bench



