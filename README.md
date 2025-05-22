# cuda-exp

A collection of CUDA programming examples to learn GPU programming with NVIDIA CUDA.

make sure change the gpu architecture `sm_61` to your own gpu architecture in Makefile

## Examples

- **basic01.cu**: Introduction to CUDA programming with a vector addition example
- **basic02.cu**: Demonstration of CUDA PTX inline assembly with a vector multiplication example
- **basic03.cu**: Comprehensive comparison of GPU programming methods including CUDA, PTX, Thrust, Unified Memory, Shared Memory, CUDA Streams, and Dynamic Parallelism using matrix multiplication
- **basic04.cu**: Detailed exploration of GPU organization hierarchy including hardware architecture, thread/block/grid structure, memory hierarchy, and execution model
- **basic05.cu**: Implementing a basic neural network forward pass on GPU with CUDA
- **basic06.cu**: GPU-accelerated convolution operations for CNN with shared memory optimization
- **basic07.cu**: CUDA implementation of attention mechanism for transformer models

## Upcoming ML/AI GPU Tutorials

- **basic08.cu**: Mixed-precision training with Tensor Cores using CUDA

Each tutorial will include comprehensive documentation explaining the concepts, implementation details, and optimization techniques used in ML/AI workloads on GPUs.