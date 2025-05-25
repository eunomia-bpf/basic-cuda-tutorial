# CPU and GPU Profiling Boundaries: What to Measure Where

This document explores the boundary between CPU and GPU profiling, examining which operations can be effectively measured on the CPU side versus which require GPU-side instrumentation. We'll also discuss how advanced CPU-side function hooking techniques like eBPF can complement GPU profiling.

## Table of Contents

1. [The CPU-GPU Boundary](#the-cpu-gpu-boundary)
2. [CPU-Side Measurable Operations](#cpu-side-measurable-operations)
3. [GPU-Side Measurable Operations](#gpu-side-measurable-operations)
4. [Hooking CPU-Side Functions with eBPF](#hooking-cpu-side-functions-with-ebpf)
5. [Integrated Profiling Approaches](#integrated-profiling-approaches)
6. [Case Studies](#case-studies)
7. [Future Directions](#future-directions)
8. [References](#references)

## The CPU-GPU Boundary

Modern GPU computing involves a complex interplay between host (CPU) and device (GPU) operations. Understanding where to place profiling instrumentation depends on what aspects of performance you're measuring:

```
┌────────────────────────┐                  ┌────────────────────────┐
│        CPU Side        │                  │        GPU Side        │
│                        │                  │                        │
│  ┌──────────────────┐  │                  │  ┌──────────────────┐  │
│  │ Application Code │  │                  │  │  Kernel Execution │  │
│  └────────┬─────────┘  │                  │  └────────┬─────────┘  │
│           │            │                  │           │            │
│  ┌────────▼─────────┐  │                  │  ┌────────▼─────────┐  │
│  │   CUDA Runtime   │  │                  │  │   Warp Scheduler  │  │
│  └────────┬─────────┘  │                  │  └────────┬─────────┘  │
│           │            │                  │           │            │
│  ┌────────▼─────────┐  │  ┌─────────┐     │  ┌────────▼─────────┐  │
│  │   CUDA Driver    │◄─┼──┤PCIe Bus │────►│  │ Memory Controller │  │
│  └────────┬─────────┘  │  └─────────┘     │  └────────┬─────────┘  │
│           │            │                  │           │            │
│  ┌────────▼─────────┐  │                  │  ┌────────▼─────────┐  │
│  │ System Software  │  │                  │  │   GPU Hardware   │  │
│  └──────────────────┘  │                  │  └──────────────────┘  │
└────────────────────────┘                  └────────────────────────┘
```

## CPU-Side Measurable Operations

The following operations can be effectively measured from the CPU side:

### 1. CUDA API Call Latency

- **Kernel Launch Overhead**: Time from the API call to when the kernel begins execution
- **Memory Allocation**: Time spent in `cudaMalloc`, `cudaFree`, etc.
- **Host-Device Transfers**: Duration of `cudaMemcpy` operations
- **Synchronization Points**: Time spent in `cudaDeviceSynchronize`, `cudaStreamSynchronize`

### 2. Resource Management

- **Memory Usage**: Tracking GPU memory allocation and deallocation patterns
- **Stream Creation**: Overhead of creating and destroying CUDA streams
- **Context Switching**: Time spent switching between CUDA contexts

### 3. CPU-GPU Interaction Patterns

- **API Call Frequency**: Rate and pattern of CUDA API calls
- **CPU Wait Time**: Time CPU spends waiting for GPU operations
- **I/O and GPU Overlap**: How I/O operations interact with GPU utilization

### 4. System-Level Metrics

- **PCIe Traffic**: Volume and timing of data transferred over PCIe
- **Power Consumption**: System-wide power usage correlated with GPU activity
- **Thermal Effects**: Temperature changes that may affect throttling

### Tools and Techniques for CPU-Side Measurement

- **CUPTI API Callbacks**: Hook into CUDA API calls via the CUPTI interface
- **Binary Instrumentation**: Tools like Pin or DynamoRIO to intercept functions
- **Interposing Libraries**: Custom libraries that intercept CUDA API calls
- **eBPF**: Linux's extended Berkeley Packet Filter for kernel-level tracing
- **Performance Counters**: Hardware-level counters accessible via PAPI or similar

## GPU-Side Measurable Operations

The following operations require GPU-side instrumentation:

### 1. Kernel Execution Details

- **Instruction Mix**: Types and frequencies of instructions executed
- **Warp Execution Efficiency**: Percentage of active threads in warps
- **Divergence Patterns**: Frequency and impact of branch divergence
- **Instruction-Level Parallelism**: Achieved ILP within each thread

### 2. Memory System Performance

- **Memory Access Patterns**: Coalescing efficiency, stride patterns
- **Cache Hit Rates**: L1/L2/Texture cache effectiveness
- **Bank Conflicts**: Shared memory access conflicts
- **Memory Divergence**: Divergent memory access patterns

### 3. Hardware Utilization

- **SM Occupancy**: Active warps relative to maximum capacity
- **Special Function Usage**: Utilization of SFUs, tensor cores, etc.
- **Memory Bandwidth**: Achieved vs. theoretical memory bandwidth
- **Compute Throughput**: FLOPS or other compute metrics

### 4. Synchronization Effects

- **Block Synchronization**: Time spent in `__syncthreads()`
- **Atomic Operation Contention**: Impact of atomic operations
- **Warp Scheduling Decisions**: How warps are scheduled on SMs

### Tools and Techniques for GPU-Side Measurement

- **SASS/PTX Analysis**: Examining low-level assembly code
- **Hardware Performance Counters**: GPU-specific counters for various metrics
- **Kernel Instrumentation**: Adding timing code directly to kernels
- **Specialized Profilers**: Nsight Compute, Nvprof for deep GPU insights
- **Visual Profilers**: Timeline views of kernel execution

## Hooking CPU-Side Functions with eBPF

eBPF (extended Berkeley Packet Filter) provides powerful mechanisms for tracing and monitoring system behavior on Linux without modifying source code. For GPU workloads, eBPF can be particularly valuable for correlating CPU-side activity with GPU performance.

### What is eBPF?

eBPF is a technology that allows running sandboxed programs in the Linux kernel without changing kernel source code or loading kernel modules. It's widely used for performance analysis, security monitoring, and networking.

### eBPF for GPU Workload Profiling

While eBPF cannot directly instrument code running on the GPU, it excels at monitoring the CPU-side interactions with the GPU:

#### 1. Tracing CUDA Driver Interactions

```c
// Example eBPF program to trace CUDA driver function calls
int trace_cuLaunchKernel(struct pt_regs *ctx) {
    u64 ts = bpf_ktime_get_ns();
    struct data_t data = {};
    
    data.timestamp = ts;
    data.pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    
    // Capture function arguments
    data.gridDimX = PT_REGS_PARM2(ctx);
    data.gridDimY = PT_REGS_PARM3(ctx);
    data.gridDimZ = PT_REGS_PARM4(ctx);
    
    events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}
```

#### 2. Correlating System Events with GPU Activity

eBPF can monitor:
- File I/O operations that may affect GPU data transfers
- Scheduler decisions that impact CPU-GPU coordination
- Memory management events relevant to GPU buffer handling

#### 3. Building a Complete Picture

By combining eBPF-gathered CPU-side data with GPU profiling information:
- Track data from its source to the GPU and back
- Identify system-level bottlenecks affecting GPU performance
- Understand scheduling issues that create GPU idle time

### eBPF Tools for GPU Workloads

1. **BCC (BPF Compiler Collection)**: Provides Python interfaces for eBPF programs
2. **bpftrace**: High-level tracing language for Linux eBPF
3. **Custom eBPF programs**: Tailored specifically for CUDA/GPU workloads

Example of tracing CUDA memory operations with bpftrace:

```
bpftrace -e '
uprobe:/usr/lib/libcuda.so:cuMemAlloc {
    printf("cuMemAlloc called: size=%llu, pid=%d, comm=%s\n", 
           arg1, pid, comm);
    @mem_alloc_bytes = hist(arg1);
}
uprobe:/usr/lib/libcuda.so:cuMemFree {
    printf("cuMemFree called: pid=%d, comm=%s\n", pid, comm);
}
'
```

## Integrated Profiling Approaches

Effective GPU application profiling requires integrating data from both CPU and GPU sides:

### 1. Timeline Correlation

Aligning events across CPU and GPU timelines to identify:
- **Kernel Launch Delays**: Gap between CPU request and GPU execution
- **Transfer-Compute Overlap**: Effectiveness of asynchronous operations
- **CPU-GPU Synchronization Points**: Where the CPU waits for the GPU

### 2. Bottleneck Identification

Using combined data to pinpoint whether bottlenecks are:
- **CPU-Bound**: CPU preparation of data or launch overhead
- **Transfer-Bound**: PCIe or memory bandwidth limitations
- **GPU Compute-Bound**: Kernel algorithm efficiency
- **GPU Memory-Bound**: GPU memory access patterns

### 3. Multi-Level Optimization Strategy

Developing a holistic optimization approach:
1. **System Level**: PCIe configuration, power settings, CPU affinity
2. **Application Level**: Kernel launch patterns, memory management
3. **Algorithm Level**: Kernel implementation, memory access patterns
4. **Instruction Level**: PTX/SASS optimizations

## Case Studies

### Case Study 1: Deep Learning Training Framework

In a deep learning framework, we observed:

- **CPU-Side Profiling**: Identified inefficient data preprocessing before GPU transfers
- **GPU-Side Profiling**: Showed high utilization but poor memory access patterns
- **eBPF Analysis**: Revealed that Linux page cache behavior was causing unpredictable data transfer timing

**Solution**: Implemented pinned memory with explicit prefetching guided by eBPF-gathered access patterns, resulting in 35% throughput improvement.

### Case Study 2: Real-time Image Processing Pipeline

For a real-time image processing application:

- **CPU-Side Profiling**: Showed bursty kernel launches causing GPU idle time
- **GPU-Side Profiling**: Indicated good kernel efficiency but poor occupancy
- **eBPF Analysis**: Discovered thread scheduling issues on CPU affecting launch timing

**Solution**: Used eBPF insights to implement CPU thread pinning and reorganized the pipeline, achieving consistent frame rates with 22% less end-to-end latency.

## Future Directions

The boundary between CPU and GPU profiling continues to evolve:

1. **Unified Memory Profiling**: As unified memory becomes more prevalent, new tools are needed to track page migrations and access patterns

2. **System-on-Chip Integration**: As GPUs become more integrated with CPUs, profiling boundaries will blur, requiring new approaches

3. **Multi-GPU Systems**: Distributed training and inference across multiple GPUs introduces new profiling challenges

4. **AI-Assisted Profiling**: Using machine learning to automatically identify patterns and suggest optimizations across the CPU-GPU boundary

## References

1. NVIDIA. "CUPTI: CUDA Profiling Tools Interface." [https://docs.nvidia.com/cuda/cupti/](https://docs.nvidia.com/cuda/cupti/)

2. Gregg, Brendan. "BPF Performance Tools: Linux System and Application Observability." Addison-Wesley Professional, 2019.

3. NVIDIA. "Nsight Systems User Guide." [https://docs.nvidia.com/nsight-systems/](https://docs.nvidia.com/nsight-systems/)

4. Awan, Ammar Ali, et al. "Characterizing Machine Learning I/O Workloads on NVME and CPU-GPU Systems." IEEE International Parallel and Distributed Processing Symposium Workshops, 2022.

5. The eBPF Foundation. "What is eBPF?" [https://ebpf.io/what-is-ebpf/](https://ebpf.io/what-is-ebpf/)

6. NVIDIA. "Tools for Profiling CUDA Applications." [https://developer.nvidia.com/tools-overview](https://developer.nvidia.com/tools-overview) 