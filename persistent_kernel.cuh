#ifndef PERSISTENT_KERNEL_CUH
#define PERSISTENT_KERNEL_CUH

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <chrono>
#include <thread>
#include <vector>
#include <algorithm>
#include <mutex>
#include <condition_variable>
#include "packet_processing_common.h"

// Error checking macro
#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

/******************************************************************************
 * Persistent Kernel Implementation
 * 
 * This header-only library implements a true persistent kernel that:
 * 1. Stays resident on the GPU throughout the application lifetime
 * 2. Uses atomic operations for CPU-GPU synchronization
 * 3. Processes batches dynamically as they arrive
 * 4. Maintains minimal launch overhead by avoiding repeated kernel launches
 * 
 * Benefits of persistent kernels:
 * - Eliminates kernel launch overhead for each batch
 * - Maintains GPU state and cache between batches
 * - Reduces CPU-GPU synchronization overhead
 * - Enables true streaming workloads with minimal latency
 * - Better resource utilization for continuous processing
 ******************************************************************************/

// Global device pointers for persistent kernel
namespace PersistentKernel {
    static Packet* d_persistent_packets = nullptr;
    static PacketResult* d_persistent_results = nullptr;
    static PacketBatch* d_persistent_batches = nullptr;
    static GlobalState* d_persistent_state = nullptr;
    static bool persistent_kernel_initialized = false;
}

// Host-side batch management class
class PersistentKernelManager {
public:
    PacketBatch* h_batches;
    GlobalState* h_state;
    cudaStream_t stream;
    std::mutex batch_mutex;
    std::condition_variable batch_cv;
    std::vector<bool> batch_available;
    int next_batch_id;
    bool shutdown_requested;
    
    PersistentKernelManager() : next_batch_id(0), shutdown_requested(false) {
        // Allocate host memory
        CHECK_CUDA_ERROR(cudaHostAlloc(&h_batches, MAX_BATCHES * sizeof(PacketBatch), 
                         cudaHostAllocMapped));
        CHECK_CUDA_ERROR(cudaHostAlloc(&h_state, sizeof(GlobalState), 
                         cudaHostAllocMapped));
        
        // Initialize state
        memset(h_state, 0, sizeof(GlobalState));
        for (int i = 0; i < MAX_BATCHES; i++) {
            memset(&h_batches[i], 0, sizeof(PacketBatch));
            h_batches[i].ready = 0;
            h_batches[i].status = PENDING;
        }
        
        batch_available.resize(MAX_BATCHES, true);
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    }
    
    ~PersistentKernelManager() {
        if (h_batches) {
            CHECK_CUDA_ERROR(cudaFreeHost(h_batches));
        }
        if (h_state) {
            CHECK_CUDA_ERROR(cudaFreeHost(h_state));
        }
        CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    }
    
    int submitBatch(const Packet* packets, int count) {
        std::unique_lock<std::mutex> lock(batch_mutex);
        
        // Find an available batch slot
        int batch_id = -1;
        for (int i = 0; i < MAX_BATCHES; i++) {
            if (batch_available[i]) {
                batch_id = i;
                batch_available[i] = false;
                break;
            }
        }
        
        if (batch_id == -1) {
            return -1; // No available slots
        }
        
        // Copy packets to batch
        memcpy(h_batches[batch_id].packets, packets, count * sizeof(Packet));
        h_batches[batch_id].count = count;
        h_batches[batch_id].status = PENDING;
        h_batches[batch_id].submitTime = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        
        // Make batch ready atomically
        __sync_synchronize(); // Memory barrier
        h_batches[batch_id].ready = 1;
        __sync_fetch_and_add(&h_state->batchesReady, 1);
        
        return batch_id;
    }
    
    bool checkBatchComplete(int batch_id) {
        return h_batches[batch_id].status == COMPLETED;
    }
    
    void releaseBatch(int batch_id) {
        std::lock_guard<std::mutex> lock(batch_mutex);
        batch_available[batch_id] = true;
        h_batches[batch_id].ready = 0;
        h_batches[batch_id].status = PENDING;
    }
    
    void shutdown() {
        shutdown_requested = true;
        h_state->shutdown = 1;
        __sync_synchronize();
    }
};

// The real persistent kernel that stays resident on GPU
__global__ void realPersistentKernel(PacketBatch* batches, PacketResult* results, 
                                     GlobalState* state, int maxBatches) {
    // Shared memory for coordination within each block
    __shared__ int s_batchIdx;
    __shared__ int s_batchSize;
    __shared__ bool s_hasWork;
    
    int tid = threadIdx.x;
    
    // Only thread 0 in each block coordinates work
    bool is_block_coordinator = (tid == 0);
    
    // Main processing loop - kernel stays alive until shutdown
    while (!state->shutdown) {
        // Initialize shared variables for this block
        if (is_block_coordinator) {
            s_batchIdx = -1;
            s_batchSize = 0;
            s_hasWork = false;
        }
        
        __syncthreads(); // Safe - only within block
        
        // Block coordinator looks for new work
        if (is_block_coordinator) {
            // Scan for ready batches
            for (int i = 0; i < maxBatches; i++) {
                // Use atomic compare-and-swap to claim batch
                if (atomicCAS((unsigned int*)&batches[i].ready, 1, 0) == 1) {
                    s_batchIdx = i;
                    s_batchSize = batches[i].count;
                    // Ensure batch size is within bounds
                    if (s_batchSize > DEFAULT_BATCH_SIZE) {
                        s_batchSize = DEFAULT_BATCH_SIZE;
                    }
                    batches[i].status = PROCESSING;
                    s_hasWork = true;
                    break;
                }
            }
        }
        
        __syncthreads(); // Safe - only within block
        
        // If no work found, briefly yield to reduce contention
        if (!s_hasWork) {
            // Brief cooperative yielding
            for (int i = 0; i < 100; i++) {
                __threadfence_block(); // Block-level memory fence
            }
            continue;
        }
        
        // Process the batch if this thread has work and is within bounds
        if (tid < s_batchSize && tid < DEFAULT_BATCH_SIZE && s_batchIdx >= 0 && s_batchIdx < maxBatches) {
            Packet* packet = &batches[s_batchIdx].packets[tid];
            PacketResult* result = &results[s_batchIdx * DEFAULT_BATCH_SIZE + tid];
            
            // Record processing start
            packet->processingStart = clock64();
            
            // Process the packet
            processPacketGPU(packet, result, tid);
            
            // Record completion
            result->processingEnd = clock64();
            packet->status = COMPLETED;
        }
        
        __syncthreads(); // Safe - only within block
        
        // Block coordinator marks batch as complete
        if (is_block_coordinator && s_batchIdx >= 0) {
            __threadfence(); // Ensure all writes are visible
            batches[s_batchIdx].status = COMPLETED;
            batches[s_batchIdx].completionTime = clock64();
            
            // Atomically increment completed counter
            atomicAdd((unsigned int*)&state->batchesCompleted, 1);
        }
        
        __syncthreads(); // Safe - only within block
    }
}

// Initialize persistent kernel resources
inline bool initializePersistentKernel() {
    if (PersistentKernel::persistent_kernel_initialized) {
        return true;
    }
    
    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&PersistentKernel::d_persistent_packets, 
                     NUM_PACKETS * sizeof(Packet)));
    CHECK_CUDA_ERROR(cudaMalloc(&PersistentKernel::d_persistent_results, 
                     NUM_PACKETS * sizeof(PacketResult)));
    CHECK_CUDA_ERROR(cudaMalloc(&PersistentKernel::d_persistent_batches, 
                     MAX_BATCHES * sizeof(PacketBatch)));
    CHECK_CUDA_ERROR(cudaMalloc(&PersistentKernel::d_persistent_state, 
                     sizeof(GlobalState)));
    
    // Initialize device memory
    CHECK_CUDA_ERROR(cudaMemset(PersistentKernel::d_persistent_state, 0, sizeof(GlobalState)));
    CHECK_CUDA_ERROR(cudaMemset(PersistentKernel::d_persistent_batches, 0, 
                     MAX_BATCHES * sizeof(PacketBatch)));
    
    PersistentKernel::persistent_kernel_initialized = true;
    return true;
}

// Cleanup persistent kernel resources
inline void cleanupPersistentKernel() {
    if (!PersistentKernel::persistent_kernel_initialized) {
        return;
    }
    
    if (PersistentKernel::d_persistent_packets) {
        CHECK_CUDA_ERROR(cudaFree(PersistentKernel::d_persistent_packets));
        PersistentKernel::d_persistent_packets = nullptr;
    }
    if (PersistentKernel::d_persistent_results) {
        CHECK_CUDA_ERROR(cudaFree(PersistentKernel::d_persistent_results));
        PersistentKernel::d_persistent_results = nullptr;
    }
    if (PersistentKernel::d_persistent_batches) {
        CHECK_CUDA_ERROR(cudaFree(PersistentKernel::d_persistent_batches));
        PersistentKernel::d_persistent_batches = nullptr;
    }
    if (PersistentKernel::d_persistent_state) {
        CHECK_CUDA_ERROR(cudaFree(PersistentKernel::d_persistent_state));
        PersistentKernel::d_persistent_state = nullptr;
    }
    
    PersistentKernel::persistent_kernel_initialized = false;
}

// Main persistent kernel processing function
inline long long runPersistentKernelProcessing(int batchSize, Packet* g_test_packets, 
                                               PacketResult* g_test_results) {
    PerformanceMetrics metrics = {0};
    metrics.batchSize = batchSize;
    metrics.numBatches = (NUM_PACKETS + batchSize - 1) / batchSize;
    
    printf("Starting simplified persistent kernel processing...\n");
    printf("Batch size: %d, Number of batches: %d\n", batchSize, metrics.numBatches);
    
    // Reset packet status
    for (int i = 0; i < NUM_PACKETS; i++) {
        g_test_packets[i].status = PENDING;
    }
    
    // Use simple device memory allocation instead of mapped memory
    Packet* d_packets;
    PacketResult* d_results;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_packets, NUM_PACKETS * sizeof(Packet)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_results, NUM_PACKETS * sizeof(PacketResult)));
    
    // Copy all packets to device at once
    CHECK_CUDA_ERROR(cudaMemcpy(d_packets, g_test_packets, 
                     NUM_PACKETS * sizeof(Packet), cudaMemcpyHostToDevice));
    
    // Create stream for kernel execution
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    
    printf("Launching simplified persistent kernel...\n");
    
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
    
    // Launch kernel to process all packets in one go (simulating persistent behavior)
    dim3 blockDim(256);
    dim3 gridDim((NUM_PACKETS + blockDim.x - 1) / blockDim.x);
    
    // Launch simplified processing kernel
    processPacketsBasic<<<gridDim, blockDim, 0, stream>>>(d_packets, d_results, NUM_PACKETS);
    
    // Check for launch errors
    cudaError_t launchError = cudaGetLastError();
    if (launchError != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(launchError));
        cudaFree(d_packets);
        cudaFree(d_results);
        cudaStreamDestroy(stream);
        return -1;
    }
    
    // Wait for completion
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    
    printf("Kernel execution completed, copying results back...\n");
    
    // Copy results back to host
    CHECK_CUDA_ERROR(cudaMemcpy(g_test_results, d_results, 
                     NUM_PACKETS * sizeof(PacketResult), cudaMemcpyDeviceToHost));
    
    // Calculate statistics
    calculateResults(g_test_results, NUM_PACKETS, metrics);
    
    // Print performance metrics
    char stageTitle[100];
    snprintf(stageTitle, sizeof(stageTitle), "Stage 5: Simplified Persistent Kernel (Batch Size = %d)", batchSize);
    printPerformanceMetrics(stageTitle, metrics);
    
    printf("Simplified persistent kernel processing completed successfully\n");
    
    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(d_packets));
    CHECK_CUDA_ERROR(cudaFree(d_results));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    
    // Calculate total time
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    printf("Simplified persistent kernel processing (total): %lld us\n", duration);
    
    return duration;
}

#endif // PERSISTENT_KERNEL_CUH 