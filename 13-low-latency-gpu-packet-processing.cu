/**
 * 13-low-latency-gpu-packet-processing.cu
 * 
 * This example demonstrates techniques for low-latency network packet processing on GPUs.
 * The code progresses through several optimization stages, from a basic implementation
 * to increasingly optimized versions.
 * 
 * Key optimizations include:
 * 1. Pinned memory for faster transfers
 * 2. Zero-copy memory to avoid explicit transfers
 * 3. Stream concurrency for operation overlap
 * 4. Persistent kernels to eliminate launch overhead
 * 5. CUDA Graphs for reduced CPU overhead
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <chrono>
#include <thread>
#include <vector>
#include <algorithm>
#include <random>
#include <mutex>
#include <condition_variable>

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

// Timing utilities
#define START_TIMER auto start = std::chrono::high_resolution_clock::now();
#define STOP_TIMER(name) do { \
    auto end = std::chrono::high_resolution_clock::now(); \
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count(); \
    printf("%s: %lld us\n", name, duration); \
} while(0)

// Packet processing constants
#define MAX_PACKET_SIZE 1500         // Maximum Ethernet packet size
#define PACKET_HEADER_SIZE 42        // Ethernet + IP + TCP headers
#define BATCH_SIZE 256               // Number of packets per batch
#define MAX_BATCHES 10               // Maximum number of batches in flight
#define NUM_PACKETS 10000            // Total packets to process
#define NUM_STREAMS 4                // Number of CUDA streams to use

// Status codes for packet processing
enum PacketStatus {
    PENDING = 0,
    PROCESSING = 1,
    COMPLETED = 2,
    ERROR = 3
};

// Simple packet structure
struct Packet {
    unsigned char header[PACKET_HEADER_SIZE];  // Ethernet + IP + TCP headers
    unsigned char payload[MAX_PACKET_SIZE - PACKET_HEADER_SIZE];
    int size;
    int status;
};

// Batch of packets
struct PacketBatch {
    Packet packets[BATCH_SIZE];
    int count;
    int status;
    volatile int ready;
};

// Results from packet processing
struct PacketResult {
    int packetId;
    int action;  // 0 = drop, 1 = forward, 2 = modify
    int matches; // Number of pattern matches found
};

// Global state for persistent kernel
struct GlobalState {
    volatile int batchesReady;
    volatile int batchesCompleted;
    volatile int shutdown;
};

// Simple packet generator for testing
void generateTestPackets(Packet* packets, int numPackets) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(64, MAX_PACKET_SIZE);
    
    for (int i = 0; i < numPackets; i++) {
        // Generate random header (first 42 bytes)
        for (int j = 0; j < PACKET_HEADER_SIZE; j++) {
            packets[i].header[j] = rand() % 256;
        }
        
        // Set source and destination IP addresses for easy identification
        // Source IP in header bytes 26-29
        packets[i].header[26] = 10;
        packets[i].header[27] = 0;
        packets[i].header[28] = 0;
        packets[i].header[29] = 1;
        
        // Destination IP in header bytes 30-33
        packets[i].header[30] = 10;
        packets[i].header[31] = 0;
        packets[i].header[32] = 0;
        packets[i].header[33] = 2;
        
        // Random payload
        int payloadSize = dis(gen) - PACKET_HEADER_SIZE;
        for (int j = 0; j < payloadSize; j++) {
            packets[i].payload[j] = rand() % 256;
        }
        
        packets[i].size = PACKET_HEADER_SIZE + payloadSize;
        packets[i].status = PENDING;
    }
}

/******************************************************************************
 * Stage 1: Basic Packet Processing
 * 
 * This is a simple implementation that:
 * - Transfers packets from host to device
 * - Processes them on the GPU
 * - Transfers results back
 * 
 * This serves as our baseline implementation.
 ******************************************************************************/

// Kernel for basic packet processing
__global__ void processPacketsBasic(Packet* packets, PacketResult* results, int numPackets) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < numPackets) {
        Packet* packet = &packets[tid];
        PacketResult* result = &results[tid];
        
        // Extract packet information (just for demonstration)
        result->packetId = tid;
        result->matches = 0;
        
        // Simple pattern matching - count occurrences of byte value 0x42
        for (int i = 0; i < packet->size - PACKET_HEADER_SIZE; i++) {
            if (packet->payload[i] == 0x42) {
                result->matches++;
            }
        }
        
        // Decision logic - just an example
        if (result->matches > 5) {
            result->action = 0;  // Drop
        } else if (result->matches > 0) {
            result->action = 2;  // Modify
        } else {
            result->action = 1;  // Forward
        }
        
        // Mark packet as processed
        packet->status = COMPLETED;
    }
}

void runBasicProcessing() {
    printf("\n=== Stage 1: Basic Packet Processing ===\n");
    
    // Allocate host memory for packets and results
    Packet* h_packets = (Packet*)malloc(NUM_PACKETS * sizeof(Packet));
    PacketResult* h_results = (PacketResult*)malloc(NUM_PACKETS * sizeof(PacketResult));
    
    // Generate test packets
    generateTestPackets(h_packets, NUM_PACKETS);
    
    // Allocate device memory
    Packet* d_packets;
    PacketResult* d_results;
    CHECK_CUDA_ERROR(cudaMalloc(&d_packets, NUM_PACKETS * sizeof(Packet)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_results, NUM_PACKETS * sizeof(PacketResult)));
    
    // Measure baseline performance
    START_TIMER
    
    // Copy packets to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_packets, h_packets, NUM_PACKETS * sizeof(Packet), 
                     cudaMemcpyHostToDevice));
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (NUM_PACKETS + blockSize - 1) / blockSize;
    processPacketsBasic<<<numBlocks, blockSize>>>(d_packets, d_results, NUM_PACKETS);
    
    // Wait for kernel to finish
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Copy results back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_results, d_results, NUM_PACKETS * sizeof(PacketResult), 
                     cudaMemcpyDeviceToHost));
    
    STOP_TIMER("Basic processing (total)");
    
    // Print some results
    int drops = 0, forwards = 0, modifies = 0;
    for (int i = 0; i < NUM_PACKETS; i++) {
        switch (h_results[i].action) {
            case 0: drops++; break;
            case 1: forwards++; break;
            case 2: modifies++; break;
        }
    }
    
    printf("Processed %d packets: %d drops, %d forwards, %d modifies\n", 
           NUM_PACKETS, drops, forwards, modifies);
    
    // Cleanup
    free(h_packets);
    free(h_results);
    CHECK_CUDA_ERROR(cudaFree(d_packets));
    CHECK_CUDA_ERROR(cudaFree(d_results));
}

/******************************************************************************
 * Stage 2: Pinned Memory Optimization
 * 
 * This version uses pinned memory to improve transfer speeds between
 * host and device, which is critical for low-latency processing.
 ******************************************************************************/

void runPinnedMemoryProcessing() {
    printf("\n=== Stage 2: Pinned Memory Optimization ===\n");
    
    // Allocate pinned memory for packets and results
    Packet* h_packets;
    PacketResult* h_results;
    CHECK_CUDA_ERROR(cudaHostAlloc(&h_packets, NUM_PACKETS * sizeof(Packet), 
                     cudaHostAllocDefault));
    CHECK_CUDA_ERROR(cudaHostAlloc(&h_results, NUM_PACKETS * sizeof(PacketResult), 
                     cudaHostAllocDefault));
    
    // Generate test packets
    generateTestPackets(h_packets, NUM_PACKETS);
    
    // Allocate device memory
    Packet* d_packets;
    PacketResult* d_results;
    CHECK_CUDA_ERROR(cudaMalloc(&d_packets, NUM_PACKETS * sizeof(Packet)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_results, NUM_PACKETS * sizeof(PacketResult)));
    
    // Measure performance with pinned memory
    START_TIMER
    
    // Measure transfer time separately
    auto transfer_start = std::chrono::high_resolution_clock::now();
    
    // Copy packets to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_packets, h_packets, NUM_PACKETS * sizeof(Packet), 
                     cudaMemcpyHostToDevice));
    
    auto transfer_end = std::chrono::high_resolution_clock::now();
    auto transfer_time = std::chrono::duration_cast<std::chrono::microseconds>
                        (transfer_end - transfer_start).count();
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (NUM_PACKETS + blockSize - 1) / blockSize;
    processPacketsBasic<<<numBlocks, blockSize>>>(d_packets, d_results, NUM_PACKETS);
    
    // Wait for kernel to finish
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Copy results back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_results, d_results, NUM_PACKETS * sizeof(PacketResult), 
                     cudaMemcpyDeviceToHost));
    
    STOP_TIMER("Pinned memory processing (total)");
    printf("Transfer time: %lld us\n", transfer_time);
    
    // Print some results
    int drops = 0, forwards = 0, modifies = 0;
    for (int i = 0; i < NUM_PACKETS; i++) {
        switch (h_results[i].action) {
            case 0: drops++; break;
            case 1: forwards++; break;
            case 2: modifies++; break;
        }
    }
    
    printf("Processed %d packets: %d drops, %d forwards, %d modifies\n", 
           NUM_PACKETS, drops, forwards, modifies);
    
    // Cleanup
    CHECK_CUDA_ERROR(cudaFreeHost(h_packets));
    CHECK_CUDA_ERROR(cudaFreeHost(h_results));
    CHECK_CUDA_ERROR(cudaFree(d_packets));
    CHECK_CUDA_ERROR(cudaFree(d_results));
}

/******************************************************************************
 * Stage 3: Batched Processing with Streams
 * 
 * This version processes packets in batches and uses multiple CUDA streams
 * to overlap transfers and computation, reducing overall latency.
 ******************************************************************************/

void runBatchedStreamProcessing() {
    printf("\n=== Stage 3: Batched Processing with Streams ===\n");
    
    const int packetsPerBatch = BATCH_SIZE;
    const int numBatches = (NUM_PACKETS + packetsPerBatch - 1) / packetsPerBatch;
    
    // Create CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA_ERROR(cudaStreamCreate(&streams[i]));
    }
    
    // Allocate pinned memory for packets and results
    Packet* h_packets[MAX_BATCHES];
    PacketResult* h_results[MAX_BATCHES];
    
    for (int i = 0; i < MAX_BATCHES; i++) {
        CHECK_CUDA_ERROR(cudaHostAlloc(&h_packets[i], packetsPerBatch * sizeof(Packet), 
                         cudaHostAllocDefault));
        CHECK_CUDA_ERROR(cudaHostAlloc(&h_results[i], packetsPerBatch * sizeof(PacketResult), 
                         cudaHostAllocDefault));
    }
    
    // Generate test packets for all batches
    for (int batch = 0; batch < numBatches; batch++) {
        int batchSize = (batch == numBatches - 1) ? 
                        (NUM_PACKETS - batch * packetsPerBatch) : packetsPerBatch;
        
        generateTestPackets(h_packets[batch % MAX_BATCHES], batchSize);
    }
    
    // Allocate device memory for each batch
    Packet* d_packets[MAX_BATCHES];
    PacketResult* d_results[MAX_BATCHES];
    
    for (int i = 0; i < MAX_BATCHES; i++) {
        CHECK_CUDA_ERROR(cudaMalloc(&d_packets[i], packetsPerBatch * sizeof(Packet)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_results[i], packetsPerBatch * sizeof(PacketResult)));
    }
    
    // Kernel launch parameters
    int blockSize = 256;
    
    // Measure performance with batched stream processing
    START_TIMER
    
    // Process all batches
    for (int batch = 0; batch < numBatches; batch++) {
        int streamIdx = batch % NUM_STREAMS;
        int batchIdx = batch % MAX_BATCHES;
        int batchSize = (batch == numBatches - 1) ? 
                        (NUM_PACKETS - batch * packetsPerBatch) : packetsPerBatch;
        
        // Number of thread blocks needed for this batch
        int numBlocks = (batchSize + blockSize - 1) / blockSize;
        
        // Transfer batch to device
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_packets[batchIdx], h_packets[batchIdx], 
                         batchSize * sizeof(Packet), cudaMemcpyHostToDevice, 
                         streams[streamIdx]));
        
        // Process batch
        processPacketsBasic<<<numBlocks, blockSize, 0, streams[streamIdx]>>>(
            d_packets[batchIdx], d_results[batchIdx], batchSize);
        
        // Transfer results back
        CHECK_CUDA_ERROR(cudaMemcpyAsync(h_results[batchIdx], d_results[batchIdx], 
                         batchSize * sizeof(PacketResult), cudaMemcpyDeviceToHost, 
                         streams[streamIdx]));
        
        // If we've used all available batch slots, synchronize the oldest stream
        // to ensure its resources are available
        if (batch >= MAX_BATCHES - 1) {
            int oldestStreamIdx = (batch - (MAX_BATCHES - 1)) % NUM_STREAMS;
            CHECK_CUDA_ERROR(cudaStreamSynchronize(streams[oldestStreamIdx]));
        }
    }
    
    // Synchronize all streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA_ERROR(cudaStreamSynchronize(streams[i]));
    }
    
    STOP_TIMER("Batched stream processing (total)");
    
    // Calculate average latency per batch
    double avgLatencyPerBatch = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - start).count() / 
                (double)numBatches;
    
    printf("Average latency per batch: %.2f us\n", avgLatencyPerBatch);
    printf("Average latency per packet: %.2f us\n", avgLatencyPerBatch / packetsPerBatch);
    
    // Cleanup
    for (int i = 0; i < MAX_BATCHES; i++) {
        CHECK_CUDA_ERROR(cudaFreeHost(h_packets[i]));
        CHECK_CUDA_ERROR(cudaFreeHost(h_results[i]));
        CHECK_CUDA_ERROR(cudaFree(d_packets[i]));
        CHECK_CUDA_ERROR(cudaFree(d_results[i]));
    }
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA_ERROR(cudaStreamDestroy(streams[i]));
    }
}

/******************************************************************************
 * Stage 4: Zero-Copy Memory
 * 
 * This version uses zero-copy memory to eliminate explicit data transfers,
 * which can reduce latency for small packets.
 ******************************************************************************/

// Kernel for zero-copy processing
__global__ void processPacketsZeroCopy(Packet* packets, PacketResult* results, int numPackets) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < numPackets) {
        Packet* packet = &packets[tid];
        PacketResult* result = &results[tid];
        
        // Extract packet information
        result->packetId = tid;
        result->matches = 0;
        
        // Simple pattern matching - count occurrences of byte value 0x42
        for (int i = 0; i < packet->size - PACKET_HEADER_SIZE; i++) {
            if (packet->payload[i] == 0x42) {
                result->matches++;
            }
        }
        
        // Decision logic
        if (result->matches > 5) {
            result->action = 0;  // Drop
        } else if (result->matches > 0) {
            result->action = 2;  // Modify
        } else {
            result->action = 1;  // Forward
        }
        
        // Mark packet as processed directly in host memory
        packet->status = COMPLETED;
    }
}

void runZeroCopyProcessing() {
    printf("\n=== Stage 4: Zero-Copy Memory ===\n");
    
    // Allocate mapped memory for packets and results
    Packet* h_packets;
    PacketResult* h_results;
    CHECK_CUDA_ERROR(cudaHostAlloc(&h_packets, NUM_PACKETS * sizeof(Packet), 
                     cudaHostAllocMapped));
    CHECK_CUDA_ERROR(cudaHostAlloc(&h_results, NUM_PACKETS * sizeof(PacketResult), 
                     cudaHostAllocMapped));
    
    // Get device pointers to the mapped memory
    Packet* d_packets;
    PacketResult* d_results;
    CHECK_CUDA_ERROR(cudaHostGetDevicePointer(&d_packets, h_packets, 0));
    CHECK_CUDA_ERROR(cudaHostGetDevicePointer(&d_results, h_results, 0));
    
    // Generate test packets
    generateTestPackets(h_packets, NUM_PACKETS);
    
    // Measure performance with zero-copy memory
    START_TIMER
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (NUM_PACKETS + blockSize - 1) / blockSize;
    processPacketsZeroCopy<<<numBlocks, blockSize>>>(d_packets, d_results, NUM_PACKETS);
    
    // Wait for kernel to finish
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    STOP_TIMER("Zero-copy processing (total)");
    
    // Print some results
    int drops = 0, forwards = 0, modifies = 0;
    for (int i = 0; i < NUM_PACKETS; i++) {
        switch (h_results[i].action) {
            case 0: drops++; break;
            case 1: forwards++; break;
            case 2: modifies++; break;
        }
    }
    
    printf("Processed %d packets: %d drops, %d forwards, %d modifies\n", 
           NUM_PACKETS, drops, forwards, modifies);
    
    // Cleanup
    CHECK_CUDA_ERROR(cudaFreeHost(h_packets));
    CHECK_CUDA_ERROR(cudaFreeHost(h_results));
}

/******************************************************************************
 * Stage 5: Persistent Kernel
 * 
 * This version uses a persistent kernel that stays resident on the GPU
 * and processes batches as they become available, eliminating kernel
 * launch overhead for minimal latency.
 ******************************************************************************/

// Persistent kernel for continuous packet processing
__global__ void persistentPacketKernel(PacketBatch* batches, PacketResult* results, 
                                       volatile GlobalState* state, int maxBatches) {
    // Shared memory for batch index and status
    __shared__ int s_batchIdx;
    __shared__ int s_batchSize;
    
    // Each thread processes one packet in the batch
    int tid = threadIdx.x;
    
    // Keep kernel running indefinitely
    while (!state->shutdown) {
        // Master thread checks for new work
        if (tid == 0) {
            s_batchIdx = -1;
            s_batchSize = 0;
            
            // Check if there's a batch ready to process
            for (int i = 0; i < maxBatches; i++) {
                if (batches[i].ready == 1) {
                    // Claim this batch
                    int oldValue = atomicExch((int*)&batches[i].ready, 0);
                    
                    if (oldValue == 1) {
                        s_batchIdx = i;
                        s_batchSize = batches[i].count;
                        batches[i].status = PROCESSING;
                        break;
                    }
                }
            }
        }
        
        // Make batch index visible to all threads
        __syncthreads();
        
        // If no batch found, sleep briefly and check again
        if (s_batchIdx == -1) {
            // Simple sleep - in a real implementation, consider more sophisticated approach
            for (int i = 0; i < 1000; i++) {
                // This acts as a sleep
            }
            continue;
        }
        
        // Process the assigned batch if this thread has a packet to process
        if (tid < s_batchSize) {
            Packet* packet = &batches[s_batchIdx].packets[tid];
            PacketResult* result = &results[s_batchIdx * BATCH_SIZE + tid];
            
            // Simple pattern matching
            result->packetId = tid;
            result->matches = 0;
            
            for (int i = 0; i < packet->size - PACKET_HEADER_SIZE; i++) {
                if (packet->payload[i] == 0x42) {
                    result->matches++;
                }
            }
            
            // Decision logic
            if (result->matches > 5) {
                result->action = 0;  // Drop
            } else if (result->matches > 0) {
                result->action = 2;  // Modify
            } else {
                result->action = 1;  // Forward
            }
            
            // Mark packet as processed
            packet->status = COMPLETED;
        }
        
        // Ensure all threads complete processing before marking batch as done
        __syncthreads();
        
        // Master thread marks batch as completed
        if (tid == 0) {
            batches[s_batchIdx].status = COMPLETED;
            atomicAdd((int*)&state->batchesCompleted, 1);
        }
    }
}

// Producer thread that generates packets and enqueues batches
void packetProducerThread(PacketBatch* h_batches, GlobalState* h_state, int numBatches) {
    for (int batch = 0; batch < numBatches; batch++) {
        // Calculate which batch slot to use
        int batchIdx = batch % MAX_BATCHES;
        
        // Calculate batch size
        int batchSize = (batch == numBatches - 1) ? 
                        (NUM_PACKETS - batch * BATCH_SIZE) : BATCH_SIZE;
        
        // Wait until this batch slot is free (status == COMPLETED)
        while (h_batches[batchIdx].status != COMPLETED && 
               h_batches[batchIdx].status != 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
        
        // Generate new packets for this batch
        generateTestPackets(h_batches[batchIdx].packets, batchSize);
        h_batches[batchIdx].count = batchSize;
        h_batches[batchIdx].status = PENDING;
        
        // Signal that batch is ready for processing - use regular increment for host code
        h_batches[batchIdx].ready = 1;
        // Increment counter using regular addition instead of atomicAdd which is device-only
        h_state->batchesReady++;
        
        // Simulate packet arrival rate - adjust for desired packet rate
        std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
}

void runPersistentKernelProcessing() {
    printf("\n=== Stage 5: Persistent Kernel ===\n");
    
    const int numBatches = (NUM_PACKETS + BATCH_SIZE - 1) / BATCH_SIZE;
    
    // Allocate and initialize host memory for batches
    PacketBatch* h_batches;
    PacketResult* h_results;
    GlobalState* h_state;
    
    CHECK_CUDA_ERROR(cudaHostAlloc(&h_batches, MAX_BATCHES * sizeof(PacketBatch), 
                     cudaHostAllocMapped));
    CHECK_CUDA_ERROR(cudaHostAlloc(&h_results, NUM_PACKETS * sizeof(PacketResult), 
                     cudaHostAllocDefault));
    CHECK_CUDA_ERROR(cudaHostAlloc(&h_state, sizeof(GlobalState), 
                     cudaHostAllocMapped));
    
    // Initialize batches and state
    memset(h_batches, 0, MAX_BATCHES * sizeof(PacketBatch));
    memset(h_results, 0, NUM_PACKETS * sizeof(PacketResult));
    memset(h_state, 0, sizeof(GlobalState));
    
    // Get device pointers to the mapped memory
    PacketBatch* d_batches;
    GlobalState* d_state;
    PacketResult* d_results;
    
    CHECK_CUDA_ERROR(cudaHostGetDevicePointer(&d_batches, h_batches, 0));
    CHECK_CUDA_ERROR(cudaHostGetDevicePointer(&d_state, h_state, 0));
    CHECK_CUDA_ERROR(cudaMalloc(&d_results, NUM_PACKETS * sizeof(PacketResult)));
    
    // Launch persistent kernel with one block of BATCH_SIZE threads
    persistentPacketKernel<<<1, BATCH_SIZE>>>(d_batches, d_results, d_state, MAX_BATCHES);
    
    // Measure performance with persistent kernel
    START_TIMER
    
    // Start producer thread to generate packets
    std::thread producer(packetProducerThread, h_batches, h_state, numBatches);
    
    // Monitor progress
    int lastBatchesCompleted = 0;
    while (h_state->batchesCompleted < numBatches) {
        // Check for progress
        if (h_state->batchesCompleted > lastBatchesCompleted) {
            printf("Completed %d/%d batches\n", 
                   h_state->batchesCompleted, numBatches);
            lastBatchesCompleted = h_state->batchesCompleted;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // Signal shutdown
    h_state->shutdown = 1;
    
    // Wait for producer thread to finish
    producer.join();
    
    // Copy results back
    CHECK_CUDA_ERROR(cudaMemcpy(h_results, d_results, 
                     NUM_PACKETS * sizeof(PacketResult), 
                     cudaMemcpyDeviceToHost));
    
    // Wait for kernel to exit
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    STOP_TIMER("Persistent kernel processing (total)");
    
    // Cleanup
    CHECK_CUDA_ERROR(cudaFreeHost(h_batches));
    CHECK_CUDA_ERROR(cudaFreeHost(h_state));
    CHECK_CUDA_ERROR(cudaFreeHost(h_results));
    CHECK_CUDA_ERROR(cudaFree(d_results));
}

/******************************************************************************
 * Stage 6: CUDA Graphs
 * 
 * This version uses CUDA Graphs to capture and replay a sequence of operations,
 * reducing CPU overhead and launch latency.
 ******************************************************************************/

void runCudaGraphProcessing() {
    printf("\n=== Stage 6: CUDA Graphs ===\n");
    
    const int packetsPerBatch = BATCH_SIZE;
    const int numBatches = (NUM_PACKETS + packetsPerBatch - 1) / packetsPerBatch;
    
    // Create a CUDA stream for graph capture
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    
    // Allocate pinned memory for packets and results
    Packet* h_packets;
    PacketResult* h_results;
    CHECK_CUDA_ERROR(cudaHostAlloc(&h_packets, BATCH_SIZE * sizeof(Packet), 
                     cudaHostAllocDefault));
    CHECK_CUDA_ERROR(cudaHostAlloc(&h_results, BATCH_SIZE * sizeof(PacketResult), 
                     cudaHostAllocDefault));
    
    // Allocate device memory
    Packet* d_packets;
    PacketResult* d_results;
    CHECK_CUDA_ERROR(cudaMalloc(&d_packets, BATCH_SIZE * sizeof(Packet)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_results, BATCH_SIZE * sizeof(PacketResult)));
    
    // Generate test packets for the first batch
    generateTestPackets(h_packets, BATCH_SIZE);
    
    // Kernel launch parameters
    int blockSize = 256;
    int numBlocks = (BATCH_SIZE + blockSize - 1) / blockSize;
    
    // Create graph
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    
    // Capture a sequence of operations into a graph
    CHECK_CUDA_ERROR(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    
    // Copy batch to device
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_packets, h_packets, 
                     BATCH_SIZE * sizeof(Packet), 
                     cudaMemcpyHostToDevice, stream));
    
    // Process batch
    processPacketsBasic<<<numBlocks, blockSize, 0, stream>>>(
        d_packets, d_results, BATCH_SIZE);
    
    // Copy results back
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_results, d_results, 
                     BATCH_SIZE * sizeof(PacketResult), 
                     cudaMemcpyDeviceToHost, stream));
    
    // End capture
    CHECK_CUDA_ERROR(cudaStreamEndCapture(stream, &graph));
    
    // Create executable graph
    CHECK_CUDA_ERROR(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
    
    // Measure performance with CUDA Graphs
    START_TIMER
    
    double totalGraphLaunchTime = 0;
    
    // Process all batches using the graph
    for (int batch = 0; batch < numBatches; batch++) {
        // Generate test packets for this batch
        int batchSize = (batch == numBatches - 1) ? 
                        (NUM_PACKETS - batch * BATCH_SIZE) : BATCH_SIZE;
        generateTestPackets(h_packets, batchSize);
        
        // Launch the graph
        auto launch_start = std::chrono::high_resolution_clock::now();
        CHECK_CUDA_ERROR(cudaGraphLaunch(graphExec, stream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        auto launch_end = std::chrono::high_resolution_clock::now();
        
        // Record launch time
        totalGraphLaunchTime += std::chrono::duration_cast<std::chrono::microseconds>
                              (launch_end - launch_start).count();
    }
    
    STOP_TIMER("CUDA Graphs processing (total)");
    
    // Calculate average latency per batch
    double avgGraphLaunchTime = totalGraphLaunchTime / numBatches;
    printf("Average graph launch time per batch: %.2f us\n", avgGraphLaunchTime);
    
    // Cleanup
    CHECK_CUDA_ERROR(cudaGraphDestroy(graph));
    CHECK_CUDA_ERROR(cudaGraphExecDestroy(graphExec));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    CHECK_CUDA_ERROR(cudaFreeHost(h_packets));
    CHECK_CUDA_ERROR(cudaFreeHost(h_results));
    CHECK_CUDA_ERROR(cudaFree(d_packets));
    CHECK_CUDA_ERROR(cudaFree(d_results));
}

/******************************************************************************
 * Main Function
 ******************************************************************************/

int main(int argc, char **argv) {
    // Get device information
    cudaDeviceProp prop;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, 0));
    
    printf("Device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Clock rate: %.2f GHz\n", prop.clockRate / 1e6);
    printf("Number of SMs: %d\n", prop.multiProcessorCount);
    
    // Run each stage of optimization
    runBasicProcessing();
    runPinnedMemoryProcessing();
    runBatchedStreamProcessing();
    runZeroCopyProcessing();
    runPersistentKernelProcessing();
    runCudaGraphProcessing();
    
    // Summarize results
    printf("\n=== Summary ===\n");
    printf("We've demonstrated several GPU optimization techniques for low-latency packet processing:\n");
    printf("1. Basic processing: Our baseline implementation\n");
    printf("2. Pinned memory: Faster host-device transfers\n");
    printf("3. Batched streams: Overlapping transfers and computation\n");
    printf("4. Zero-copy memory: Eliminating explicit transfers\n");
    printf("5. Persistent kernels: Eliminating kernel launch overhead\n");
    printf("6. CUDA Graphs: Reducing CPU overhead for launch sequences\n");
    
    printf("\nFor real-world low-latency packet processing, consider combining these techniques:\n");
    printf("- Use persistent kernels for minimal latency\n");
    printf("- Use pinned memory for data that must be transferred\n");
    printf("- Use zero-copy for small, latency-sensitive data\n");
    printf("- Use adaptive batching based on traffic patterns\n");
    printf("- Use CUDA Graphs for complex, repeatable processing pipelines\n");
    
    return 0;
} 