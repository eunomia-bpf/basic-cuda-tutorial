#ifndef PACKET_PROCESSING_KERNELS_CUH
#define PACKET_PROCESSING_KERNELS_CUH

#include "packet_processing_common.h"

// Kernel for basic packet processing
__global__ void processPacketsBasic(Packet* packets, PacketResult* results, int numPackets) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < numPackets) {
        Packet* packet = &packets[tid];
        PacketResult* result = &results[tid];
        
        // Record processing start time
        packet->processingStart = clock64();
        
        // Call the core packet processing function
        processPacketGPU(packet, result, tid);
        
        // Record processing end time
        result->processingEnd = clock64();
        
        // Mark packet as processed
        packet->status = COMPLETED;
    }
}

// Kernel for zero-copy processing
__global__ void processPacketsZeroCopy(Packet* packets, PacketResult* results, int numPackets) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < numPackets) {
        Packet* packet = &packets[tid];
        PacketResult* result = &results[tid];
        
        // Record processing start time
        packet->processingStart = clock64();
        
        // Call the core packet processing function
        processPacketGPU(packet, result, tid);
        
        // Record processing end time
        result->processingEnd = clock64();
        
        // Mark packet as processed directly in host memory
        packet->status = COMPLETED;
    }
}

// Persistent kernel for continuous packet processing
__global__ void persistentPacketKernel(PacketBatch* batches, PacketResult* results, 
                                       volatile GlobalState* state, int maxBatches) {
    // Shared memory for batch index and status
    __shared__ int s_batchIdx;
    __shared__ int s_batchSize;
    __shared__ long long s_batchStartTime;
    
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
                        s_batchStartTime = clock64();
                        break;
                    }
                }
            }
        }
        
        // Make batch index visible to all threads
        __syncthreads();
        
        // If no batch found, yield briefly and check again
        if (s_batchIdx == -1) {
            // Short yield - reduces resource contention but doesn't affect benchmarking
            __threadfence(); // Memory fence to ensure visibility across threads
            continue;
        }
        
        // Process the assigned batch if this thread has a packet to process
        if (tid < s_batchSize) {
            Packet* packet = &batches[s_batchIdx].packets[tid];
            PacketResult* result = &results[s_batchIdx * DEFAULT_BATCH_SIZE + tid];
            
            // Record processing start time
            packet->processingStart = clock64();
            
            // Call the core packet processing function
            processPacketGPU(packet, result, tid);
            
            // Record processing end time
            result->processingEnd = clock64();
            
            // Mark packet as processed
            packet->status = COMPLETED;
        }
        
        // Ensure all threads complete processing before marking batch as done
        __syncthreads();
        
        // Master thread marks batch as completed
        if (tid == 0) {
            batches[s_batchIdx].status = COMPLETED;
            batches[s_batchIdx].completionTime = clock64();
            atomicAdd((int*)&state->batchesCompleted, 1);
        }
    }
}

#endif // PACKET_PROCESSING_KERNELS_CUH 