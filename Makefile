NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_61
NVCC_DP_FLAGS = -O3 -arch=sm_61 -rdc=true

all: basic01 basic02 basic03 basic04

basic01: basic01.cu
	$(NVCC) $(NVCC_FLAGS) -o basic01 basic01.cu

basic02: basic02.cu
	$(NVCC) $(NVCC_FLAGS) -o basic02 basic02.cu

basic03: basic03.cu
	$(NVCC) $(NVCC_DP_FLAGS) -o basic03 basic03.cu

basic04: basic04.cu
	$(NVCC) $(NVCC_FLAGS) -o basic04 basic04.cu

clean:
	rm -f basic01 basic02 basic03 basic04 