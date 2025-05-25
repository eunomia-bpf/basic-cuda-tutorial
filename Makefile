NVCC = nvcc

# Auto-detect GPU architecture
ARCH := $(shell $(NVCC) -run ./detect_arch.cu 2>/dev/null || echo "sm_61")

NVCC_FLAGS = -O3 -arch=$(ARCH)
NVCC_DP_FLAGS = -O3 -arch=$(ARCH) -rdc=true
NVCC_PROF_FLAGS = -O3 -arch=$(ARCH) -lcupti -lnvToolsExt
NVCC_EXT_FLAGS = -O3 -arch=$(ARCH) -ldl -lpthread

.PHONY: all clean detect_arch

all: detect_arch basic01 basic02 basic03 basic04 basic05 basic06 basic07 basic08 basic09

detect_arch:
	@echo "Detected GPU architecture: $(ARCH)"
	@if [ "$(ARCH)" = "sm_61" ]; then \
		echo "Using default sm_61 architecture. To use actual GPU architecture:"; \
		echo "1. Create detect_arch.cu with the code to detect architecture"; \
		echo "2. Or manually set architecture in Makefile"; \
	fi

basic01: basic01.cu
	$(NVCC) $(NVCC_FLAGS) -o basic01 basic01.cu

basic02: basic02.cu
	$(NVCC) $(NVCC_FLAGS) -o basic02 basic02.cu

basic03: basic03.cu
	$(NVCC) $(NVCC_DP_FLAGS) -o basic03 basic03.cu

basic04: basic04.cu
	$(NVCC) $(NVCC_FLAGS) -o basic04 basic04.cu

basic05: basic05.cu
	$(NVCC) $(NVCC_FLAGS) -o basic05 basic05.cu

basic06: basic06.cu
	$(NVCC) $(NVCC_FLAGS) -o basic06 basic06.cu

basic07: basic07.cu
	$(NVCC) $(NVCC_FLAGS) -o basic07 basic07.cu

basic08: basic08.cu
	$(NVCC) $(NVCC_PROF_FLAGS) -o basic08 basic08.cu

basic09: basic09.cu
	$(NVCC) $(NVCC_EXT_FLAGS) -o basic09 basic09.cu

clean:
	rm -f basic01 basic02 basic03 basic04 basic05 basic06 basic07 basic08 basic09 