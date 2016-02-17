CC := /usr/local/cuda/bin/nvcc
CCFLAGS := -O3 -Xcompiler -fopenmp -std=c++11
LIBS := -lm -lntl -lgmp

CCDEP := $(CC) $(CCFLAGS) -gencode arch=compute_52,code=sm_52 -odir "build" -M -o 
CCOBJ := $(CC) $(CCFLAGS) --compile --relocatable-device-code=false -gencode arch=compute_52,code=compute_52 -gencode arch=compute_52,code=sm_52 -x cu -o
CCEXE := $(CC) --cudart static -Xlinker $(CCFLAGS)
all: test
test: test.o DHS.o
	$(CCEXE) -o test test.o DHS.o ../../cuhe/libcuHE.a $(LIBS)
test.o: test.cu
	$(CCDEP) "$(@:%.o=%.d)" "$<"
	$(CCOBJ) "$@" "$<"
DHS.o: DHS.cu
	$(CCDEP) "$(@:%.o=%.d)" "$<"
	$(CCOBJ) "$@" "$<"
clean:
	rm *.o *.d test