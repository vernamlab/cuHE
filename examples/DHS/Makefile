#################################################
#################################################
## Compilers
CC = gcc
CX = g++
CU = /usr/local/cuda/bin/nvcc

## Flags
FMACH	= -m64
FOPTM	= -O2
FC++11 = -std=c++11
FOMP = -fopenmp
# CUDA Flags
FARCH	= -arch=compute_50
FCODE	= -code=sm_50
FPTX = -code=compute_50
CXFLAGS	= $(FMACH) $(FOPTM) $(FC++11)
CUFLAGS	= $(FARCH) $(FCODE) $(FPTX)

## Libraries
LCUDART	= --cudart static
LMATH	= -lm
LNTL = -lntl
LGMP = -lgmp
LOMP = -lgomp
LINKERS = $(LCUDART) $(LMATH) $(LNTL) $(LGMP)

## Library Files
LIBS = ../../cuhe/libcuHE.a

# Commands
COMPCC = $(CC) $(CXFLAGS)
COMPCX = $(CX) $(CXFLAGS)
COMPCU = $(CU) $(CXFLAGS) $(CUFLAGS)
LINKCC = $(CC) $(CXFLAGS) $(LINKERS) -w
LINKCX = $(CX) $(CXFLAGS) $(LINKERS) -w
LINKCU = $(CU) $(CXFLAGS) $(CUFLAGS) $(LINKERS) -w

#################################################
#################################################
## Source Files
SRCCC	=
SRCCX	=
SRCCU	= DHS.cu\
				simple_DHS.cu
SRCS = $(SRCCC) $(SRCCX) $(SRCCU)

## Dependency Files
DEPCC	= $(SRCCC:.c=.d)
DEPCX	= $(SRCCX:.cpp=.d)
DEPCU	= $(SRCCU:.cu=.d)
DEPS = $(DEPCC) $(DEPCX) $(DEPCU)

## Object Files
OBJCC	= $(SRCCC:.c=.o)
OBJCX	= $(SRCCX:.cpp=.o)
OBJCU	= $(SRCCU:.cu=.o)
OBJS = $(OBJCC) $(OBJCX) $(OBJCU)

## Build Targets
TARS = simple_DHS

#################################################
#################################################
all: $(TARS)

simple_DHS: $(OBJS) $(LIBS)
	$(LINKCU) -o $@ $(OBJS) $(LIBS)

$(OBJCC): $(@:%.o=%.c)
	$(COMPCC) -M $(@:%.o=%.c) -o $(@:%.o=%.d)
	$(COMPCC) -c $(@:%.o=%.c) -o $@
$(OBJCX): $(@:%.o=%.cpp)
	$(COMPCX) -M $(@:%.o=%.cpp) -o $(@:%.o=%.d)
	$(COMPCX) -c $(@:%.o=%.cpp) -o $@
$(OBJCU): $(@:%.o=%.cu)
	$(COMPCU) -M $(@:%.o=%.cu) -o $(@:%.o=%.d)
	$(COMPCU) -c $(@:%.o=%.cu) -o $@

# Other Targets
clean:
	rm $(DEPS) $(OBJS) $(TARS)