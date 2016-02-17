#################################################
#################################################
## Compilers
CC	= gcc
CX	= g++
CU	= /usr/local/cuda/bin/nvcc

## Flags
FMACH	= -m64
FOPTM	= -O2
FC++11	= -std=c++11
FOMP	= -fopenmp
# CUDA Flags
FARCH	= -arch=compute_52
FCODE	= -code=sm_52
FPTX	= -code=compute_52
CXFLAGS	= $(FMACH) $(FOPTM) $(FC++11)
CUFLAGS	= $(FARCH) $(FCODE) $(FPTX)

## Libraries
LCUDART	= --cudart static
LMATH	= -lm
LNTL	= -lntl
LGMP	= -lgmp
LOMP	= -lgomp
LINKERS	= $(LCUDART) $(LMATH) $(LNTL) $(LGMP)

# Commands
COMPCC	= $(CC) $(CXFLAGS)
COMPCX	= $(CX) $(CXFLAGS)
COMPCU	= $(CU) $(CXFLAGS) $(CUFLAGS)
LINKCC	= $(CC) $(CXFLAGS) $(LINKERS) -w
LINKCX	= $(CX) $(CXFLAGS) $(LINKERS) -w
LINKCU	= $(CU) $(CXFLAGS) $(CUFLAGS) $(LINKERS) -w

#################################################
#################################################
## Source Files
SRCCC	=
SRCCX	=
SRCCU	= Parameters.cu\
		  DeviceManager.cu\
		  Base.cu\
		  CuHE.cu\
		  Operations.cu\
		  Relinearization.cu

SRCS	= $(SRCCC) $(SRCCX) $(SRCCU)

## Dependency Files
DEPCC	= $(SRCCC:.c=.d)
DEPCX	= $(SRCCX:.cpp=.d)
DEPCU	= $(SRCCU:.cu=.d)
DEPS	= $(DEPCC) $(DEPCX) $(DEPCU)

## Object Files
OBJCC	= $(SRCCC:.c=.o)
OBJCX	= $(SRCCX:.cpp=.o)
OBJCU	= $(SRCCU:.cu=.o)
OBJS	= $(OBJCC) $(OBJCX) $(OBJCU)

## Library Files
LIBS	= libcuHE.a

## Executable Targets
TARS	=

#################################################
#################################################
all: $(LIBS) $(TARS)

libcuHE.a: $(OBJS)
	ar -cvq libcuHE.a $(OBJS)

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
	rm $(DEPS) $(OBJS) $(TARS) $(LIBS)