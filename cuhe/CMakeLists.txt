CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(cuhe)

SET(GPU_ARCH "50" CACHE STRING "Cuda architecture number, one of ('20', '30', '32', '35', '37', '50', '52', '53')")
SET(GCC_CUDA_VERISON "4.9" CACHE STRING "GCC compiler to be used with cuda")

SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

FIND_PACKAGE(NTL REQUIRED)
FIND_PACKAGE(OpenMP REQUIRED)
FIND_PACKAGE(CUDA REQUIRED)

INCLUDE(FindCUDA)

FILE(GLOB SOURCES "*.cu" "*.h")

LIST(APPEND CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS} -m64 -std=c++11 -O3 -ffast-math -Wall")

LIST(APPEND CUDA_NVCC_FLAGS "-ccbin gcc-${GCC_CUDA_VERISON} -std=c++11 -gencode arch=compute_${GPU_ARCH},code=sm_${GPU_ARCH}")

SET(LIB_NAME "cuHE")

CUDA_BUILD_CLEAN_TARGET()

CUDA_ADD_LIBRARY( cuHE
       Parameters.cu 
	   DeviceManager.cu
	   Base.cu
	   CuHE.cu
	   Operations.cu
	   Relinearization.cu
	   STATIC
)

TARGET_LINK_LIBRARIES(cuHE ${NTL_LIB} ${M_LIB} ${GMP_LIB} ${GOMP_LIB})

CUDA_ADD_LIBRARY( cuHEShared
       Parameters.cu 
	   DeviceManager.cu
	   Base.cu
	   CuHE.cu
	   Operations.cu
	   Relinearization.cu
	   SHARED
)

TARGET_LINK_LIBRARIES(cuHEShared ${NTL_LIB} ${M_LIB} ${GMP_LIB} ${GOMP_LIB})
