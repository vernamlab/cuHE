/* 
 *	The MIT License (MIT)
 *	Copyright (c) 2013-2015 Wei Dai
 *
 *	Permission is hereby granted, free of charge, to any person obtaining a copy
 *	of this software and associated documentation files (the "Software"), to deal
 *	in the Software without restriction, including without limitation the rights
 *	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *	copies of the Software, and to permit persons to whom the Software is
 *	furnished to do so, subject to the following conditions:
 *
 *	The above copyright notice and this permission notice shall be included in
 *	all copies or substantial portions of the Software.
 *
 *	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *	THE SOFTWARE.
 */

/*	Based on
	"Thrust": [https://parallel-computing.pro/index.php/9-cuda/
	34-thrust-cuda-tip-reuse-temporary-buffers-across-transforms]
*/
#include "DeviceManager.h"
#include "Debug.h"

namespace cuHE {

///////////////////////////////////////////////////////////////////////////////
//// Device Setup /////////////////////////////////////////////////////////////
static int nGPUs = 1;

void setNumDevices(int val) {
	nGPUs = val;
	for (int i=0; i<nGPUs; i++) {
		CSC(cudaSetDevice(i));
		CSC(cudaDeviceReset());
	}
}

int numDevices() {
	return nGPUs;
}

///////////////////////////////////////////////////////////////////////////////
//// Functions to Call ////////////////////////////////////////////////////////
static bool deviceAllocatorIsOn_ = false;
static DeviceAllocator *deviceAllocator;

void bootDeviceAllocator(size_t size, unsigned long int num) {
	deviceAllocatorIsOn_ = true;
	deviceAllocator = new DeviceAllocator[numDevices()];
	for (int dev=0; dev<numDevices(); dev++) {
		size_t empty, total;
		CSC(cudaSetDevice(dev));
		CSC(cudaMemGetInfo(&empty, &total));
		int cnt;
		if (num == 0)
			cnt = empty/size;
		else
			cnt = empty/size >= num ? num : empty/size;
		char** ptr = new char* [cnt];
		for (int i=0; i<cnt; i++)
			ptr[i] = deviceAllocator[dev].allocate(size);
		for (int i=0; i<cnt; i++)
			deviceAllocator[dev].deallocate(ptr[i]);
		delete [] ptr;
		printf("On Device %d: %f MBytes of memory is pre-allocated.\n", dev, cnt*size*1.0/1024/1024);
	}
}

void haltDeviceAllocator() {
	deviceAllocatorIsOn_ = false;
	delete [] deviceAllocator;
}

bool deviceAllocatorIsOn() {
	return deviceAllocatorIsOn_;
}

static __inline__ int nowDevice() {
	int ret;
	cudaGetDevice(&ret);
	return ret;
} // return the id of current selected device

void* deviceMalloc(size_t size) {
	return (void *)deviceAllocator[nowDevice()].allocate(size);
}

void deviceFree(void* ptr) {
	deviceAllocator[nowDevice()].deallocate((char* )ptr);
}

///////////////////////////////////////////////////////////////////////////////
//// @class DeviceAllocator ///////////////////////////////////////////////////
DeviceAllocator::DeviceAllocator() {
}

DeviceAllocator::~DeviceAllocator() {
	freeAll();
}

char* DeviceAllocator::allocate(std::ptrdiff_t size) {
	char* result = 0;
	FreeBlocks::iterator freeBlock = freeBlocks.find(size);
	if (freeBlock != freeBlocks.end()) {	// found a virtual block
		result = freeBlock->second;
		freeBlocks.erase(freeBlock);
	}
	else {	// not available in map
		try {	// cudaMalloc
			void* temp;
			CSC(cudaMalloc(&temp, size));
			result = (char*)temp;
		}
		catch(std::runtime_error &e) {	// not enough memory
			//throw;
			printf("Out of memory on Device %d.\n", nowDevice());
			return NULL;
		}
	}
	allocatedBlocks.insert(std::make_pair(result, size));
	return result;
}

void DeviceAllocator::deallocate(char* ptr) {
	AllocatedBlocks::iterator iter = allocatedBlocks.find(ptr);
	std::ptrdiff_t size = iter->second;
	allocatedBlocks.erase(iter);
	freeBlocks.insert(std::make_pair(size, ptr));
}

void DeviceAllocator::freeAll() {
#if PRINT_DETAILS == 1
	printf("WARNING: program will free allocated memory on device[%d]!\n", nowDevice());
#endif
	for (FreeBlocks::iterator i = freeBlocks.begin();
		i != freeBlocks.end(); i++) {
		CSC(cudaFree(i->second));
	}
	for (AllocatedBlocks::iterator i = allocatedBlocks.begin();
		i != allocatedBlocks.end(); i++) {
		CSC(cudaFree(i->first));
    }
}
} // end cuHE