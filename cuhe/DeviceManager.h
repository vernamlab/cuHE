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

/*!	/file Device.h
 *	/brief Set and record the number of devices in use.
 *	A customized device memory allocator.
 *	(Manage devices and streams).
 */

#pragma once

#include <stdexcept>
#include <map>

namespace cuHE {

/** A customized memory method for each GPU device.	*/
class DeviceAllocator {
public:
	DeviceAllocator();
	~DeviceAllocator();

	char* allocate(std::ptrdiff_t size);
	// find empty block in map;
	// if not available, cudaMalloc() and add a block to map

	void deallocate(char* ptr);
	// remove ptr from map, set block to empty

	void freeAll();
	// cudaFree() all blocks in map

private:
	typedef std::multimap<std::ptrdiff_t, char*> FreeBlocks;
	typedef std::map<char*, std::ptrdiff_t> AllocatedBlocks;
	FreeBlocks freeBlocks;
	AllocatedBlocks allocatedBlocks;
};

/**	Reset and adopt 'num' GPUs(devices) for computation.
	GPU IDs are 0 ~ 'num'-1. */
void setNumDevices(int num);

/** Return the number of GPUs(devices) in use. */
int numDevices();

/** Return a pointer on current device. */
void* deviceMalloc(size_t size);

/** Free a pointer on current device. */
void deviceFree(void* ptr);

/** Check if DeviceAllocator is enabled. */
bool deviceAllocatorIsOn();

/** Create a deviceAllocator for each GPU;
	slice currect available GPU memory into blocks with a large "size";
	(now has problem when size is too small map is too large). */
void bootDeviceAllocator(size_t size, unsigned long int num = 0);

/** Free all allocated GPU memory and deviceAllocators. */
void haltDeviceAllocator();

} // end cuHE

