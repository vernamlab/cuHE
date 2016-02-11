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

/*!	/file Debug.h
 *	/brief	For error checking purpose.
 *			Provide error types in the future.
 */

#pragma once

/** Report error location and terminate
	if "cudaError != SUCCESS". */
#define CSC(err)	__cudaSafeCall(err, __FILE__, __LINE__)
/** Report error location and terminate
	if "cudaError != SUCCESS" occurs previously. */
#define CCE()		__cudaCheckError(__FILE__, __LINE__)


#include <cuda_runtime_api.h>
#include <cuda.h>
#include <stdio.h>

inline void __cudaSafeCall(cudaError err, const char *file, const int line) {
	if (cudaSuccess != err) {
		fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
		exit(-1);
	}
	return;
}

inline void __cudaCheckError(const char *file, const int line) {
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
		exit(-1);
	}
	// More careful checking. However, this will affect performance.
	// Comment out if needed.
	//#define safer
	#ifdef safer
	err = cudaDeviceSynchronize();
	if (cudaSuccess != err) {
		fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
		exit(-1);
	}
	#endif
	return;
}

/*
// print 10 unsigned long values
#define	chk64_d(d_ptr, dev){\
	cudaDeviceSynchronize();\
	unsigned long int h[10];\
	CSC(cudaSetDevice(dev));\
	CSC(cudaMemcpy(h, d_ptr, 10*sizeof(unsigned long int), cudaMemcpyDeviceToHost));\
	for(int i=0; i<10; i++)\
		cout<<h[i]<<endl;\
}

// print 10 unsigned values
#define	chk32_d(d_ptr, dev){\
	cudaDeviceSynchronize();\
	unsigned int h[10];\
	CSC(cudaSetDevice(dev));\
	CSC(cudaMemcpy(h, d_ptr, 10*sizeof(unsigned int), cudaMemcpyDeviceToHost));\
	for(int i=0; i<10; i++)\
		cout<<h[i]<<" ";\
}
*/