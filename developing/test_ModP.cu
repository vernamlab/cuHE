/*
The MIT License (MIT)

Copyright (c) 2015 Wei Dai

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

// Test all inline device mod P = 2^64-2^32+1 operations.

#include "ModP.h"
#include "Debug.h"
#include <time.h>
#include <stdio.h>
#include <NTL/ZZ.h>
NTL_CLIENT

#define num (1024*1024)
const ZZ P = to_ZZ(0xffffffff00000001);
ZZ temp;

void rand_array(uint64 *ptr) {
	for (int i=0; i<num; i++) {
		ptr[i] = rand();
		ptr[i] <<= 32;
		ptr[i] |= rand();
	}
}
void rand_offset(int *l) {
	for (int i=0; i<num; i++) {
		l[i] = (rand()%8)*(rand()%8)*3;
	}
}
void rand_exp(int *e) {
	for (int i=0; i<num; i++) {
		e[i] = (unsigned)rand()>>1;
		if (e[i] < 0) {
			printf("Error: random exponent has opposite value.\n");
			exit(-1);
		}
	}
}
__global__ void _kernel_ls_modP(uint64 *dst, uint64 *src, int *offset) {
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx < num) {
		uint64 t = cuHE::_ls_modP(src[idx], offset[idx]);
		dst[idx] = t;
	}
}
bool _test_ls_modP(uint64 *z, uint64 *x, int *l) {
	_kernel_ls_modP<<<(num+1023)/1024, 1024>>>(z, x, l);
	CCE();
	CSC(cudaDeviceSynchronize());
	for (int i=0; i<num; i++) {
		conv(temp, x[i]);
		temp <<= l[i];
		temp %= P;
		if (temp != to_ZZ(z[i])) {
			return false;
		}
	}
	return true;
}
void print(bool result) {
	if (result)
		printf("pass\n");
	else
		printf("fail\n");
}
int main() {
	uint64 *x;
	uint64 *y;
	uint64 *z;
	int *l;
	int *e;
	CSC(cudaMallocManaged(&x, num*sizeof(uint64)));
	CSC(cudaMallocManaged(&y, num*sizeof(uint64)));
	CSC(cudaMallocManaged(&z, num*sizeof(uint64)));
	CSC(cudaMallocManaged(&l, num*sizeof(int)));
	CSC(cudaMallocManaged(&e, num*sizeof(int)));
	srand(time(NULL));
	rand_array(x);
	rand_offset(l);
	printf("_ls_modP:\t");
	print(_test_ls_modP(z, x, l));
	CSC(cudaFree(x));
	CSC(cudaFree(y));
	CSC(cudaFree(z));
	CSC(cudaFree(l));
	CSC(cudaFree(e));
	return 0;
}
