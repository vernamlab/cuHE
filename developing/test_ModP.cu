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
#include <time.h>
#include <stdio.h>
#include <NTL/ZZ.h>
NTL_CLIENT

#define		repeat	1000000
const ZZ P = to_ZZ(0xffffffff00000001);

__global__ void _kernel_ls_modP(uint64 *x, int *l) {
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx < repeat) {
		uint64 t = cuHE::_ls_modP(x[idx], l[idx]);
		x[idx] = t;
	}
}
int _test_ls_modP() {
	uint64 *x;
	int *l;
	ZZ *chk = new ZZ[repeat];
	cudaMallocManaged(&x, repeat*sizeof(uint64));
	cudaMallocManaged(&l, repeat*sizeof(int));
	for (int i=0; i<repeat; i++) {
		chk[i] = RandomBits_ZZ(64)%P;
		conv(x[i], chk[i]);
		l[i] = (rand()%8)*(rand()%8)*3;
		chk[i] <<= l[i];
		chk[i] %= P;
	}
	_kernel_ls_modP<<<(repeat+1023)/1024, 1024>>>(x, l);
	cudaDeviceSynchronize();
	for (int i=0; i<repeat; i++) {
		if (chk[i] != to_ZZ(x[i])) {
			delete [] chk;
			return -1;
		}
	}
	delete [] chk;
	cudaFree(x);
	cudaFree(l);
	return 0;
}
int _test_add_modP() {
	return 0;
}
int main() {
	int ret = 0;
	ret += _test_ls_modP();
	printf("test result: %d\n", ret);
	return ret;
}
