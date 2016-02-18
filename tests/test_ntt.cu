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

#include "../cuhe/Base.h"
#include "../cuhe/Debug.h"
#include "../cuhe/DeviceManager.h"
#include <time.h>
#include <NTL/ZZ.h>
using namespace cuHE;
NTL_CLIENT

#define cnt (1*1024)
#define DEV 0
#define CORRECTNESS

#ifdef CORRECTNESS
void check_ntt_ext(int len, uint64 *gpu, uint32 *in) {
  const ZZ P = to_ZZ(0xffffffff00000001);
  const ZZ g = to_ZZ((uint64)15893793146607301539);
  int e0 = 65536/len;
  ZZ w0 = PowerMod(g, e0, P);
  ZZ *r = new ZZ[len];
  for (int i=0; i<len; i++)
    r[i] = PowerMod(w0, i, P);
  ZZ *chk = new ZZ[len];
  for (int idx=5; idx<6; idx++) {
    uint32 *src = in+idx*len;
    uint64 *dst = gpu+idx*len;
    for (int i=0; i<len; i++) {
      chk[i] = 0;
      for (int j=0; j<len/2; j++) {
        chk[i] += to_ZZ(src[j])*r[(i*j)%len];
        chk[i] %= P;
      }
      if (chk[i] != to_ZZ(dst[i])) {
        cout<<"wrong"<<endl;
        break;
      }
    }
  }
  delete [] r;
  delete [] chk;
}
#endif

float time_ntt(int num, int len, uint64 *dst, uint64 *tmp, uint32 *src) {
	CSC(cudaSetDevice(DEV));
	float ret;
	cudaEvent_t start, stop;
	CSC(cudaEventCreate(&start));
	CSC(cudaEventCreate(&stop));
	dim3 grid(len/512, num);
	CSC(cudaEventRecord(start, 0));
	for (int i=0; i<cnt/num; i++) {
	  if (len == 16384) {
		  ntt_1_16k_ext<<<grid, 64>>>(tmp+num*len*i, src+num*len*i);
		  ntt_2_16k<<<grid, 64>>>(tmp+num*len*i);
		  ntt_3_16k<<<grid, 64>>>(dst+num*len*i, tmp+num*len*i);
	  }
	  else if (len == 32768) {
		  ntt_1_32k_ext<<<grid, 64>>>(tmp+num*len*i, src+num*len*i);
		  ntt_2_32k<<<grid, 64>>>(tmp+num*len*i);
		  ntt_3_32k<<<grid, 64>>>(dst+num*len*i, tmp+num*len*i);
	  }
	  else if (len == 65536) {
		  ntt_1_64k_ext<<<grid, 64>>>(tmp+num*len*i, src+num*len*i);
		  ntt_2_64k<<<grid, 64>>>(tmp+num*len*i);
		  ntt_3_64k<<<grid, 64>>>(dst+num*len*i, tmp+num*len*i);
	  }
	}
    CCE();
	CSC(cudaEventRecord(stop, 0));
	CSC(cudaEventSynchronize(stop));
	CSC(cudaEventElapsedTime(&ret, start, stop));
	ret /= cnt;
	CSC(cudaEventDestroy(start));
	CSC(cudaEventDestroy(stop));
	return ret;
}

// y = ntt(x), s is temp result
void test_ntt(float *perf, int numCases, int len) {
  preload_ntt(len);
  CSC(cudaSetDevice(DEV));
  uint32 *hx, *dx;
  uint64 *hs, *ds;
  uint64 *hy, *dy;
  CSC(cudaMalloc(&dx, cnt*len*sizeof(uint32)));
  CSC(cudaMalloc(&ds, cnt*len*sizeof(uint64)));
  CSC(cudaMalloc(&dy, cnt*len*sizeof(uint64)));
  CSC(cudaMallocHost(&hx, cnt*len*sizeof(uint32)));
  CSC(cudaMallocHost(&hs, cnt*len*sizeof(uint64)));
  CSC(cudaMallocHost(&hy, cnt*len*sizeof(uint64)));
  for (int i=0; i<cnt*len; i++) {
    //hs[i] = ((uint64)rand()<<32)+rand();
	  hx[i] = rand();
    hs[i] = 0;
    hy[i] = 0;
  }
  //CSC(cudaMemcpy(ds, hs, cnt*len*sizeof(uint64), cudaMemcpyHostToDevice));
  CSC(cudaMemcpy(dx, hx, cnt*len*sizeof(uint32), cudaMemcpyHostToDevice));
  for (int i=0; i<numCases; i++)
	  perf[i] = time_ntt((0x1<<i), len, dy, ds, dx);
  CSC(cudaMemcpy(hy, dy, cnt*len*sizeof(uint64), cudaMemcpyDeviceToHost));
#ifdef CORRECTNESS
  check_ntt_ext(len, hy+5*len, hx+5*len);
#endif
  CSC(cudaDeviceSynchronize());
  CSC(cudaFree(dx));
  CSC(cudaFree(ds));
  CSC(cudaFree(dy));
  CSC(cudaFreeHost(hx));
  CSC(cudaFreeHost(hs));
  CSC(cudaFreeHost(hy));
  cleanup_ntt();
  return;
}

#define nCases 10
int main() {
  srand(time(NULL));
  cuHE::setNumDevices(1);
  float perf[3][nCases];
  test_ntt(perf[0], nCases, 16384);
  test_ntt(perf[1], nCases, 32768);
  test_ntt(perf[2], nCases, 65536);

  cout<<"Num\t16k-NTT (ms)\t32k-NTT (ms)\t64k-NTT (ms)"<<endl;
  for (int i=0; i<nCases; i++)
    cout<<(0x1<<i)<<"\t"<<perf[0][i]<<"\t"<<perf[1][i]<<"\t"<<perf[2][i]<<endl;

  return 0;
}
