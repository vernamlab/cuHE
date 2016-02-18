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

#include "newBase.h"
#include "../cuhe/DeviceManager.h"
#include "../cuhe/Debug.h"
#include "../cuhe/ModP.h"
#include <NTL/ZZ.h>
#include <NTL/ZZX.h>
NTL_CLIENT

#define bidx  blockIdx.x
#define bidy  blockIdx.y
#define bidz  blockIdx.z
#define tidx  threadIdx.x
#define tidy  threadIdx.y
#define tidz  threadIdx.z
#define bdimx blockDim.x
#define bdimy blockDim.y
#define bdimz blockDim.z
#define gdimx gridDim.x
#define gdimy gridDim.y
#define gdimz gridDim.z

namespace cuHE {

// ntt twiddle factors in device global memory
uint64 **d_roots_16k = NULL;
uint64 **d_roots_32k = NULL;
uint64 **d_roots_64k = NULL;
// ntt twiddle factors in device texture memory
texture<uint32, 1> tex_roots_16k;
texture<uint32, 1> tex_roots_32k;
texture<uint32, 1> tex_roots_64k;
// pre-load ntt twiddle factors for a specific size
void preload_ntt(int len) {
  if (len != 16384 && len != 32768 && len != 65536) {
    printf("Error: pre-load NTT with wrong length.\n");
    exit(-1);
  }
  // generate twiddle factors on host
  const ZZ P = to_ZZ(0xffffffff00000001);
  const ZZ g = to_ZZ((uint64)15893793146607301539);
  int e0 = 65536/len;
  ZZ w0 = PowerMod(g, e0, P);
  uint64 *h_roots = new uint64[len];
  for (int i=0; i<len; i++)
    conv(h_roots[i], PowerMod(w0, i, P));
  // copy to device memory
  int nGPUs = numDevices();
  if (len == 16384) {
    d_roots_16k = new uint64 *[nGPUs];
    for (int dev=0; dev<nGPUs; dev++) {
      CSC(cudaSetDevice(dev));
      CSC(cudaMalloc(&d_roots_16k[dev], len*sizeof(uint64)));
      CSC(cudaMemcpy(d_roots_16k[dev], h_roots, len*sizeof(uint64),
          cudaMemcpyHostToDevice));
      CSC(cudaBindTexture(NULL, tex_roots_16k, d_roots_16k[dev],
          len*sizeof(uint64)));
    }
  }
  else if (len == 32768) {
    d_roots_32k = new uint64 *[nGPUs];
    for (int dev=0; dev<nGPUs; dev++) {
      CSC(cudaSetDevice(dev));
      CSC(cudaMalloc(&d_roots_32k[dev], len*sizeof(uint64)));
      CSC(cudaMemcpy(d_roots_32k[dev], h_roots, len*sizeof(uint64),
          cudaMemcpyHostToDevice));
      CSC(cudaBindTexture(NULL, tex_roots_32k, d_roots_32k[dev],
          len*sizeof(uint64)));
    }
  }
  else if (len == 65536) {
    d_roots_64k = new uint64 *[nGPUs];
    for (int dev=0; dev<nGPUs; dev++) {
      CSC(cudaSetDevice(dev));
      CSC(cudaMalloc(&d_roots_64k[dev], len*sizeof(uint64)));
      CSC(cudaMemcpy(d_roots_64k[dev], h_roots, len*sizeof(uint64),
          cudaMemcpyHostToDevice));
      CSC(cudaBindTexture(NULL, tex_roots_64k, d_roots_64k[dev],
          len*sizeof(uint64)));
    }
  }
  delete [] h_roots;
  return;
}
// free and delete allocated memory space
void cleanup_ntt() {
  int nGPUs = numDevices();
  if (d_roots_16k != NULL) {
    for (int dev=0; dev<nGPUs; dev++) {
      CSC(cudaSetDevice(dev));
      CSC(cudaFree(d_roots_16k[dev]));
    }
    delete [] d_roots_16k;
    d_roots_16k = NULL;
  }
  if (d_roots_32k != NULL) {
    for (int dev=0; dev<nGPUs; dev++) {
      CSC(cudaSetDevice(dev));
      CSC(cudaFree(d_roots_32k[dev]));
    }
    delete [] d_roots_32k;
    d_roots_32k = NULL;
  }
  if (d_roots_64k != NULL) {
    for (int dev=0; dev<nGPUs; dev++) {
      CSC(cudaSetDevice(dev));
      CSC(cudaFree(d_roots_64k[dev]));
    }
    delete [] d_roots_64k;
    d_roots_64k = NULL;
  }
  return;
}
// 4-point NTT
__inline__ __device__
void _ntt4(uint64 *x) {
  register uint64 s[4], temp;
  s[0] = _add_modP(x[0], x[2]);
  s[1] = _sub_modP(x[0], x[2]);
  s[2] = _add_modP(x[1], x[3]);
  s[3] = _sub_modP(x[1], x[3]);
  temp = _ls_modP(s[3], 48);
  x[0] = _add_modP(s[0], s[2]);
  x[1] = _add_modP(s[1], temp);
  x[2] = _sub_modP(s[0], s[2]);
  x[3] = _sub_modP(s[1], temp);
}
// 8-point NTT with 0 paddings
__inline__ __device__
void _ntt8_ext(uint64 *x) {
  register uint64 s[8], temp;
  temp = _ls_modP(x[2], 48);
  s[0] = _add_modP(x[0], x[2]);
  s[1] = _add_modP(x[0], temp);
  s[2] = _sub_modP(x[0], x[2]);
  s[3] = _sub_modP(x[0], temp);
  temp = _ls_modP(x[3], 48);
  s[4] = _add_modP(x[1], x[3]);
  s[5] = _add_modP(x[1], temp);
  s[6] = _sub_modP(x[1], x[3]);
  s[7] = _sub_modP(x[1], temp);
  x[0] = _add_modP(s[0], s[4]);
  x[4] = _sub_modP(s[0], s[4]);
  temp = _ls_modP(s[5], 24);
  x[1] = _add_modP(s[1], temp);
  x[5] = _sub_modP(s[1], temp);
  temp = _ls_modP(s[6], 48);
  x[2] = _add_modP(s[2], temp);
  x[6] = _sub_modP(s[2], temp);
  temp = _ls_modP(s[7], 72);
  x[3] = _add_modP(s[3], temp);
  x[7] = _sub_modP(s[3], temp);
}
// 8-point NTT
__inline__ __device__
void _ntt8(uint64 *x) {
  register uint64 s[8], temp;
  s[0] = _add_modP(x[0], x[4]);
  s[1] = _sub_modP(x[0], x[4]);
  s[2] = _add_modP(x[2], x[6]);
  s[3] = _sub_modP(x[2], x[6]);
  s[4] = _add_modP(x[1], x[5]);
  s[5] = _sub_modP(x[1], x[5]);
  s[6] = _add_modP(x[3], x[7]);
  s[7] = _sub_modP(x[3], x[7]);
  x[0] = _add_modP(s[0], s[2]);
  x[2] = _sub_modP(s[0], s[2]);
  temp = _ls_modP(s[3], 48);
  x[1] = _add_modP(s[1], temp);
  x[3] = _sub_modP(s[1], temp);
  x[4] = _add_modP(s[4], s[6]);
  x[6] = _sub_modP(s[4], s[6]);
  temp = _ls_modP(s[7], 48);
  x[5] = _add_modP(s[5], temp);
  x[7] = _sub_modP(s[5], temp);
  s[0] = _add_modP(x[0], x[4]);
  s[4] = _sub_modP(x[0], x[4]);
  temp = _ls_modP(x[5], 24);
  s[1] = _add_modP(x[1], temp);
  s[5] = _sub_modP(x[1], temp);
  temp = _ls_modP(x[6], 48);
  s[2] = _add_modP(x[2], temp);
  s[6] = _sub_modP(x[2], temp);
  temp = _ls_modP(x[7], 72);
  s[3] = _add_modP(x[3], temp);
  s[7] = _sub_modP(x[3], temp);
  x[0] = s[0];
  x[1] = s[1];
  x[2] = s[2];
  x[3] = s[3];
  x[4] = s[4];
  x[5] = s[5];
  x[6] = s[6];
  x[7] = s[7];
}
__global__ void ntt_1_16k_ext(uint64 *dst, uint32 *src) {
  __shared__ uint64 buffer[512];
  __shared__ uint64 roots[128];
  register uint64 samples[8];
  register uint32 fmem, tmem, fbuf, tbuf; // from/to mem/buffer addr mapping
  //coalesced global memory access & minimum shared memory bank conflicts
  fmem = ((tidx&0x38)<<5)|(bidx<<3)|(tidx&0x7);
  tbuf = ((tidx&0x38)<<3)|(tidx&0x7);
  fbuf = tidx;
  tmem = (bidx<<9)|((tidx&0x7)>>2<<8)|(tidx>>3<<2)|(tidx&0x3);
  roots[tidx] = tex1Dfetch(tex_roots_16k, (((bidx*2)*tidx)<<3)+1);
  roots[tidx] <<= 32;
  roots[tidx] += tex1Dfetch(tex_roots_16k, ((bidx*2)*tidx)<<3);
  roots[tidx+64] = tex1Dfetch(tex_roots_16k, (((bidx*2+1)*tidx)<<3)+1);
  roots[tidx+64] <<= 32;
  roots[tidx+64] += tex1Dfetch(tex_roots_16k, ((bidx*2+1)*tidx)<<3);
  //load 4 or 8 samples from global memory, compute 8-sample ntt
#pragma unroll
  for (int i=0; i<4; i++)
    samples[i] = (src+(bidy<<14))[(i<<11)|fmem];
  _ntt8_ext(samples);
  //times twiddle factors of 64 samples, store to buffer
#pragma unroll
  for(int i=0; i<8; i++)
    buffer[(i<<3)|tbuf] = _ls_modP(samples[i], (tidx>>3)*i*3);
  __syncthreads();
  //load 8 samples from shared mem in transposed way, compute 8-sample ntt
#pragma unroll
  for(int i=0; i<8; i++)
    samples[i] = buffer[(i<<6)|fbuf];
  _ntt8(samples);
  register uint64 temp_root;
#pragma unroll
  for(int i=0; i<8; i++) {
	  temp_root = roots[((tidx&0x7)>>2<<6)|(tidx>>3)|(i<<3)];
    (dst+(bidy<<14))[tmem|(i<<5)] = _mul_modP(samples[i], temp_root);
  }
}
__global__ void ntt_2_16k(uint64 *src) {
  __shared__ uint64 buffer[512];
  register uint64 samples[8];
  register uint32 fmem, tmem, fbuf, tbuf, addrx, addry;
  fmem = (tidx>>3<<8)|(bidx<<3)|((tidx&0x7)>>2<<2)|(tidx&0x3);
  tbuf = ((tidx&0x38)<<3)|(tidx&0x7);
  fbuf = tidx;
  tmem = fmem;
  addrx = tmem&0x3;
  addry = tmem>>2;
#pragma unroll
  for (int i=0; i<8; i++)
    samples[i] = src[(i<<11)|fmem];
  _ntt8(samples);
#pragma unroll
  for (int i=0; i<8; i++)
    buffer[(i<<3)|tbuf] = _ls_modP(samples[i], ((tidx>>3)*i)*3);
  __syncthreads();
#pragma unroll
  for (int i=0; i<8; i++)
    samples[i] = buffer[(i<<6)|fbuf];
  _ntt8(samples);
  register uint64 root;
#pragma unroll
  for (int i=0; i<8; i++) {
    root = tex1Dfetch(tex_roots_16k, ((addry|(i<<9))*addrx*2)+1);
    root <<= 32;
    root += tex1Dfetch(tex_roots_16k, (addry|(i<<9))*addrx*2);
    src[(i<<11)|tmem] = _mul_modP(samples[i], root);
  }
}
__global__ void ntt_3_16k(uint64 *dst, uint64 *src) {
  __shared__ uint64 buffer[512];
  register uint64 samples[8];
  register uint32 fmem, tmem, fbuf, tbuf;
  fmem = (bidx<<9)|tidx;
  tbuf = tidx;
  fbuf = tidx<<3;
  tmem = (bidx<<7)|(tidx<<1);
#pragma unroll
  for (int i=0; i<8; i++)
    buffer[(i<<6)|tbuf] = src[(i<<6)|fmem];
  __syncthreads();
#pragma unroll
  for (int i=0; i<8; i++)
    samples[i] = buffer[fbuf|i];
  _ntt4(samples);
  _ntt4(samples+4);
#pragma unroll
  for (int i=0; i<8; i++)
    dst[((i&0x3)<<12)|((i&0x7)>>2)|tmem] = samples[i];
}
__global__ void ntt_1_32k_ext(uint64 *dst, uint32 *src) {
	__shared__ uint64 buffer[512], roots[64];
	register uint64 samples[8];
	register uint32 fmem, tmem, fbuf, tbuf;
	fmem = ((tidx&0x38)<<6)|(bidx<<3)|(tidx&0x7);
	tbuf = ((tidx&0x38)<<3)|(tidx&0x7);
	fbuf = tidx;
	tmem = (bidx<<9)|fbuf;
	roots[tidx] = tex1Dfetch(tex_roots_32k, ((bidx*tidx)<<4)+1);
	roots[tidx] <<= 32;
	roots[tidx] += tex1Dfetch(tex_roots_32k, (bidx*tidx)<<4);
#pragma unroll
	for (int i=0; i<4; i++)
    samples[i] = src[(i<<12)|fmem];
	_ntt8_ext(samples);
#pragma unroll
	for (int i=0; i<8; i++)
    buffer[(i<<3)|tbuf] = _ls_modP(samples[i], (tidx>>3)*i*3);
	__syncthreads();
#pragma unroll
	for (int i=0; i<8; i++)
    samples[i] = buffer[fbuf|(i<<6)];
	_ntt8(samples);
#pragma unroll
	for (int i=0; i<8; i++)
    dst[tmem|(i<<6)] = _mul_modP(samples[i], roots[(tidx>>3)|(i<<3)]);
}
__global__ void ntt_2_32k(uint64 *src) {
	__shared__ uint64 buffer[512];
	register uint64 samples[8];
	register uint32 fmem, tmem, fbuf, tbuf, addrx, addry;
	fmem = ((tidx&0x38)<<6)|(bidx<<3)|(tidx&0x7);
	tbuf = ((tidx&0x38)<<3)|(tidx&0x7);
	fbuf = tidx;
	tmem = ((tidx&0x38)<<6)|(bidx<<3)|(tidx&0x7);
	addrx = (tidx&0x7);
	addry = ((tidx&0x38)<<3)|bidx;
#pragma unroll
	for (int i=0; i<8; i++)
    samples[i] = src[(i<<12)|fmem];
	_ntt8(samples);
#pragma unroll
	for (int i=0; i<8; i++)
    buffer[(i<<3)|tbuf] = _ls_modP(samples[i], ((tidx>>3)*i)*3);
	__syncthreads();
#pragma unroll
	for (int i=0; i<8; i++)
    samples[i] = buffer[(i<<6)|fbuf];
	_ntt8(samples);
	register uint64 root;
#pragma unroll
	for (int i=0; i<8; i++){
		root = tex1Dfetch(tex_roots_32k, ((addry|(i<<9))*addrx*2)+1);
		root <<= 32;
		root += tex1Dfetch(tex_roots_32k, (addry|(i<<9))*addrx*2);
		src[(i<<12) | tmem] = _mul_modP(samples[i], root);
	}
}
__global__ void ntt_3_32k(uint64 *dst, uint64 *src) {
	__shared__ uint64 buffer[512];
	register uint64 samples[8];
	register uint32 fmem, tmem, fbuf, tbuf;
	fmem = (bidx<<9)|tidx;
	tbuf = tidx;
	fbuf = tidx<<3;
	tmem = (bidx<<6)|tidx;
#pragma unroll
	for (int i=0; i<8; i++)
    buffer[(i<<6)|tbuf] = src[(i<<6)|fmem];
	__syncthreads();
#pragma unroll
	for (int i=0; i<8; i++)
    samples[i] = buffer[fbuf|i];
	_ntt8(samples);
#pragma unroll
	for (int i=0; i<8; i++)
    dst[(i<<12)|tmem] = samples[i];
}
__global__ void ntt_1_64k_ext(uint64 *dst, uint32 *src) {
	__shared__ uint64 buffer[512], roots[64];
	register uint64 samples[8];
	register uint32 fmem, tmem, fbuf, tbuf;
	fmem = ((tidx&0x38)<<7)|(bidx<<3)|(tidx&0x7);
	tbuf = ((tidx&0x38)<<3)|(tidx&0x7);
	fbuf = tidx;
	tmem = ((bidx&0x7E)<<9)|((tidx&0x38)<<1)|((bidx&0x1)<<3)|(tidx&0x7);
	roots[tidx] = tex1Dfetch(tex_roots_64k, (((bidx>>1)*tidx)<<5)+1);
	roots[tidx] <<= 32;
	roots[tidx] += tex1Dfetch(tex_roots_64k, ((bidx>>1)*tidx)<<5);
#pragma unroll
	for (int i=0; i<4; i++)
    samples[i] = src[(i<<13)|fmem];
	_ntt8_ext(samples);
#pragma unroll
	for (int i=0; i<8; i++)
    buffer[(i<<3) | tbuf] = _ls_modP(samples[i], ((tidx>>3)*i)*3);
	__syncthreads();
#pragma unroll
	for (int i=0; i<8; i++)
    samples[i] = buffer[fbuf | (i<<6)];
	_ntt8(samples);
#pragma unroll
	for (int i=0; i<8; i++)
    dst[tmem | (i<<7)] = _mul_modP(samples[i], roots[(tidx>>3)|(i<<3)]);
}
__global__ void ntt_2_64k(uint64 *src){
	__shared__ uint64 buffer[512];
	register uint64 samples[8];
	register uint32 fmem, tmem, fbuf, tbuf, addrx, addry;
	fmem = ((tidx&0x38)<<7)|(bidx<<3)|(tidx&0x7);
	tbuf = (tidx&0x38)<<3|(tidx&0x7);
	fbuf = tidx;
	tmem = fmem;
	addrx = (tidx&0x7)|((bidx&0x1)<<3);
	addry = ((tidx&0x38)<<3)|(bidx>>1);
#pragma unroll
	for (int i=0; i<8; i++)
    samples[i] = src[(i<<13)|fmem];
	_ntt8(samples);
#pragma unroll
	for (int i=0; i<8; i++)
    buffer[(i<<3)|tbuf] = _ls_modP(samples[i], (tidx>>3)*i*3);
	__syncthreads();
#pragma unroll
	for (int i=0; i<8; i++)
    samples[i] = buffer[(i<<6)|fbuf];
	_ntt8(samples);
	register uint64 root;
#pragma unroll
	for (int i=0; i<8; i++) {
		root = tex1Dfetch(tex_roots_64k, ((addry|(i<<9))*addrx*2)+1);
		root <<= 32;
		root += tex1Dfetch(tex_roots_64k, (addry|(i<<9))*addrx*2);
		src[(i<<13)|tmem] = _mul_modP(samples[i], root);
	}
}
__global__ void ntt_3_64k(uint64 *dst, uint64 *src){
	__shared__ uint64 buffer[512];
	register uint64 samples[8];
	register uint32 fmem, tmem, fbuf, tbuf;
	fmem = (bidx<<9)|((tidx&0x3E)<<3)|(tidx&0x1);
	tbuf = tidx<<3;
	fbuf = ((tidx&0x38)<<3)|(tidx&0x7);
	tmem = (bidx<<9)|((tidx&0x38)<<3)|(tidx&0x7);
#pragma unroll
	for (int i=0; i<8; i++)
    samples[i] = src[fmem|(i<<1)];
	_ntt8(samples);
#pragma unroll
	for (int i=0; i<8; i++)
    buffer[i|tbuf] = _ls_modP(samples[i], ((tidx&0x1)<<2)*i*3);
	__syncthreads();
#pragma unroll
	for (int i=0; i<8; i++)
    samples[i] = buffer[fbuf|(i<<3)];
	register uint32 addr;
#pragma unroll
	for (int i=0; i<4; i++) {
		addr = tmem|((2*i)<<3);
		dst[((addr&0xf)<<12)|(addr>>4)] = _add_modP(samples[2*i], samples[2*i+1]);
		addr = tmem|((2*i+1)<<3);
		dst[((addr&0xf)<<12)|(addr>>4)] = _sub_modP(samples[2*i], samples[2*i+1]);
	}
}

} // namespace cuHE
