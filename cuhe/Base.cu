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

#include "Base.h"
#include "DeviceManager.h"
#include "Debug.h"
#include "ModP.h"
#include <NTL/ZZ.h>
#include <NTL/ZZX.h>
NTL_CLIENT

#define	bidx	blockIdx.x
#define	bidy	blockIdx.y
#define	bidz	blockIdx.z
#define	tidx	threadIdx.x
#define	tidy	threadIdx.y
#define	tidz	threadIdx.z
#define	bdimx	blockDim.x
#define	bdimy	blockDim.y
#define	bdimz	blockDim.z
#define	gdimx	gridDim.x
#define	gdimy	gridDim.y
#define	gdimz	gridDim.z

namespace cuHE {

///////////////////////////////////////////////////////////
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
// crt constant memory
#define maxNumPrimes 103 // chosen for 64 KB constant memory
__constant__ uint32 const_p[maxNumPrimes];
__constant__ uint32 const_invp[maxNumPrimes*(maxNumPrimes-1)/2];
__constant__ uint32 const_M[maxNumPrimes];
__constant__ uint32 const_mi[maxNumPrimes*maxNumPrimes];
__constant__ uint32 const_bi[maxNumPrimes];
void preload_crt_p(uint32* src, int words) {
	for (int dev=0; dev<numDevices(); dev++) {
		CSC(cudaSetDevice(dev));
		CSC(cudaMemcpyToSymbol(const_p, src,
				words*sizeof(uint32), 0, cudaMemcpyHostToDevice));
	}
}
void preload_crt_invp(uint32* src, int words) {
	for (int dev=0; dev<numDevices(); dev++) {
		CSC(cudaSetDevice(dev));
		CSC(cudaMemcpyToSymbol(const_invp, src, words*sizeof(uint32),
				0, cudaMemcpyHostToDevice));
	}
}
void load_icrt_M(uint32* src, int words, int dev, cudaStream_t st) {
	CSC(cudaSetDevice(dev));
	CSC(cudaMemcpyToSymbolAsync(const_M, src, words*sizeof(uint32),
			0, cudaMemcpyHostToDevice, st));
}
void load_icrt_mi(uint32* src, int words, int dev, cudaStream_t st) {
	CSC(cudaSetDevice(dev));
	CSC(cudaMemcpyToSymbolAsync(const_mi, src, words*sizeof(uint32),
			0, cudaMemcpyHostToDevice, st));
}
void load_icrt_bi(uint32* src, int words, int dev, cudaStream_t st) {
	CSC(cudaSetDevice(dev));
	CSC(cudaMemcpyToSymbolAsync(const_bi, src, words*sizeof(uint32),
			0, cudaMemcpyHostToDevice, st));
}
// barrett reduction texture and device memory
static uint64 **d_u_ntt;
static uint64 **d_m_ntt;
static uint32 **d_m_crt;
texture<uint32, 1> tex_u_ntt;
texture<uint32, 1> tex_m_ntt;
texture<uint32, 1> tex_m_crt;
void preload_barrett_u_n(uint64* src, size_t size) {
	d_u_ntt = new uint64*[numDevices()];
	for (int dev=0; dev<numDevices(); dev++) {
		CSC(cudaSetDevice(dev));
		CSC(cudaMalloc(&d_u_ntt[dev], size));
	}
	CSC(cudaSetDevice(0));
	for (int dev=0; dev<numDevices(); dev++)
		CSC(cudaMemcpyPeer(d_u_ntt[dev], dev, src, 0, size));
	for (int dev=0; dev<numDevices(); dev++) {
		CSC(cudaSetDevice(dev));
		CSC(cudaBindTexture(NULL, tex_u_ntt, d_u_ntt[dev], size));
	}
	CSC(cudaStreamSynchronize(0));
}
void preload_barrett_m_n(uint64* src, size_t size) {
	d_m_ntt = new uint64*[numDevices()];
	for (int dev=0; dev<numDevices(); dev++) {
		CSC(cudaSetDevice(dev));
		CSC(cudaMalloc(&d_m_ntt[dev], size));
	}
	CSC(cudaSetDevice(0));
	for (int dev=0; dev<numDevices(); dev++)
		CSC(cudaMemcpyPeer(d_m_ntt[dev], dev, src, 0, size));
	for (int dev=0; dev<numDevices(); dev++) {
		CSC(cudaSetDevice(dev));
		CSC(cudaBindTexture(NULL, tex_m_ntt, d_m_ntt[dev], size));
	}
}
void preload_barrett_m_c(uint32* src, size_t size) {
	d_m_crt = new uint32*[numDevices()];
	for (int dev=0; dev<numDevices(); dev++) {
		CSC(cudaSetDevice(dev));
		CSC(cudaMalloc(&d_m_crt[dev], size));
	}
	CSC(cudaSetDevice(0));
	for (int dev=0; dev<numDevices(); dev++)
		CSC(cudaMemcpyPeer(d_m_crt[dev], dev, src, 0, size));
	for (int dev=0; dev<numDevices(); dev++) {
		CSC(cudaSetDevice(dev));
		CSC(cudaBindTexture(NULL, tex_m_crt, d_m_crt[dev], size));
	}
}

///////////////////////////////////////////////////////////
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
// 16384-point NTT/INTT
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
#pragma unroll
  for(int i=0; i<8; i++)
    (dst+(bidy<<14))[tmem|(i<<5)] = _mul_modP(samples[i],
    		roots[((tidx&0x7)>>2<<6)|(tidx>>3)|(i<<3)]);
}
__global__ void ntt_1_16k_ext_block(uint64 *dst, uint32 *src, int w, int wid,
		int w32){
	__shared__ uint64 buffer[512], roots[128];
	register uint64 samples[8];
	register uint32 fmem, tmem, fbuf, tbuf;
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
#pragma unroll
	for (int i=0; i<4; i++) {
		if ((((w*wid)>>5)+1) < w32) {
			samples[i] = (src+(bidy<<14)*w32)[((i<<11)|fmem)*w32+((w*wid)>>5)+1];
			samples[i] <<= 32;
			samples[i] += (src+(bidy<<14)*w32)[((i<<11)|fmem)*w32+((w*wid)>>5)];
		}
		else
			samples[i] = (src+(bidy<<14)*w32)[((i<<11)|fmem)*w32+((w*wid)>>5)];
		samples[i] >>= (w*wid)&0x1f;
		samples[i] &= (0x1<<w)-1;
	}
	_ntt8_ext(samples);
#pragma unroll
	for (int i=0; i<8; i++)
		buffer[(i<<3)|tbuf] = _ls_modP(samples[i], (tidx>>3)*i*3);
	__syncthreads();
#pragma unroll
	for (int i=0; i<8; i++)
		samples[i] = buffer[(i<<6)|fbuf];
	_ntt8(samples);
#pragma unroll
	for (int i=0; i<8; i++)
		(dst+(bidy<<14))[tmem|(i<<5)] = _mul_modP(samples[i],
				roots[((tidx&0x7)>>2<<6)|(tidx>>3)|(i<<3)]);
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
    samples[i] = (src+(bidy<<14))[(i<<11)|fmem];
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
    (src+(bidy<<14))[(i<<11)|tmem] = _mul_modP(samples[i], root);
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
    buffer[(i<<6)|tbuf] = (src+(bidy<<14))[(i<<6)|fmem];
  __syncthreads();
#pragma unroll
  for (int i=0; i<8; i++)
    samples[i] = buffer[fbuf|i];
  _ntt4(samples);
  _ntt4(samples+4);
#pragma unroll
  for (int i=0; i<8; i++)
    (dst+(bidy<<14))[((i&0x3)<<12)|((i&0x7)>>2)|tmem] = samples[i];
}
__global__ void intt_1_16k(uint64 *dst, uint64 *src) {
	__shared__ uint64 buffer[512], roots[128];
	register uint64 samples[8];
	register uint32 fmem, tmem, fbuf, tbuf;
	fmem = (tidx>>3<<8)|(bidx<<3)|((tidx&0x7)>>2<<2)|(tidx&0x3);
	tbuf = ((tidx&0x38)<<3)|(tidx&0x7);
	fbuf = tidx;
	tmem = (bidx<<9)|((tidx&0x7)>>2<<8)|(tidx>>3<<2)|(tidx&0x3);
	roots[tidx] = tex1Dfetch(tex_roots_16k, (((bidx*2)*tidx)<<3)+1);
	roots[tidx] <<= 32;
	roots[tidx] += tex1Dfetch(tex_roots_16k, ((bidx*2)*tidx)<<3);
	roots[tidx+64] = tex1Dfetch(tex_roots_16k, (((bidx*2+1)*tidx)<<3)+1);
	roots[tidx+64] <<= 32;
	roots[tidx+64] += tex1Dfetch(tex_roots_16k, ((bidx*2+1)*tidx)<<3);
#pragma unroll
	for (int i=0; i<8; i++)
		samples[i] = (src+(bidy<<14))[(16384-((i<<11)|fmem))%16384];
	_ntt8(samples);
#pragma unroll
	for (int i=0; i<8; i++)
		buffer[(i<<3)|tbuf] = _ls_modP(samples[i], (tidx>>3)*i*3);
	__syncthreads();
#pragma unroll
	for (int i=0; i<8; i++)
		samples[i] = buffer[(i<<6)|fbuf];
	_ntt8(samples);
#pragma unroll
	for (int i=0; i<8; i++)
		(dst+(bidy<<14))[tmem|(i<<5)] = _mul_modP(samples[i],
				roots[((tidx&0x7)>>2<<6)|(tidx>>3)|(i<<3)]);
}
__global__ void intt_3_16k_modcrt(uint32 *dst, uint64 *src, int crtidx) {
	__shared__ uint64 buffer[512];
	register uint64 samples[8];
	register uint32 fmem, tmem, fbuf, tbuf;
	fmem = (bidx<<9)|tidx;
	tbuf = tidx;
	fbuf = tidx<<3;
	tmem = (bidx<<7)|(tidx<<1);
#pragma unroll
	for (int i=0; i<8; i++)
		buffer[(i<<6)|tbuf] = (src+(bidy<<14))[(i<<6)|fmem];
	__syncthreads();
#pragma unroll
	for (int i=0; i<8; i++)
		samples[i] = buffer[fbuf|i];
	_ntt4(samples);
	_ntt4(samples+4);
#pragma unroll
	for (int i=0; i<8; i++)
		(dst+(bidy<<14))[((i&0x3)<<12)|((i&0x7)>>2)|tmem] = 
			(uint32)(_mul_modP(samples[i], 18445618169508003841)%const_p[crtidx]);
}
// 32768-point NTT/INTT
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
    samples[i] = (src+(bidy<<15))[(i<<12)|fmem];
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
    (dst+(bidy<<15))[tmem|(i<<6)] = _mul_modP(samples[i],
    		roots[(tidx>>3)|(i<<3)]);
}
__global__ void ntt_1_32k_ext_block(uint64 *dst, uint32 *src, int w, int wid,
		int w32) {
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
	for (int i=0; i<4; i++) {
		if ((((w*wid)>>5)+1) < w32) {
			samples[i] = (src+(bidy<<15)*w32)[((i<<12)|fmem)*w32+((w*wid)>>5)+1];
			samples[i] <<= 32;
			samples[i] += (src+(bidy<<15)*w32)[((i<<12)|fmem)*w32+((w*wid)>>5)];
		}
		else
			samples[i] = (src+(bidy<<15)*w32)[((i<<12)|fmem)*w32+((w*wid)>>5)];
		samples[i] >>= (w*wid)&0x1f;
		samples[i] &= (0x1<<w)-1;
	}
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
		(dst+(bidy<<15))[tmem|(i<<6)] = _mul_modP(samples[i],
				roots[(tidx>>3)|(i<<3)]);
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
    samples[i] = (src+(bidy<<15))[(i<<12)|fmem];
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
		(src+(bidy<<15))[(i<<12) | tmem] = _mul_modP(samples[i], root);
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
    buffer[(i<<6)|tbuf] = (src+(bidy<<15))[(i<<6)|fmem];
	__syncthreads();
#pragma unroll
	for (int i=0; i<8; i++)
    samples[i] = buffer[fbuf|i];
	_ntt8(samples);
#pragma unroll
	for (int i=0; i<8; i++)
    (dst+(bidy<<15))[(i<<12)|tmem] = samples[i];
}
__global__ void intt_1_32k(uint64 *dst, uint64 *src) {
	__shared__ uint64 buffer[512], roots[64];
	register uint64 samples[8];
	register uint32 fmem, tmem, fbuf, tbuf;
	fmem = ((tidx&0x38)<<6)|(bidx<<3)|(tidx&0x7);
	tbuf = (tidx&0x38)<<3|(tidx&0x7);
	fbuf = tidx;
	tmem = (bidx<<9)|fbuf;
	roots[tidx] = tex1Dfetch(tex_roots_32k, ((bidx*tidx)<<4)+1);
	roots[tidx] <<= 32;
	roots[tidx] += tex1Dfetch(tex_roots_32k, (bidx*tidx)<<4);
#pragma unroll
	for (int i=0; i<8; i++)
		samples[i] = (src+(bidy<<15))[(32768-((i<<12)|fmem))%32768];
	_ntt8(samples);
#pragma unroll
	for (int i=0; i<8; i++)
		buffer[(i<<3)|tbuf] = _ls_modP(samples[i], ((tidx>>3)*i)*3);
	__syncthreads();
#pragma unroll
	for (int i=0; i<8; i++)
		samples[i] = buffer[fbuf|(i<<6)];
	_ntt8(samples);
#pragma unroll
	for (int i=0; i<8; i++)
		(dst+(bidy<<15))[tmem|(i<<6)] = _mul_modP(samples[i],
				roots[(tidx>>3)|(i<<3)]);
}
__global__ void intt_3_32k_modcrt(uint32 *dst, uint64 *src, int crtidx) {
	__shared__ uint64 buffer[512];
	register uint64 samples[8];
	register uint32 fmem, tmem, fbuf, tbuf;
	fmem = (bidx<<9)|tidx;
	tbuf = tidx;
	fbuf = tidx<<3;
	tmem = (bidx<<6)|tidx;
#pragma unroll
	for (int i=0; i<8; i++)
		buffer[(i<<6)|tbuf] = (src+(bidy<<15))[(i<<6)|fmem];
	__syncthreads();
#pragma unroll
	for (int i=0; i<8; i++)
		samples[i] = buffer[fbuf|i];
	_ntt8(samples);
#pragma unroll
	for (int i=0; i<8; i++)
		(dst+(bidy<<15))[(i<<12)|tmem] =
			(uint32)(_mul_modP(samples[i], 18446181119461294081)%const_p[crtidx]);
}
// 65536-point NTT/INTT
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
    samples[i] = (src+(bidy<<16))[(i<<13)|fmem];
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
    (dst+(bidy<<16))[tmem | (i<<7)] = _mul_modP(samples[i],
    		roots[(tidx>>3)|(i<<3)]);
}
__global__ void ntt_1_64k_ext_block(uint64 *dst, uint32 *src, int w, int wid,
	int w32) {
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
	for (int i=0; i<4; i++) {
		if ((((w*wid)>>5)+1) < w32) {
			samples[i] = (src+(bidy<<16)*w32)[((i<<13)|fmem)*w32+((w*wid)>>5)+1];
			samples[i] <<= 32;
			samples[i] += (src+(bidy<<16)*w32)[((i<<13)|fmem)*w32+((w*wid)>>5)];
		}
		else
			samples[i] = (src+(bidy<<16)*w32)[((i<<13)|fmem)*w32+((w*wid)>>5)];
		samples[i] >>= (w*wid)&0x1f;
		samples[i] &= (0x1<<w)-1;
	}
	_ntt8_ext(samples);
#pragma unroll
	for (int i=0; i<8; i++)
		buffer[(i<<3)|tbuf] = _ls_modP(samples[i], ((tidx>>3)*i)*3);
	__syncthreads();
#pragma unroll
	for (int i=0; i<8; i++)
		samples[i] = buffer[fbuf|(i<<6)];
	_ntt8(samples);
#pragma unroll
	for (int i=0; i<8; i++)
		(dst+(bidy<<16))[tmem|(i<<7)] = _mul_modP(samples[i],
				roots[(tidx>>3)|(i<<3)]);
}
__global__ void ntt_2_64k(uint64 *src) {
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
    samples[i] = (src+(bidy<<16))[(i<<13)|fmem];
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
		(src+(bidy<<16))[(i<<13)|tmem] = _mul_modP(samples[i], root);
	}
}
__global__ void ntt_3_64k(uint64 *dst, uint64 *src) {
	__shared__ uint64 buffer[512];
	register uint64 samples[8];
	register uint32 fmem, tmem, fbuf, tbuf;
	fmem = (bidx<<9)|((tidx&0x3E)<<3)|(tidx&0x1);
	tbuf = tidx<<3;
	fbuf = ((tidx&0x38)<<3)|(tidx&0x7);
	tmem = (bidx<<9)|((tidx&0x38)<<3)|(tidx&0x7);
#pragma unroll
	for (int i=0; i<8; i++)
    samples[i] = (src+(bidy<<16))[fmem|(i<<1)];
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
		(dst+(bidy<<16))[((addr&0xf)<<12)|(addr>>4)] = _add_modP(samples[2*i],
				samples[2*i+1]);
		addr = tmem|((2*i+1)<<3);
		(dst+(bidy<<16))[((addr&0xf)<<12)|(addr>>4)] = _sub_modP(samples[2*i],
				samples[2*i+1]);
	}
}
__global__ void intt_1_64k(uint64 *dst, uint64 *src) {
	__shared__ uint64 buffer[512], roots[64];
	register uint64 samples[8];
	register uint32 fmem, tmem, fbuf, tbuf;
	fmem = ((tidx&0x38)<<7)|(bidx<<3)|(tidx&0x7);
	tbuf = (tidx&0x38)<<3|(tidx&0x7);
	fbuf = tidx;
	tmem = ((bidx&0x7E)<<9)|((tidx&0x38)<<1)|((bidx&0x1)<<3)|(tidx&0x7);
	roots[tidx] = tex1Dfetch(tex_roots_64k, (((bidx>>1)*tidx)<<5)+1);
	roots[tidx] <<= 32;
	roots[tidx] += tex1Dfetch(tex_roots_64k, ((bidx>>1)*tidx)<<5);
#pragma unroll
	for (uint32 i=0; i<8; i++)
		samples[i] = (src+(bidy<<16))[(65536-(fmem|(i<<13)))%65536];
	_ntt8(samples);
#pragma unroll
	for (int i=0; i<8; i++)
		buffer[(i<<3)|tbuf] = _ls_modP(samples[i], ((tidx>>3)*i)*3);
	__syncthreads();
#pragma unroll
	for (int i=0; i<8; i++)
		samples[i] = buffer[fbuf|(i<<6)];
	_ntt8(samples);
#pragma unroll
	for (int i=0; i<8; i++)
		(dst+(bidy<<16))[tmem|(i<<7)] = _mul_modP(samples[i],
				roots[(tidx>>3)|(i<<3)]);
}
__global__ void intt_3_64k_modcrt(uint32 *dst, uint64 *src, int crtidx) {
	__shared__ uint64 buffer[512];
	register uint64 samples[8], s8[8];
	register uint32 fmem, tmem, fbuf, tbuf;
	fmem = (bidx<<9)|((tidx&0x3E)<<3)|(tidx&0x1);
	tbuf = tidx<<3;
	fbuf = ((tidx&0x38)<<3) | (tidx&0x7);
	tmem = (bidx<<9)|((tidx&0x38)<<3) | (tidx&0x7);
#pragma unroll
	for (int i=0; i<8; i++)
		samples[i] = (src+(bidy<<16))[fmem|(i<<1)];
	_ntt8(samples);
#pragma unroll
	for (int i=0; i<8; i++)
		buffer[tbuf|i] = _ls_modP(samples[i], ((tidx&0x1)<<2)*i*3);
	__syncthreads();
#pragma unroll
	for (int i=0; i<8; i++)
		samples[i] = buffer[fbuf|(i<<3)];
#pragma unroll
	for (int i=0; i<4; i++) {
		s8[2*i] = _add_modP(samples[2*i], samples[2*i+1]);
		s8[2*i+1] = _sub_modP(samples[2*i], samples[2*i+1]);
	}
#pragma unroll
	for (int i=0; i<8; i++)
		(dst+(bidy<<16))[(((tmem|(i<<3))&0xf)<<12)|((tmem|(i<<3))>>4)] =
			(uint32)(_mul_modP(s8[i], 18446462594437939201)%const_p[crtidx]);
}

/** ----- CRT kernels -----	*/
/**	crt kernels	*/
__device__ __inline__
bool leq_M(unsigned int *x, int M_w32) {	// less or equal than M ?
	if (x[M_w32] > 0)
		return true;
	for (int i=M_w32-1; i>=0; i--) {
		if (x[i] < const_M[i])
			return false;
		else if (x[i] > const_M[i])
			return true;
	}
	return true;
}

__global__ void crt(uint32 *dst, uint32 *src, int pnum, int w32, int mlen, int clen) {
	register int idx = bidx*bdimx+tidx;
	extern __shared__ uint32 buff[];
	uint32 *in = buff;
	for (int i=0; i<w32; i++)
		in[bdimx*i+tidx] = src[bidx*bdimx*w32+bdimx*i+tidx];
	__syncthreads();
	if (idx < mlen) {
		uint32 *coeff = &in[tidx*w32];
		register uint32 h, l;
		for (int crt=0; crt<pnum; crt++) {
			l = coeff[w32-1];
			l %= const_p[crt];
			for (int i = w32-2; i>=0; i--) {
				h = l;
				l = coeff[i]%const_p[crt];
				l = (uint32)((((uint64)h<<32)+l)%const_p[crt]);
			}
			dst[crt*clen+idx] = l;
		}
	}
}

__global__ void icrt(uint32 *dst, uint32 *src, int pnum, int M_w32, int mi_w32, int mlen, int clen) {
	int idx = bidx*bdimx+tidx;
	if (idx < mlen) {
		register uint32 sum[maxNumPrimes+1];
		register uint64 tar;
		register uint32 tt;
		for (int i=0; i<maxNumPrimes+1; i++)
			sum[i] = 0;
		for (int crt=0; crt<pnum; crt++) {
			tar = src[crt*clen+idx];
			tar %= const_p[crt];
			tar *= const_bi[crt];
			tt = (uint32)(tar%const_p[crt]);
			// low
			asm("mad.lo.cc.u32 %0,%1,%2,%3;" : "=r"(sum[0]) : "r"(tt), "r"(const_mi[crt*mi_w32+0]), "r"(sum[0]));
			for (int i=1; i<mi_w32; i++)
				asm("madc.lo.cc.u32 %0,%1,%2,%3;" : "=r"(sum[i]) : "r"(tt), "r"(const_mi[crt*mi_w32+i]), "r"(sum[i]));
			for (int i=mi_w32; i<M_w32; i++)
				asm("addc.cc.u32 %0,%0,%1;" : "+r"(sum[i]) : "r"(0));
			asm("addc.u32 %0,%0,%1;" : "+r"(sum[M_w32]) : "r"(0));
			// high
			asm("mad.hi.cc.u32 %0,%1,%2,%3;" : "=r"(sum[1]) : "r"(tt), "r"(const_mi[crt*mi_w32+0]), "r"(sum[1]));
			for (int i=1; i<mi_w32; i++)
				asm("madc.hi.cc.u32 %0,%1,%2,%3;" : "=r"(sum[i+1]) : "r"(tt), "r"(const_mi[crt*mi_w32+i]), "r"(sum[i+1]));
			for (int i=mi_w32; i<M_w32-1; i++)
				asm("addc.cc.u32 %0,%0,%1;" : "+r"(sum[i+1]) : "r"(0));
			asm("addc.u32 %0,%0,%1;" : "+r"(sum[M_w32]) : "r"(0));
			// mod M
			if (leq_M(sum, M_w32)) {
				asm("sub.cc.u32 %0,%0,%1;" : "+r"(sum[0]) : "r"(const_M[0]));
				for (int i=1; i<M_w32; i++)
					asm("subc.cc.u32 %0,%0,%1;" : "+r"(sum[i]) : "r"(const_M[i]));
				asm("subc.u32 %0,%0,%1;" : "+r"(sum[M_w32]) : "r"(0));
			}
		}
		for (int i=0; i<M_w32; i++) {
			dst[bidx*M_w32*bdimx+tidx*M_w32+i] = sum[i];
		}
		// not coalesced???
	}
}
/* end CRT kernels	*/




/** ----- Barrett kernels -----	*/
__global__ void barrett_mul_un(uint64 *tar, int pnum, int nlen) {
	register int idx = bidx*bdimx+tidx;
	register uint64 a, b, offset = 0;
	for(int crt=0; crt<pnum; crt++){
		a = tar[offset+idx];
		b = tex1Dfetch(tex_u_ntt, offset*2+idx*2+1);
		b <<= 32;
		b += tex1Dfetch(tex_u_ntt, offset*2+idx*2);
		tar[offset+idx] = _mul_modP(a, b);
		offset += nlen;
	}
}

__global__ void barrett_mul_mn(uint64 *tar, int pnum, int nlen) {
	register int idx = bidx*bdimx+tidx;
	register uint64 a, b, offset = 0;
	for(int crt=0; crt<pnum; crt++){
		a = tar[offset+idx];
		b = tex1Dfetch(tex_m_ntt, offset*2+idx*2+1);
		b <<= 32;
		b += tex1Dfetch(tex_m_ntt, offset*2+idx*2);
		tar[offset+idx] = _mul_modP(a, b);
		offset += nlen;
	}
}

__global__ void barrett_sub_1(uint32 *y, uint32 *x, int pnum, int mlen, int nlen) {
	register int idx = bidx*bdimx+tidx;
	register uint32 a, b;
	if (idx < mlen) {
		for (int crt=0; crt<pnum; crt++) {
			a = y[crt*nlen+mlen+idx];
			b = x[crt*nlen+mlen+idx];
			if (a < b)
				a += const_p[crt];
			a -= b;
			y[crt*nlen+mlen+idx] = a;
		}
	}
}

__global__ void barrett_sub_2(uint32 *y, uint32 *x, int pnum, int nlen) {
	register int idx = bidx*bdimx+tidx;
	register uint32 a, b;
	for (int crt=0; crt<pnum; crt++) {
		a = y[crt*nlen+idx];
		b = x[crt*nlen+idx];
		if (a < b)
			a += const_p[crt];
		a -= b;
		y[crt*nlen+idx] = a;
	}
}

__global__ void barrett_sub_mc(uint32 *x, int pnum, int mlen, int clen, int nlen) {
	register int idx = bidx*bdimx+tidx;
	register uint32 d, s;
	extern __shared__ uint32 t[];
	int offset = 0;
	while (offset+tidx < pnum) {
		t[offset+tidx] = x[(offset+tidx)*nlen+mlen];
		offset += bdimx;
	}
	__syncthreads();
	if (idx < mlen-1) {
		for (int crt=0; crt<pnum; crt++) {
			if (t[crt] > 0) {
				d = x[crt*nlen+idx];
				s = tex1Dfetch(tex_m_crt, crt*clen+idx);
				if (d < s)
					d += const_p[crt];
				d -= s;
				x[crt*nlen+idx] = d;
			}
		}
	}
}
/* end Barrett kernels	*/

/** ----- Relinearization kernels -----	*/
// ek[knum][pnum][NTTLEN]
__global__ void relinMulAddAll(uint64 *dst, uint64 *c, uint64 *ek, int pnum, int knum, int nlen) {
	register int idx = bidx*bdimx+tidx;
	extern __shared__ uint64 cbuff[];	// cache c[knum*bdimx]
	for (int i=0; i<knum; i++)
		cbuff[i*bdimx+tidx] = c[i*nlen+idx];
	__syncthreads();
	register uint64 sum = 0, offset = 0;
	for (int i=0; i<pnum; i++) {
		sum = 0;
		offset = i*nlen;
		for (int j=0; j<knum; j++) {
			sum = _add_modP(sum, _mul_modP(cbuff[j*bdimx+tidx], ek[offset+idx]));
			offset += pnum*nlen;
		}
		dst[i*nlen+idx] = sum;
	}
}

// pnum * ek[knum][NTTLEN]
__global__ void relinMulAddPerCrt(uint64 *dst, uint64 *c, uint64 *ek, int knum, int nlen) {
	register int idx = bidx*bdimx+tidx;
	register uint64 sum = 0, offset = 0;
	for (int i=0; i<knum; i++) {
		sum = _add_modP(sum, _mul_modP(c[offset+idx], ek[offset+idx]));
		offset += nlen;
	}
	dst[idx] = sum;
}
/* end Relinearization kernels	*/


/** Operations	*/
__global__ void ntt_mul(uint64 *z, uint64 *x, uint64 *y, int pnum, int nlen) {
	register int idx = bidx*bdimx+tidx;
	register uint64 a, b, offset = 0;
	for (int crt=0; crt<pnum; crt++) {
		a = x[offset+idx];
		b = y[offset+idx];
		z[offset+idx] = _mul_modP(a, b);
		offset += nlen;
	}
}

__global__ void ntt_add(uint64 *z, uint64 *x, uint64 *y, int pnum, int nlen) {
	register int idx = bidx*bdimx+tidx;
	register uint64 a, b, offset = 0;
	for (int crt=0; crt<pnum; crt++) {
		a = x[offset+idx];
		b = y[offset+idx];
		z[offset+idx] = _add_modP(a, b);
		offset += nlen;
	}
}

__global__ void ntt_mul_nx1(uint64 *z, uint64 *x, uint64 *scalar, int pnum, int nlen) {
	register int idx = bidx*bdimx+tidx;
	register uint64 a, b = scalar[idx], offset = 0;
	for (int crt=0; crt<pnum; crt++) {
		a = x[offset+idx];
		z[offset+idx] = _mul_modP(a, b);
		offset += nlen;
	}
}

__global__ void ntt_add_nx1(uint64 *z, uint64 *x, uint64 *scalar, int pnum, int nlen) {
	register int idx = bidx*bdimx+tidx;
	register uint64 a, b = scalar[idx], offset = 0;
	for (int crt=0; crt<pnum; crt++) {
		a = x[offset+idx];
		z[offset+idx] = _add_modP(a, b);
		offset += nlen;
	}
}

__global__ void crt_mul_int(uint32 *y, uint32 *x, int a, int pnum, int clen) {
	register int idx = bidx*bdimx+tidx;
	if (idx < pnum) {
		register uint64 temp;
		temp = x[idx*clen];
		temp *= a;
		temp %= const_p[idx];
		y[idx*clen] = temp;
	}
}

__global__ void crt_add(uint32 *x, uint32 *a, uint32 *b, int pnum, int mlen, int clen) {
	register int idx = bidx*bdimx+tidx;
	if (idx < mlen) {
		for (int i=0; i<pnum; i++)
			x[i*clen+idx] = (a[i*clen+idx]+b[i*clen+idx])%const_p[i];
	}
}

__global__ void crt_add_int(uint32 *y, uint32 *x, int a, int pnum, int clen) {
	register int idx = bidx*bdimx+tidx;
	if (idx < pnum)
		y[idx*clen] = (x[idx*clen]+(a%const_p[idx]))%const_p[idx];
}

__global__ void crt_add_nx1(uint32 *x, uint32 *a, uint32 *scalar, int pnum, int mlen, int clen) {
	register int idx = bidx*bdimx+tidx;
	register uint32 b = scalar[idx];
	if (idx < mlen) {
		for (int i=0; i<pnum; i++)
			x[i*clen+idx] = (a[i*clen+idx]+b)%const_p[i];
	}
}

__global__ void modswitch(uint32 *dst, uint32 *src, int pnum, int mlen, int clen, int modmsg) {
	register int idx = bidx*bdimx+tidx;
	if (idx < mlen) {
		register int dirty = src[(pnum-1)*clen+idx];
		register uint32 pt = const_p[pnum-1];
		register int ep = dirty%modmsg;
		if (ep != 0) {
			if (dirty>((pt-1)/2))
				dirty -= ep*pt;
			else
				dirty += ep*pt;
		}
		register int temp;
		register uint64 tt;
		for (int i=0; i<pnum-1; i++) {
			temp = src[i*clen+idx];
			while (temp < dirty)
				temp += const_p[i];
			temp -= dirty;
			tt = temp;
			tt *= const_invp[(pnum-1)*(pnum-2)/2+i];
			tt %= const_p[i];
			dst[i*clen+idx] = (uint32)tt;
		}
	}
}

} // end cuHE
