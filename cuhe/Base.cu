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

//#include "Parameters.h"
//#include "CuHE.h"

namespace cuHE {
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ntt texture and device memory
static uint64** d_roots;
texture<uint32, 1> tex_roots_16k;
texture<uint32, 1> tex_roots_32k;
texture<uint32, 1> tex_roots_64k;
void preload_ntt(uint64* src, int len) {
	d_roots = new uint64* [numDevices()];
	for (int dev=0; dev<numDevices(); dev++) {
		CSC(cudaSetDevice(dev));
		CSC(cudaMalloc(&d_roots[dev], len*sizeof(uint64)));
		CSC(cudaMemcpy(d_roots[dev], src,
				len*sizeof(uint64), cudaMemcpyHostToDevice));
		switch (len) {
		case 16384:
			CSC(cudaBindTexture(NULL, tex_roots_16k,
					d_roots[dev], len*sizeof(uint64)));
			break;
		case 32768:
			CSC(cudaBindTexture(NULL, tex_roots_32k,
					d_roots[dev], len*sizeof(uint64)));
			break;
		case 65536:
			CSC(cudaBindTexture(NULL, tex_roots_64k,
					d_roots[dev], len*sizeof(uint64)));
			break;
		}
	}
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// name def
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

/** ----- NTT kernels -----	*/
// 4-point NTT
__device__ __host__ __inline__
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
__device__ __host__ __inline__
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
__device__ __host__ __inline__
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

__global__ void ntt_1_16k_ext(uint64 *dst, uint32 *src){
	__shared__ uint64 buffer[512], roots[128];//always compute 8 64-sample ntt's in a block
	register uint64 samples[8];//ntt samples and results
	register uint32 from_mem, to_mem, from_buffer, to_buffer;//address mapping
	//coalesced global memory access & minimum shared memory bank conflicts
	from_mem	=	((tidx&0x38)<<5) | (bidx<<3) | (tidx&0x7);
	to_buffer	=	((tidx&0x38)<<3) | (tidx&0x7);
	from_buffer	=	tidx;
	to_mem		=	(bidx<<9) | ((tidx&0x7)>>2<<8) | (tidx>>3<<2) | (tidx&0x3);
	roots[tidx] 	=	tex1Dfetch(tex_roots_16k, (((bidx*2)*tidx)<<3)+1);
	roots[tidx] 	<<=	32;
	roots[tidx] 	+=	tex1Dfetch(tex_roots_16k, ((bidx*2)*tidx)<<3);
	roots[tidx+64] 	=	tex1Dfetch(tex_roots_16k, (((bidx*2+1)*tidx)<<3)+1);
	roots[tidx+64] 	<<=	32;
	roots[tidx+64] 	+=	tex1Dfetch(tex_roots_16k, ((bidx*2+1)*tidx)<<3);
	//load 4 or 8 samples from global memory and compute ntt of 8 samples
#pragma unroll
	for(int i=0; i<4; i++)
		samples[i]=src[(i<<11)|from_mem];
	_ntt8_ext(samples);
	//times twiddle factors of 64 samples, store to buffer
#pragma unroll
	for(int i=0; i<8; i++)
		buffer[(i<<3) | to_buffer]=_ls_modP(samples[i], (tidx>>3)*i*3);
	__syncthreads();
	//load 8 samples from shared memory in transposed way and compute ntt of 8 samples
#pragma unroll
	for(int i=0; i<8; i++)
		samples[i]=buffer[(i<<6) | from_buffer];
	_ntt8(samples);
#pragma unroll
	for(int i=0; i<8; i++)
		dst[to_mem | (i<<5)] = _mul_modP(samples[i], roots[((tidx&0x7)>>2<<6) | (tidx>>3) | (i<<3)]);
//		dst[i] = (uint32)tex1Dfetch(tex_roots_16k, i);//roots[((tidx&0x7)>>2<<6) | (tidx>>3) | (i<<3)];
}

__global__ void ntt_1_16k_ext_block(uint64 *dst, uint32 *src, int chunksize, int chunkid, int w32){
	__shared__ uint64 buffer[512], roots[128];//always compute 8 64-sample ntt's in a block
	register uint64 samples[8];//ntt samples and results
	register uint32 from_mem, to_mem, from_buffer, to_buffer;//address mapping
	//coalesced global memory access & minimum shared memory bank conflicts
	from_mem	=	((tidx&0x38)<<5) | (bidx<<3) | (tidx&0x7);
	to_buffer	=	((tidx&0x38)<<3) | (tidx&0x7);
	from_buffer	=	tidx;
	to_mem		=	(bidx<<9) | ((tidx&0x7)>>2<<8) | (tidx>>3<<2) | (tidx&0x3);
	roots[tidx] 	=	tex1Dfetch(tex_roots_16k, (((bidx*2)*tidx)<<3)+1);
	roots[tidx] 	<<=	32;
	roots[tidx] 	+=	tex1Dfetch(tex_roots_16k, ((bidx*2)*tidx)<<3);
	roots[tidx+64] 	=	tex1Dfetch(tex_roots_16k, (((bidx*2+1)*tidx)<<3)+1);
	roots[tidx+64] 	<<=	32;
	roots[tidx+64] 	+=	tex1Dfetch(tex_roots_16k, ((bidx*2+1)*tidx)<<3);
	//load 4 or 8 samples from global memory and compute ntt of 8 samples
#pragma unroll
	for(int i=0; i<4; i++){
		if ((((chunksize*chunkid)>>5)+1) < w32) {
			samples[i] = src[((i<<11)|from_mem)*w32+((chunksize*chunkid)>>5)+1];
			samples[i] <<= 32;
			samples[i] += src[((i<<11)|from_mem)*w32+((chunksize*chunkid)>>5)];
		}
		else
			samples[i] = src[((i<<11)|from_mem)*w32+((chunksize*chunkid)>>5)];
		samples[i] >>= ((chunksize*chunkid)&0x1f);
		samples[i] &= (0x1<<chunksize)-1;
	}
	_ntt8_ext(samples);
#pragma unroll
	for(int i=0; i<8; i++)
		buffer[(i<<3) | to_buffer]=_ls_modP(samples[i], (tidx>>3)*i*3);
	__syncthreads();
	//load 8 samples from shared memory in transposed way and compute ntt of 8 samples
#pragma unroll
	for(int i=0; i<8; i++)
		samples[i]=buffer[(i<<6) | from_buffer];
	_ntt8(samples);
#pragma unroll
	for(int i=0; i<8; i++)
		dst[to_mem | (i<<5)]=_mul_modP(samples[i], roots[((tidx&0x7)>>2<<6) | (tidx>>3) | (i<<3)]);
}

__global__ void ntt_1_32k_ext(uint64 *dst, uint32 *src){
	__shared__ uint64 buffer[512], roots[64];
	register uint64 samples[8];
	register uint32 from_mem, to_mem, from_buffer, to_buffer;
	from_mem	=	((tidx&0x38)<<6) | (bidx<<3) | (tidx&0x7);
	to_buffer	=	((tidx&0x38)<<3) | (tidx&0x7);
	from_buffer	=	tidx;
	to_mem		=	(bidx<<9) | from_buffer;
	roots[tidx] = tex1Dfetch(tex_roots_32k, ((bidx*tidx)<<4)+1);
	roots[tidx] <<= 32;
	roots[tidx] += tex1Dfetch(tex_roots_32k, (bidx*tidx)<<4);
#pragma unroll
	for(int i=0; i<4; i++) samples[i]=src[(i<<12)|from_mem];
	_ntt8_ext(samples);
#pragma unroll
	for(int i=0; i<8; i++) buffer[(i<<3) | to_buffer]=_ls_modP(samples[i], (tidx>>3)*i*3);
	__syncthreads();
#pragma unroll
	for(int i=0; i<8; i++) samples[i]=buffer[from_buffer | (i<<6)];
	_ntt8(samples);
#pragma unroll
	for(int i=0; i<8; i++) dst[to_mem | (i<<6)]=_mul_modP(samples[i], roots[(tidx>>3) | (i<<3)]);
}

__global__ void ntt_1_32k_ext_block(uint64 *dst, uint32 *src, int chunksize, int chunkid, int w32){
	__shared__ uint64 buffer[512], roots[64];
	register uint64 samples[8];
	register uint32 from_mem, to_mem, from_buffer, to_buffer;
	from_mem	=	((tidx&0x38)<<6) | (bidx<<3) | (tidx&0x7);
	to_buffer	=	((tidx&0x38)<<3) | (tidx&0x7);
	from_buffer	=	tidx;
	to_mem		=	(bidx<<9) | from_buffer;
	roots[tidx] = tex1Dfetch(tex_roots_32k, ((bidx*tidx)<<4)+1);
	roots[tidx] <<= 32;
	roots[tidx] += tex1Dfetch(tex_roots_32k, (bidx*tidx)<<4);
#pragma unroll
	for(int i=0; i<4; i++){
		if ((((chunksize*chunkid)>>5)+1) < w32) {
			samples[i] = src[((i<<12)|from_mem)*w32+((chunksize*chunkid)>>5)+1];
			samples[i] <<= 32;
			samples[i] += src[((i<<12)|from_mem)*w32+((chunksize*chunkid)>>5)];
		}
		else
			samples[i] = src[((i<<12)|from_mem)*w32+((chunksize*chunkid)>>5)];
		samples[i] >>= ((chunksize*chunkid)&0x1f);
		samples[i] &= (0x1<<chunksize)-1;
	}
	_ntt8_ext(samples);
#pragma unroll
	for(int i=0; i<8; i++) buffer[(i<<3) | to_buffer]=_ls_modP(samples[i], (tidx>>3)*i*3);
	__syncthreads();
#pragma unroll
	for(int i=0; i<8; i++) samples[i]=buffer[from_buffer | (i<<6)];
	_ntt8(samples);
#pragma unroll
	for(int i=0; i<8; i++) dst[to_mem | (i<<6)]=_mul_modP(samples[i], roots[(tidx>>3) | (i<<3)]);
}

__global__ void ntt_1_64k_ext(uint64 *dst, uint32 *src){
	__shared__ uint64 buffer[512], roots[64];
	register uint64 samples[8];
	register uint32 from_mem, to_mem, from_buffer, to_buffer;
	from_mem	=	((tidx&0x38)<<7) | (bidx<<3) | (tidx&0x7);
	to_buffer	=	((tidx&0x38)<<3) | (tidx&0x7);
	from_buffer	=	tidx;
	to_mem		=	((bidx&0x7E)<<9) | ((tidx&0x38)<<1) | ((bidx&0x1)<<3) | (tidx&0x7);
	roots[tidx] = tex1Dfetch(tex_roots_64k, (((bidx>>1)*tidx)<<5)+1);
	roots[tidx] <<= 32;
	roots[tidx] += tex1Dfetch(tex_roots_64k, ((bidx>>1)*tidx)<<5);
#pragma unroll
	for(int i=0; i<4; i++) samples[i]=src[(i<<13)|from_mem];
	_ntt8_ext(samples);
#pragma unroll
	for(int i=0; i<8; i++) buffer[(i<<3) | to_buffer]=_ls_modP(samples[i], ((tidx>>3)*i)*3);
	__syncthreads();
#pragma unroll
	for(int i=0; i<8; i++) samples[i]=buffer[from_buffer | (i<<6)];
	_ntt8(samples);
#pragma unroll
	for(int i=0; i<8; i++) dst[to_mem | (i<<7)]=_mul_modP(samples[i], roots[(tidx>>3) | (i<<3)]);
}

__global__ void ntt_1_64k_ext_block(uint64 *dst, uint32 *src, int chunksize, int chunkid, int w32){
	__shared__ uint64 buffer[512], roots[64];
	register uint64 samples[8];
	register uint32 from_mem, to_mem, from_buffer, to_buffer;
	from_mem	=	((tidx&0x38)<<7) | (bidx<<3) | (tidx&0x7);
	to_buffer	=	((tidx&0x38)<<3) | (tidx&0x7);
	from_buffer	=	tidx;
	to_mem		=	((bidx&0x7E)<<9) | ((tidx&0x38)<<1) | ((bidx&0x1)<<3) | (tidx&0x7);
	roots[tidx] = tex1Dfetch(tex_roots_64k, (((bidx>>1)*tidx)<<5)+1);
	roots[tidx] <<= 32;
	roots[tidx] += tex1Dfetch(tex_roots_64k, ((bidx>>1)*tidx)<<5);
#pragma unroll
	for(int i=0; i<4; i++){
		if ((((chunksize*chunkid)>>5)+1) < w32) {
			samples[i] = src[((i<<13)|from_mem)*w32+((chunksize*chunkid)>>5)+1];
			samples[i] <<= 32;
			samples[i] += src[((i<<13)|from_mem)*w32+((chunksize*chunkid)>>5)];
		}
		else
			samples[i] = src[((i<<13)|from_mem)*w32+((chunksize*chunkid)>>5)];
		samples[i] >>= ((chunksize*chunkid)&0x1f);
		samples[i] &= (0x1<<chunksize)-1;
	}
	_ntt8_ext(samples);
#pragma unroll
	for(int i=0; i<8; i++) buffer[(i<<3) | to_buffer]=_ls_modP(samples[i], ((tidx>>3)*i)*3);
	__syncthreads();
#pragma unroll
	for(int i=0; i<8; i++) samples[i]=buffer[from_buffer | (i<<6)];
	_ntt8(samples);
#pragma unroll
	for(int i=0; i<8; i++) dst[to_mem | (i<<7)]=_mul_modP(samples[i], roots[(tidx>>3) | (i<<3)]);
}
//// NTT kernel b: ntt64 + roots(all)
// 16384
__global__ void ntt_2_16k(uint64 *src){
	__shared__ uint64 buffer[512];
	register uint64 samples[8];
	register uint32 from_mem, to_mem, from_buffer, to_buffer, addrx, addry;
	from_mem		=	(tidx>>3<<8) | (bidx<<3) | ((tidx&0x7)>>2<<2) | (tidx&0x3);
	to_buffer		=	((tidx&0x38)<<3) | (tidx&0x7);
	from_buffer		=	tidx;
	to_mem			=	from_mem;
	addrx			=	to_mem&0x3;
	addry			=	to_mem>>2;
#pragma unroll
	for(int i=0; i<8; i++) samples[i]=src[(i<<11)|from_mem];
	_ntt8(samples);
#pragma unroll
	for(int i=0; i<8; i++) buffer[(i<<3) | to_buffer]=_ls_modP(samples[i], ((tidx>>3)*i)*3);
	__syncthreads();
#pragma unroll
	for(int i=0; i<8; i++) samples[i]=buffer[(i<<6) | from_buffer];
	_ntt8(samples);
	register uint64 root;
#pragma unroll
	for(int i=0; i<8; i++){
		root = tex1Dfetch(tex_roots_16k, ((addry | (i<<9))*addrx*2)+1);
		root <<= 32;
		root += tex1Dfetch(tex_roots_16k, (addry | (i<<9))*addrx*2);
		src[(i<<11) | to_mem]=_mul_modP(samples[i], root);
	}
}

__global__ void ntt_2_32k(uint64 *src){
	__shared__ uint64 buffer[512];
	register uint64 samples[8];
	register uint32 from_mem, to_mem, from_buffer, to_buffer, addrx, addry;
	from_mem		=	((tidx&0x38)<<6) | (bidx<<3) | (tidx&0x7);
	to_buffer		=	((tidx&0x38)<<3) | (tidx&0x7);
	from_buffer		=	tidx;
	to_mem			=	((tidx&0x38)<<6) | (bidx<<3) | (tidx&0x7);
	addrx			=	(tidx&0x7);
	addry			=	((tidx&0x38)<<3) | bidx;
#pragma unroll
	for(int i=0; i<8; i++) samples[i]=src[(i<<12)|from_mem];
	_ntt8(samples);
#pragma unroll
	for(int i=0; i<8; i++) buffer[(i<<3) | to_buffer]=_ls_modP(samples[i], ((tidx>>3)*i)*3);
	__syncthreads();
#pragma unroll
	for(int i=0; i<8; i++) samples[i]=buffer[(i<<6) | from_buffer];
	_ntt8(samples);
	register uint64 root;
#pragma unroll
	for(int i=0; i<8; i++){
		root = tex1Dfetch(tex_roots_32k, ((addry | (i<<9))*addrx*2)+1);
		root <<= 32;
		root += tex1Dfetch(tex_roots_32k, (addry | (i<<9))*addrx*2);
		src[(i<<12) | to_mem]=_mul_modP(samples[i], root);
	}
}

__global__ void ntt_2_64k(uint64 *src){
	__shared__ uint64 buffer[512];
	register uint64 samples[8];
	register uint32 from_mem, to_mem, from_buffer, to_buffer, addrx, addry;
	from_mem		=	((tidx&0x38)<<7) | (bidx<<3) | (tidx&0x7);
	to_buffer		=	(tidx&0x38)<<3 | (tidx&0x7);
	from_buffer		=	tidx;
	to_mem			=	from_mem;
	addrx			=	(tidx&0x7) | ((bidx&0x1)<<3);
	addry			=	((tidx&0x38)<<3) | (bidx>>1);
#pragma unroll
	for(int i=0; i<8; i++) samples[i]=src[(i<<13)|from_mem];
	_ntt8(samples);
#pragma unroll
	for(int i=0; i<8; i++) buffer[(i<<3) | to_buffer]=_ls_modP(samples[i], (tidx>>3)*i*3);
	__syncthreads();
#pragma unroll
	for(int i=0; i<8; i++) samples[i]=buffer[(i<<6) | from_buffer];
	_ntt8(samples);
	register uint64 root;
#pragma unroll
	for(int i=0; i<8; i++){
		root = tex1Dfetch(tex_roots_64k, ((addry | (i<<9))*addrx*2)+1);
		root <<= 32;
		root += tex1Dfetch(tex_roots_64k, (addry | (i<<9))*addrx*2);
		src[(i<<13) | to_mem]=_mul_modP(samples[i], root);
	}
}
//// NTT kernel c: 2*_ntt4 or 1*ntt8 or 1/2*ntt16
// 16384
__global__ void ntt_3_16k(uint64 *dst, uint64 *src){
	__shared__ uint64 buffer[512];
	register uint64 samples[8];
	register uint32 from_mem, to_mem, from_buffer, to_buffer;
	from_mem	=	(bidx<<9) | tidx;
	to_buffer	=	tidx;
	from_buffer	=	tidx<<3;
	to_mem		=	(bidx<<7) | (tidx<<1);
#pragma unroll
	for(int i=0; i<8; i++) buffer[(i<<6) | to_buffer]=src[(i<<6) | from_mem];
	__syncthreads();
#pragma unroll
	for(int i=0; i<8; i++) samples[i]=buffer[from_buffer | i];
	_ntt4(samples);
	_ntt4(samples+4);
#pragma unroll
	for(int i=0; i<8; i++) dst[((i&0x3)<<12) | ((i&0x7)>>2) | to_mem]=samples[i];
}

__global__ void ntt_3_32k(uint64 *dst, uint64 *src){
	__shared__ uint64 buffer[512];
	register uint64 samples[8];
	register uint32 from_mem, to_mem, from_buffer, to_buffer;
	from_mem	=	(bidx<<9) | tidx;
	to_buffer	=	tidx;
	from_buffer	=	tidx<<3;
	to_mem		=	(bidx<<6) | tidx;
#pragma unroll
	for(int i=0; i<8; i++) buffer[(i<<6) | to_buffer]=src[(i<<6)|from_mem];
	__syncthreads();
#pragma unroll
	for(int i=0; i<8; i++) samples[i]=buffer[from_buffer | i];
	_ntt8(samples);
#pragma unroll
	for(int i=0; i<8; i++) dst[(i<<12) | to_mem]=samples[i];
}

__global__ void ntt_3_64k(uint64 *dst, uint64 *src){
	__shared__ uint64 buffer[512];
	register uint64 samples[8];
	register uint32 from_mem, to_mem, from_buffer, to_buffer;
	from_mem	=	(bidx<<9)|((tidx&0x3E)<<3)|(tidx&0x1);
	to_buffer	=	tidx<<3;
	from_buffer	=	((tidx&0x38)<<3) | (tidx&0x7);
	to_mem		=	(bidx<<9)|((tidx&0x38)<<3) | (tidx&0x7);
#pragma unroll
	for(int i=0; i<8; i++) samples[i]=src[from_mem | (i<<1)];
	_ntt8(samples);
#pragma unroll
	for(int i=0; i<8; i++) buffer[i | to_buffer]=_ls_modP(samples[i], ((tidx&0x1)<<2)*i*3);
	__syncthreads();
#pragma unroll
	for(int i=0; i<8; i++) samples[i]=buffer[from_buffer | (i<<3)];
	register uint32 addr;
#pragma unroll
	for(int i=0; i<4; i++){
		addr=to_mem | ((2*i)<<3);
		dst[((addr&0xf)<<12) | (addr>>4)]=_add_modP(samples[2*i], samples[2*i+1]);
		addr=to_mem | ((2*i+1)<<3);
		dst[((addr&0xf)<<12) | (addr>>4)]=_sub_modP(samples[2*i], samples[2*i+1]);
	}
}

//// INTT kernel a: ntt64 + roots(4096)
// 16384
__global__ void intt_1_16k(uint64 *dst, uint64 *src){
	__shared__ uint64 buffer[512], roots[128];
	register uint64 samples[8];
	register uint32 from_mem, to_mem, from_buffer, to_buffer;
	from_mem	=	(tidx>>3<<8) | (bidx<<3) | ((tidx&0x7)>>2<<2) | (tidx&0x3);
	to_buffer	=	((tidx&0x38)<<3) | (tidx&0x7);
	from_buffer	=	tidx;
	to_mem		=	(bidx<<9) | ((tidx&0x7)>>2<<8) | (tidx>>3<<2) | (tidx&0x3);
	roots[tidx] 		=	tex1Dfetch(tex_roots_16k, (((bidx*2)*tidx)<<3)+1);
	roots[tidx] 		<<=	32;
	roots[tidx] 		+=	tex1Dfetch(tex_roots_16k, ((bidx*2)*tidx)<<3);
	roots[tidx+64] 	=	tex1Dfetch(tex_roots_16k, (((bidx*2+1)*tidx)<<3)+1);
	roots[tidx+64] 	<<=	32;
	roots[tidx+64] 	+=	tex1Dfetch(tex_roots_16k, ((bidx*2+1)*tidx)<<3);
#pragma unroll
	for(int i=0; i<8; i++) samples[i]=src[(16384-((i<<11)|from_mem))%16384];
	_ntt8(samples);
#pragma unroll
	for(int i=0; i<8; i++) buffer[(i<<3) | to_buffer]=_ls_modP(samples[i], (tidx>>3)*i*3);
	__syncthreads();
#pragma unroll
	for(int i=0; i<8; i++) samples[i]=buffer[(i<<6) | from_buffer];
	_ntt8(samples);
#pragma unroll
	for(int i=0; i<8; i++) dst[to_mem | (i<<5)]=_mul_modP(samples[i], roots[((tidx&0x7)>>2<<6) | (tidx>>3) | (i<<3)]);
}
__global__ void intt_1_32k(uint64 *dst, uint64 *src){
	__shared__ uint64 buffer[512], roots[64];
	register uint64 samples[8];
	register uint32 from_mem, to_mem, from_buffer, to_buffer;
	from_mem	=	((tidx&0x38)<<6) | (bidx<<3) | (tidx&0x7);
	to_buffer	=	(tidx&0x38)<<3 | (tidx&0x7);
	from_buffer	=	tidx;
	to_mem		=	(bidx<<9) | from_buffer;
	roots[tidx] = tex1Dfetch(tex_roots_32k, ((bidx*tidx)<<4)+1);
	roots[tidx] <<= 32;
	roots[tidx] += tex1Dfetch(tex_roots_32k, (bidx*tidx)<<4);
#pragma unroll
	for(int i=0; i<8; i++) samples[i]=src[(32768-((i<<12)|from_mem))%32768];
	_ntt8(samples);
#pragma unroll
	for(int i=0; i<8; i++) buffer[(i<<3) | to_buffer]=_ls_modP(samples[i], ((tidx>>3)*i)*3);
	__syncthreads();
#pragma unroll
	for(int i=0; i<8; i++) samples[i]=buffer[from_buffer | (i<<6)];
	_ntt8(samples);
#pragma unroll
	for(int i=0; i<8; i++) dst[to_mem | (i<<6)]=_mul_modP(samples[i], roots[(tidx>>3) | (i<<3)]);
}
__global__ void intt_1_64k(uint64 *dst, uint64 *src){
	__shared__ uint64 buffer[512], roots[64];
	register uint64 samples[8];
	register uint32 from_mem, to_mem, from_buffer, to_buffer;
	from_mem	=	((tidx&0x38)<<7) | (bidx<<3) | (tidx&0x7);
	to_buffer	=	(tidx&0x38)<<3 | (tidx&0x7);
	from_buffer	=	tidx;
	to_mem		=	((bidx&0x7E)<<9) | ((tidx&0x38)<<1) | ((bidx&0x1)<<3) | (tidx&0x7);
	roots[tidx] = tex1Dfetch(tex_roots_64k, (((bidx>>1)*tidx)<<5)+1);
	roots[tidx] <<= 32;
	roots[tidx] += tex1Dfetch(tex_roots_64k, ((bidx>>1)*tidx)<<5);
#pragma unroll
	for(uint32 i=0; i<8; i++) samples[i]=src[(65536-(from_mem | (i<<13)))%65536];
	_ntt8(samples);
#pragma unroll
	for(int i=0; i<8; i++) buffer[(i<<3) | to_buffer]=_ls_modP(samples[i], ((tidx>>3)*i)*3);
	__syncthreads();
#pragma unroll
	for(int i=0; i<8; i++) samples[i]=buffer[from_buffer | (i<<6)];
	_ntt8(samples);
#pragma unroll
	for(int i=0; i<8; i++) dst[to_mem | (i<<7)]=_mul_modP(samples[i], roots[(tidx>>3) | (i<<3)]);
}

//// INTT kernel c: ntt64 + roots(4096)
__global__ void intt_3_16k_modcrt(uint32 *dst, uint64 *src, int crtidx) {
	__shared__ uint64 buffer[512];
	register uint64 samples[8];
	register uint32 from_mem, to_mem, from_buffer, to_buffer;
	from_mem	=	(bidx<<9) | tidx;
	to_buffer	=	tidx;
	from_buffer	=	tidx<<3;
	to_mem		=	(bidx<<7) | (tidx<<1);
#pragma unroll
	for(int i=0; i<8; i++) buffer[(i<<6) | to_buffer]=src[(i<<6) | from_mem];
	__syncthreads();
#pragma unroll
	for(int i=0; i<8; i++) samples[i]=buffer[from_buffer | i];
	_ntt4(samples);
	_ntt4(samples+4);
#pragma unroll
	for(int i=0; i<8; i++) dst[((i&0x3)<<12) | ((i&0x7)>>2) | to_mem]=(uint32)(_mul_modP(samples[i], 18445618169508003841)%const_p[crtidx]);
}

__global__ void intt_3_32k_modcrt(uint32 *dst, uint64 *src, int crtidx){
	__shared__ uint64 buffer[512];
	register uint64 samples[8];
	register uint32 from_mem, to_mem, from_buffer, to_buffer;
	from_mem	=	(bidx<<9) | tidx;
	to_buffer	=	tidx;
	from_buffer	=	tidx<<3;
	to_mem		=	(bidx<<6) | tidx;
#pragma unroll
	for(int i=0; i<8; i++) buffer[(i<<6) | to_buffer]=src[(i<<6)|from_mem];
	__syncthreads();
#pragma unroll
	for(int i=0; i<8; i++) samples[i]=buffer[from_buffer | i];
	_ntt8(samples);
#pragma unroll
	for(int i=0; i<8; i++) dst[(i<<12) | to_mem]=(uint32)(_mul_modP(samples[i], 18446181119461294081)%const_p[crtidx]);
}

__global__ void intt_3_64k_modcrt(uint32 *dst, uint64 *src, int crtidx){
	__shared__ uint64 buffer[512];
	register uint64 samples[8], s8[8];
	register uint32 from_mem, to_mem, from_buffer, to_buffer;
	from_mem	=	(bidx<<9)|((tidx&0x3E)<<3)|(tidx&0x1);
	to_buffer	=	tidx<<3;
	from_buffer	=	((tidx&0x38)<<3) | (tidx&0x7);
	to_mem		=	(bidx<<9)|((tidx&0x38)<<3) | (tidx&0x7);
#pragma unroll
	for(int i=0; i<8; i++) samples[i]=src[from_mem | (i<<1)];
	_ntt8(samples);
#pragma unroll
	for(int i=0; i<8; i++) buffer[to_buffer | i]=_ls_modP(samples[i], ((tidx&0x1)<<2)*i*3);
	__syncthreads();
#pragma unroll
	for(int i=0; i<8; i++) samples[i]=buffer[from_buffer | (i<<3)];
#pragma unroll
	for(int i=0; i<4; i++){
		s8[2*i]=_add_modP(samples[2*i], samples[2*i+1]);
		s8[2*i+1]=_sub_modP(samples[2*i], samples[2*i+1]);
	}
#pragma unroll
	for(int i=0; i<8; i++) dst[(((to_mem | (i<<3))&0xf)<<12) | ((to_mem | (i<<3))>>4)]=(uint32)(_mul_modP(s8[i], 18446462594437939201)%const_p[crtidx]);
}
/* end NTT kernels	*/

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
