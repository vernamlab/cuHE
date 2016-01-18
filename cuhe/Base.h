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

/*!	/file Base.h
 *	/brief Precompute device data on GPUs and define CUDA kernels that need those data.
 */

#pragma once

//#include "Configurations.h"
//#include "Parameters.h"
#include <NTL/ZZ.h>
#include <NTL/ZZX.h>
NTL_CLIENT
typedef	unsigned int		uint32;	// 32-bit
typedef unsigned long int	uint64;	// 64-bit

namespace cuHE {

///////////////////////////////////////////////////////////////////////////////
//// Pre-computated Data Tranfer //////////////////////////////////////////////
void preload_ntt(uint64* src, int len);
// copy twiddle factors from h_roots to device and bind to texture
void preload_crt_p(uint32* src, int words);
// copy all generated CRT prime numbers to device constant memory
void preload_crt_invp(uint32* src, int words);
// copy all p_i^(-1) mod p_j to device constant memory
void load_icrt_M(uint32* src, int words, int dev, cudaStream_t st = 0);
// copy all generated CRT prime numbers to device constant memory
void load_icrt_mi(uint32* src, int words, int dev, cudaStream_t st = 0);
// copy all generated CRT prime numbers to device constant memory
void load_icrt_bi(uint32* src, int words, int dev, cudaStream_t st = 0);
// copy all generated CRT prime numbers to device constant memory
void preload_barrett_u_n(uint64* src, size_t size);
// copy and bind ntt(crt(x^(2n-1)/m))
void preload_barrett_m_n(uint64* src, size_t size);
// copy and bind ntt(crt(m))
void preload_barrett_m_c(uint32* src, size_t size);
// copy and bind crtt(m)


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/**	crt kernels	*/
__global__ void crt(uint32 *dst, uint32 *src, int pnum, int w32,
				int mlen, int clen);
__global__ void icrt(uint32 *dst, uint32 *src, int pnum, int M_w32,
				int mi_w32, int mlen, int clen);

/**	ntt kernels	*/
__global__ void ntt_1_16k_ext(uint64 *dst, uint32 *src);
__global__ void ntt_1_16k_ext_block(uint64 *dst, uint32 *src, int chunksize, int chunkid, int w32);
__global__ void ntt_2_16k(uint64 *src);
__global__ void ntt_3_16k(uint64 *dst, uint64 *src);
__global__ void intt_1_16k(uint64 *dst, uint64 *src);
__global__ void intt_3_16k_modcrt(uint32 *dst, uint64 *src, int crtidx);

__global__ void ntt_1_32k_ext(uint64 *dst, uint32 *src);
__global__ void ntt_1_32k_ext_block(uint64 *dst, uint32 *src, int chunksize, int chunkid, int w32);
__global__ void ntt_2_32k(uint64 *src);
__global__ void ntt_3_32k(uint64 *dst, uint64 *src);
__global__ void intt_1_32k(uint64 *dst, uint64 *src);
__global__ void intt_3_32k_modcrt(uint32 *dst, uint64 *src, int crtidx);

__global__ void ntt_1_64k_ext(uint64 *dst, uint32 *src);
__global__ void ntt_1_64k_ext_block(uint64 *dst, uint32 *src, int chunksize, int chunkid, int w32);
__global__ void ntt_2_64k(uint64 *src);
__global__ void ntt_3_64k(uint64 *dst, uint64 *src);
__global__ void intt_1_64k(uint64 *dst, uint64 *src);
__global__ void intt_3_64k_modcrt(uint32 *dst, uint64 *src, int crtidx);

/** barrett kernels	*/
__global__ void barrett_mul_un(uint64 *tar, int pnum, int nlen);
__global__ void barrett_mul_mn(uint64 *tar, int pnum, int nlen);
__global__ void barrett_sub_1(uint32 *y, uint32 *x, int pnum, int mlen, int nlen);
__global__ void barrett_sub_2(uint32 *y, uint32 *x, int pnum, int nlen);
__global__ void barrett_sub_mc(uint32 *x, int pnum, int mlen, int clen, int nlen);

/**	relinearization kernels	*/
// ek[knum][pnum][NTTLEN]
__global__ void relinMulAddAll(uint64 *dst, uint64 *c, uint64 *ek, int pnum, int knum, int nlen);
// pnum * ek[knum][NTTLEN]
__global__ void relinMulAddPerCrt(uint64 *dst, uint64 *c, uint64 *ek, int knum, int nlen);

/** operations	*/
__global__ void ntt_mul(uint64 *z, uint64 *x, uint64 *y, int pnum, int nlen);
__global__ void ntt_add(uint64 *z, uint64 *x, uint64 *y, int pnum, int nlen);
__global__ void ntt_mul_nx1(uint64 *z, uint64 *x, uint64 *scalar, int pnum, int nlen);
__global__ void ntt_add_nx1(uint64 *z, uint64 *x, uint64 *scalar, int pnum, int nlen);

__global__ void crt_mul_int(uint32 *y, uint32 *x, int a, int pnum, int clen);
__global__ void crt_add(uint32 *x, uint32 *a, uint32 *b, int pnum, int mlen, int clen);
__global__ void crt_add_int(uint32 *y, uint32 *x, int a, int pnum, int clen);
__global__ void crt_add_nx1(uint32 *y, uint32 *x, uint32 *scalar, int pnum, int mlen, int clen);

__global__ void modswitch(uint32 *dst, uint32 *src, int pnum, int mlen, int clen, int modmsg);

} // end cuHE
