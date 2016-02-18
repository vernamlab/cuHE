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

// Precompute device data and tranfer to GPUs:
//    NTT twiddle factors,
//    CRT prime numbers, etc.,
//    Barrett Reduction polynomial modulus in ntt/crt domain.
// Declare CUDA kernels that need those data:
//    NTT/INTT conversion,
//    CRT/ICRT conversion,
//    Barrett Reduction,
//    Relinearization,
//    NTT domain arithmetic,
//    CRT domain arithmetic,
//    Modulus Switching.

#pragma once
typedef	unsigned int		uint32;	// 32-bit
typedef unsigned long int	uint64;	// 64-bit

namespace cuHE {

/////////////////////////////////////////////////////
//// Transfer Pre-computed Data to GPU Device(s) ////
/////////////////////////////////////////////////////
// generate & copy twiddle factors to device, bind to texture
// "len": length of NTT (a power of 2)
void preload_ntt(int len);
// free and delete allocated memory space
void cleanup_ntt();

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

/////////////////////////////////////////////////////////////////////
//// CUDA Kernel Functions That Require Pre-computed Device Data ////
/////////////////////////////////////////////////////////////////////
// NTT kernels with size: 16384, 32768 or 65536.
// Each NTT/invrese-NTT conversion consists of 3 kernels:
//   - NTT:
//      ntt_1 -> ntt_2 ->  ntt_3
//   - inverse-INTT:
//     intt_1 -> ntt_2 -> intt_3.
// NTT conversion either takes an array of 32-bit integers, or takes a chunck
// of each elements in an array of large integers as input.
// Inverse-NTT conversion gives an array of 32-bit integers (modulo CRT prime
// numbers) as output.
// NTT 16384 kernels
__global__ void ntt_1_16k_ext(uint64 *dst, uint32 *src);
__global__ void ntt_1_16k_ext_block(uint64 *dst, uint32 *src, int chunksize,
    int chunkid, int w32);
__global__ void ntt_2_16k(uint64 *src);
__global__ void ntt_3_16k(uint64 *dst, uint64 *src);
__global__ void intt_1_16k(uint64 *dst, uint64 *src);
__global__ void intt_3_16k_modcrt(uint32 *dst, uint64 *src, int crtidx);
// NTT 16384 kernels
__global__ void ntt_1_32k_ext(uint64 *dst, uint32 *src);
__global__ void ntt_1_32k_ext_block(uint64 *dst, uint32 *src, int chunksize,
    int chunkid, int w32);
__global__ void ntt_2_32k(uint64 *src);
__global__ void ntt_3_32k(uint64 *dst, uint64 *src);
__global__ void intt_1_32k(uint64 *dst, uint64 *src);
__global__ void intt_3_32k_modcrt(uint32 *dst, uint64 *src, int crtidx);
// NTT 16384 kernels
__global__ void ntt_1_64k_ext(uint64 *dst, uint32 *src);
__global__ void ntt_1_64k_ext_block(uint64 *dst, uint32 *src, int chunksize,
    int chunkid, int w32);
__global__ void ntt_2_64k(uint64 *src);
__global__ void ntt_3_64k(uint64 *dst, uint64 *src);
__global__ void intt_1_64k(uint64 *dst, uint64 *src);
__global__ void intt_3_64k_modcrt(uint32 *dst, uint64 *src, int crtidx);

// CRT conversion kernel
__global__ void crt(uint32 *dst, uint32 *src, int pnum, int w32,
    int mlen, int clen);
// Inverse-CRT conversion kernel
__global__ void icrt(uint32 *dst, uint32 *src, int pnum, int M_w32,
		int mi_w32, int mlen, int clen);

// Barrett Reduction kernels (all 5 of them are needed for one reduction)
__global__ void barrett_mul_un(uint64 *tar, int pnum, int nlen);
__global__ void barrett_mul_mn(uint64 *tar, int pnum, int nlen);
__global__ void barrett_sub_1(uint32 *y, uint32 *x, int pnum, int mlen,
    int nlen);
__global__ void barrett_sub_2(uint32 *y, uint32 *x, int pnum, int nlen);
__global__ void barrett_sub_mc(uint32 *x, int pnum, int mlen, int clen,
    int nlen);
// Relinearization kernels: offer two modes
// Large device memory & small key size: ek[knum][pnum][NTTLEN]
__global__ void relinMulAddAll(uint64 *dst, uint64 *c, uint64 *ek, int pnum,
    int knum, int nlen);
// Small device memory & large key size: pnum * ek[knum][NTTLEN]
__global__ void relinMulAddPerCrt(uint64 *dst, uint64 *c, uint64 *ek, int knum,
    int nlen);
// NTT domain arithmetic
__global__ void ntt_mul(uint64 *z, uint64 *x, uint64 *y, int pnum, int nlen);
__global__ void ntt_add(uint64 *z, uint64 *x, uint64 *y, int pnum, int nlen);
__global__ void ntt_mul_nx1(uint64 *z, uint64 *x, uint64 *scalar, int pnum,
    int nlen);
__global__ void ntt_add_nx1(uint64 *z, uint64 *x, uint64 *scalar, int pnum,
    int nlen);
// CRT domain arithmetic
__global__ void crt_mul_int(uint32 *y, uint32 *x, int a, int pnum, int clen);
__global__ void crt_add(uint32 *x, uint32 *a, uint32 *b, int pnum, int mlen,
    int clen);
__global__ void crt_add_int(uint32 *y, uint32 *x, int a, int pnum, int clen);
__global__ void crt_add_nx1(uint32 *y, uint32 *x, uint32 *scalar, int pnum,
    int mlen, int clen);
// Modulus Switching
__global__ void modswitch(uint32 *dst, uint32 *src, int pnum, int mlen,
    int clen, int modmsg);

} // end cuHE
