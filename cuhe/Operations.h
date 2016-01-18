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

/*!	/file Operations.h
 *	/brief Combines NTT, CRT, Barrett reduction.
 *	Pre-computation and operations.
 */

#pragma once

#include <NTL/ZZ.h>
#include <NTL/ZZX.h>
NTL_CLIENT

typedef	unsigned int		uint32;	// 32-bit
typedef unsigned long int	uint64;	// 64-bit

namespace cuHE {

///////////////////////////////////////////////////////////////////////////////
//// Pre-computation //////////////////////////////////////////////////////////
void initCrt(ZZ* coeffModulus);
// generate CRT prime numbers
// compute all useful constant device data
// output computed coefficient moduli

void loadIcrtConst(int lvl, int dev, cudaStream_t st = 0);
// for a specific lvl, update ICRT device constant memory

void initNtt();
// generate twiddle factors
// allocate device memory for temporary results from NTT conversions

void initBarrett(ZZX m);
// compute crt&ntt of polynomial modulus m
// compute ntt of x^(2n-1)/m
// store on device and bind to texture memory
// allocate device memory for temporary results from Barrett reduction

///////////////////////////////////////////////////////////////////////////////
//// Get Buffers //////////////////////////////////////////////////////////////
uint32 *inttResult(int dev);
// return holded intt result pointer

///////////////////////////////////////////////////////////////////////////////
//// Operations ///////////////////////////////////////////////////////////////
void crt(uint32* dst, uint32* src, int logq, int dev, cudaStream_t st = 0);
// crt conversion

void icrt(uint32* dst, uint32* src, int logq, int dev, cudaStream_t st = 0);
// icrt conversion, update icrt constants

void crtAdd(uint32* sum, uint32* x, uint32* y, int logq, int dev, cudaStream_t st = 0);
// crt polynomial set addition

void crtAddInt(uint32* sum, uint32* x, unsigned a, int logq, int dev, cudaStream_t st = 0);
// crt polynomial set and integer addition

void crtAddNX1(uint32* sum, uint32* x, uint32* scalar, int logq, int dev, cudaStream_t st = 0);
// crt polynomial set and single polynomial addition

void crtModSwitch(uint32 *dst, uint32 *src, int logq, int dev, cudaStream_t st = 0);
// modulus switching in crt domain, with double-crt setup
//void crtMulInt(uint32 *prod, uint32 *x, int a, int logq, int dev, cudaStream_t st);
// crt polynomial set and integer multiplication

void _ntt(uint64 *X, uint32 *x, int dev, cudaStream_t st = 0);
void _nttw(uint64 *X, uint32 *x, int coeffwords, int relinIdx, int dev, cudaStream_t st = 0);
void _intt(uint32 *x, uint64 *X, int crtidx, int dev, cudaStream_t st = 0);
// NTT on one polynomial with small coefficient

void ntt(uint64 *X, uint32 *x, int logq, int dev, cudaStream_t st = 0);
void nttw(uint64 *X, uint32 *x, int logq, int dev, cudaStream_t st = 0);
void intt(uint32 *x, uint64 *X, int logq, int dev, cudaStream_t st = 0);
void inttHold(uint64 *X, int logq, int dev, cudaStream_t st = 0);
void inttDoubleDeg(uint32 *x, uint64 *X, int logq, int dev, cudaStream_t st = 0);
void inttMod(uint32 *x, uint64 *X, int logq, int dev, cudaStream_t st = 0);
// NTT on a set of crt polynomials

void nttMul(uint64 *z, uint64 *y, uint64 *x, int logq, int dev, cudaStream_t st = 0);
// NTT polynomial set multiplication

void nttMulNX1(uint64 *z, uint64 *x, uint64 *scalar, int logq, int dev, cudaStream_t st = 0);
// NTT polynomial set and NTT single polynomial multiplication

void nttAdd(uint64 *z, uint64 *y, uint64 *x, int logq, int dev, cudaStream_t st = 0);
// NTT polynomial set addition

void nttAddNX1(uint64 *z, uint64 *x, uint64 *scalar, int logq, int dev, cudaStream_t st = 0);
// NTT polynomial set and NTT single polynomial addition

void barrett(uint32 *dst, uint32 *src, int lvl, int dev, cudaStream_t st = 0);
void barrett(uint32 *dst, int lvl, int dev, cudaStream_t st = 0);

///////////////////////////////////////////////////////////////////////////////
//// Miscellaneous ////////////////////////////////////////////////////////////
void genCrtPrimes();
// generate all CRT prime numbers: = 1 (mod modMsg)
// for cutting levels, prime size is cutting size
// for the rest, as large as possible: prime < sqrt(P/n)
// adjust prime size according to coefficient moduli sizes

void genCoeffModuli();
// compute a decreasing sequence of coefficient moduli for polynomial rings

void genCrtInvPrimes();
// compute p_i^(-1) mod p_j, for modulus switching in all cutting levels

void genIcrtByLevel(int lvl);
// compute M=p_0*p_1*...*p_t (t=numCrtPrime(lvl)-1)
// compute m_i = M/p_j
// compute b_i = m_i^(-1)
// for a specific lvl

void genIcrt();
// genIcrtByLevel(all levels)

void getCoeffModuli(ZZ *dst);
// obtain all coefficient moduli (count = depth())

// not called externally
uint64 **ptrNttSwap();
uint32 **ptrNttHold();
uint64 *ptrNttSwap(int dev);
uint32 *ptrNttHold(int dev);
void createBarrettTemporySpace();
static uint32 *ptrBarrettCrt(int dev);
static uint64 *ptrBarrettNtt(int dev);
static uint32 *ptrBarrettSrc(int dev);
void setPolyModulus(ZZX m);
} // end cuHE
