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

/*!	/file CuHE.h
 *	/brief Provides an API for cuHE library.
 	Initialization steps.
 	Defines polynomials on GPUs.
 */

#pragma once

#include "Relinearization.h"
#include "Parameters.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <NTL/ZZ.h>
#include <NTL/ZZX.h>
NTL_CLIENT

typedef unsigned char		uint8;	// 8-bit
typedef	unsigned int		uint32;	// 32-bit
typedef unsigned long int	uint64;	// 64-bit

namespace cuHE {

///////////////////////////////////////////////////////////////////////////////
//// @class CuPolynomial //////////////////////////////////////////////////////
class CuPolynomial {
// Defines an abstract polynomial on GPU.
// It has no knowledge of circuits or levels.
public:
	//-- Constructor -------------------------------------------
	CuPolynomial();
	~CuPolynomial();
	void reset();
	
	//-- Set Methods -------------------------------------------
	void set(int logq, int dom, int dev, cudaStream_t st = 0);
	// initialize witout value

	void set(int logq, int dev, ZZX val);
	// initialize with ZZX domain value

	void logq(int val); // logq_ = val
	void domain(int val); // domain_ = val
	void device(int val); // device_ = val
	void isProd(bool val); // isProd_ = val
	void zRep(ZZX val); // zRep_ = val
	void rRep(uint32* val); // rRep_ = val (pointer copy)
	void cRep(uint32* val); // cRep_ = val (pointer copy)
	void nRep(uint64* val); // nRep_ = val (pointer copy)

	//-- Get Methods -------------------------------------------
	int		logq(void);
	int		domain(void);
	int		device(void);
	bool	isProd(void);
	ZZX		zRep(void);
	uint32* rRep(void);
	uint32* cRep(void);
	uint64* nRep(void);

	//-- Domain Conversions ------------------------------------
	void x2z(cudaStream_t st = 0); // any -> ZZX
	void x2r(cudaStream_t st = 0); // any -> RAW
	void x2c(cudaStream_t st = 0); // any -> CRT
	void x2n(cudaStream_t st = 0); // any -> NTT

	//-- Memory management -------------------------------------
	void rRepCreate(cudaStream_t st = 0);
	void cRepCreate(cudaStream_t st = 0);
	void nRepCreate(cudaStream_t st = 0);
	void rRepFree();
	void cRepFree();
	void nRepFree();
	
	// Utilities -----------------------------------------------
	int coeffWords();
	size_t rRepSize();
	virtual size_t cRepSize() = 0;
	virtual size_t nRepSize() = 0;

protected:
	//-- Domain Conversions ------------------------------------
	void z2r(cudaStream_t st = 0); // ZZX -> RAW
	void r2z(cudaStream_t st = 0); // RAW -> ZZX
	void r2c(cudaStream_t st = 0); // CRT
	void c2r(cudaStream_t st = 0); // ICRT
	void c2n(cudaStream_t st = 0); // NTT
	void n2c(cudaStream_t st = 0); // INTT (with and without barrett)

	//-- Properties and Values ---------------------------------
	int logq_;		// bit-length of coefficient modulus
	int domain_;	// <- {zzx(0), raw(1), crt(2), ntt(3)}
	int device_;	// assigned CUDA device
	bool isProd_;	// true if any multiplication occurs
	ZZX zRep_;		// host ZZX type value
	uint32 *rRep_;	// device pointer to raw value
	uint32 *cRep_;	// device pointer to crt value
	uint64 *nRep_;	// device pointer to ntt value

}; // end CuPolynomial


///////////////////////////////////////////////////////////////////////////////
//// @class CuCtxt ////////////////////////////////////////////////////////////
class CuCtxt: public CuPolynomial {
// Defines a ciphertext on GPU.
// It includes knowledge of circuits or levels.
// Temporarily it needs more than enough memory size.
public:
	//-- Get Methods -------------------------------------------
	int level();
	//-- Noise Control -----------------------------------------
	void modSwitch(cudaStream_t st = 0);
	// modulus switching by one level

	void modSwitch(int lvl, cudaStream_t st = 0);
	// modulus switching repeatedly to a level

	void relin(cudaStream_t st = 0);
	// relinearization

	size_t cRepSize();
	size_t nRepSize();
	///void relin(cudaStream_t st = 0);
	// relinearization (not yet ready)
}; // end CuCtxt

///////////////////////////////////////////////////////////////////////////////
//// @class CuPtxt ////////////////////////////////////////////////////////////
class CuPtxt: public CuPolynomial {
// Defines a plaintext on GPU.
// Use this class when a plaintext has batched format, i.e. a polynomial.
// Otherwise a plaintext is an integer.
public:
	size_t cRepSize();
	size_t nRepSize();
}; // end CuPtxt

///////////////////////////////////////////////////////////////////////////////
//// Initailization ///////////////////////////////////////////////////////////
void initCuHE(ZZ *coeffMod_, ZZX modulus);
// Input: polynomial modulus;
// Process: enables CUDA functionalities, pre-compute device data;
// Output: coefficient moduli.

void startAllocator();
// Start before circuit evaluation to use an customized CUDA device
// memory allocator. Only use it for online allocations.

void stopAllocator();
// Stop after circuit evaluation.
// (Customized allocator improves performance, especially for multi-GPUs.)

void multiGPUs(int num);
// Set the number of GPUs to use.

int numGPUs();
// Return the number of GPUs in use.

void setParameters(int d, int p, int w, int min, int cut, int m);
//	Set new parameters with:
//	d -- maximum depth of the circuit;
//	p -- message modulus;
//	w -- bit-length of windowed relinearition;
//	min -- bit-length of the smallest coefficient modulus;
//	cut -- coefficient cutting size in bits (per level);
//	m -- degree of the cyclic polynomial.

void resetParameters();
// Reset parameters to zeros
//	and delete global parameter.

///////////////////////////////////////////////////////////////////////////////
//// NTL Interface ////////////////////////////////////////////////////////////
void mulZZX(ZZX& x, ZZX a, ZZX b, int lvl, int dev, cudaStream_t st = 0);
// x = a*b
// assume that in0 and in1 are as large as ciphertexts
// lvl is the circuit level, dev and st are the chosen CUDA device and stream

///////////////////////////////////////////////////////////////////////////////
//// Operations: CuCtxt & CuPtxt //////////////////////////////////////////////
void copy(CuCtxt& x, CuCtxt a, cudaStream_t st = 0);
// x = a
// values are copied with memory copy

void cAnd(CuCtxt& x, CuCtxt& a, CuCtxt& b, cudaStream_t st = 0);
// x = a * b

void cAnd(CuCtxt& x, CuCtxt& c, CuPtxt& p, cudaStream_t st = 0);
// x = c * p

void cXor(CuCtxt& x, CuCtxt& a, CuCtxt& b, cudaStream_t st = 0);
// x = a + b

void cXor(CuCtxt& x, CuCtxt& c, CuPtxt& p, cudaStream_t st = 0);
// x = a + b

void cNot(CuCtxt& x, CuCtxt& a, cudaStream_t st = 0);
// x = a + modMsg - 1 = !a
// modMsg is the message space

void moveTo(CuCtxt& x, int dstDev, cudaStream_t st = 0);
// move x to destination device (dstDev)
// remove x from the source device
// !!! stream must be one on the source device

void copyTo(CuCtxt& dst, CuCtxt& src, int dstDev, cudaStream_t st = 0);
// dst becomes a copy of src execpt dst in on device dstDev
// src remains unchanged
// !!! stream must be one on the source device

///////////////////////////////////////////////////////////////////////////////
//// Miscellaneous ////////////////////////////////////////////////////////////

} // end cuHE
