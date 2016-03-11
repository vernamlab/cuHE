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

/*!	/file DHS.h
 *	/brief Homomorphic encryption scheme (DHS as an example).
 */

#pragma once

#include <NTL/ZZ.h>
#include <NTL/ZZX.h>
#include <NTL/mat_ZZ.h>
#include <NTL/mat_ZZ_p.h>
#include <NTL/ZZ_pE.h>
#include <NTL/ZZ_pX.h>
#include <NTL/ZZ_pXFactoring.h>
NTL_CLIENT

///////////////////////////////////////////////////////////////////////////////
/** @class Batcher */
/*!	/brief Batching technique.	*/
class Batcher{
public:
	Batcher(ZZX polymod, int f_degree, int f_size);
	~Batcher();
	void SetModulus(ZZX m);
	void ComputeFactors(int f_degree, int f_size);
	void CalculateMs();
	void CalculateNs();
	void CalculateMxNs();

	void encode(ZZX &poly, ZZX mess);
	void decode(ZZX &mess, ZZX poly);

	ZZX num2ZZX(int num);
private:
	vec_ZZ_pX M;
	vec_ZZ_pX N;
	vec_ZZ_pX MxN;
	vec_ZZ_pX factors;
	ZZ_pX modulus;
	int size;
};

///////////////////////////////////////////////////////////////////////////////
/** @class DHS */
/*!	/brief Somewhat homomorphic encryption scheme by Doroz, Hu and Sunar.	*/
class CuDHS {
public:
	//// Constructor //////////////////////////////////////
	CuDHS(int d, int p, int w, int min, int cut, int m);
	~CuDHS();

	//// Get methods //////////////////////////////////////
	ZZX polyMod();
	ZZ* coeffMod();
	int numSlot();
	ZZX* ek();
	//// Primitives ///////////////////////////////////////
	void keyGen();

	void encrypt(ZZX& out, ZZX in, int lvl);

	void decrypt(ZZX& out, ZZX in, int lvl, int maxMulPath = 1);
	// if no relinearization is performed, maxMulPath should be
	// the maximum number of multipliers included in a ciphertext

	void balance(ZZX& x, int lvl);

	void unbalance(ZZX& x, int lvl);

	void coeffReduce(ZZX& out, ZZX in, int lvl);

	Batcher *batcher;
protected:
	int B; // bound of distribution to sample polynomials from
	ZZX polyMod_; // x^n-1 usually
	ZZ *coeffMod_; // a decreasing sequence of odd moduli
	ZZX *pk_; // public key
	ZZX *sk_; // secret key
	ZZX *ek_; // evaluation key, not always
	int numSlot_;
	//// Tools ////////////////////////////////////////////
	ZZX sample();
	void findInverse(ZZX& f_inv, ZZX& f, ZZ& q, bool& isfound);
	int mobuisFunction(int n);

	void coeffReduce(ZZX& out, ZZX in, ZZ q);
	void genPkSk();
	void genEk();
	void genPolyMod_();
	int factorDegree();
}; // end DHS
