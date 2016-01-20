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

/*!	/file Parameters.h
 *	/brief	Set parameters according to the algorithm.
 *			Reset parameters when necessary.
 *			All files get parameters from the method here.
 */

#pragma once

namespace cuHE {

/** Parameters are initialized and reset
	by calling methods of this class. */
struct GlobalParameters{ // global Parameters
	// Ring
	int mSize; // M_SIZE
	int modLen; // degree of polynomial modulus
	int modLen2; // round modLen up to the minimum power of 2
	int rawLen; // length of a raw domain polynomial
	int crtLen; // length of a crt domain polynomial
	int nttLen; // length of a ntt domain polynomial
	int logCoeffMax; // size of maximum coeff modulus in bits
	int logCoeffMin; // size of minimum coeff modulus in bits
	int logCoeffCut; // cutting size of coeff moduli in bits
	// Circuit
	int depth; // # of multiplicative levels + 1
	int modMsg; // message modulus
	int logMsg; // message size in bits
	int wordsMsg; // message size in 32-bit words
	// Relinearization
	int logRelin; // number of bits in relinearization, no relin if 0
	int numEvalKey; // maximum number of evaluation keys
	// CuHE
	int logCrtPrime; // size of crt prime numbers in bits
	int numCrtPrime; // number of crt prime numbers in total
	// For a specific level
	int _numCrtPrime(int lvl);
	int _logCoeff(int lvl);
	int _wordsCoeff(int lvl);
	int _numEvalKey(int lvl);
	int _getLevel(int logq);
};

extern GlobalParameters param;

/**	Set new parameters with:
	d -- maximum depth of the circuit;
	p -- message modulus;
	w -- bit-length of windowed relinearition;
	min -- bit-length of the smallest coefficient modulus;
	cut -- coefficient cutting size in bits (per level);
	m -- degree of the cyclic polynomial. */
void setParam(int d, int p, int w, int min, int cut, int m);

/** Reset parameters to zeros
	and delete global parameter. */
void resetParam();

} // end cuHE
