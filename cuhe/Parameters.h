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

/*
template <typename INTEGER>
__inline__ int numBits(INTEGER val) {
	int ret = 0;
	while (val > 0) {
		ret ++;
		val >>= 1;
	}
	return ret;
}
*/

/** Parameters are initialized and reset
	by calling methods of this class. */
class Param {
public:
	Param();
	Param(int d, int p, int w, int min, int cut, int m);
	~Param();
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

/** Get the global parameter. */
Param globalParam();

/*
__inline__ int mSize() { return globalParam().mSize;}
__inline__ int modLen() { return globalParam().modLen;}
__inline__ int rawLen() { return globalParam().modLen2;}
__inline__ int crtLen() { return globalParam().modLen2;}
__inline__ int nttLen() { return 2*globalParam().modLen2;}
__inline__ int depth() { return globalParam().depth;}
__inline__ int logCoeffMax() { return globalParam().logCoeffMax;}
__inline__ int logCoeffMin() { return globalParam().logCoeffMin;}
__inline__ int logCoeffCut() { return globalParam().logCoeffCut;}
__inline__ int modMsg() { return globalParam().modMsg;}
__inline__ int logMsg() { return globalParam().logMsg;}
__inline__ int wordsMsg() { return (logMsg()+31)/32;}
__inline__ int numCrtPrime() { return globalParam().numCrtPrime;}
__inline__ int logCrtPrime() { return globalParam().logCrtPrime;}
__inline__ int logRelin() { return globalParam().logRelin;}
__inline__ int numEvalKey() { return (logCoeffMax()+logRelin()-1)/logRelin();}

__inline__ int numCrtPrime(int lvl) {
	if (lvl == -1)
		return 1;
	else if (lvl >= depth()) {
		cout<<"Error: numCrtPrime(lvl) has lvl: "<<lvl<<endl;
		exit(0);
	}
	else
		return numCrtPrime()-lvl;
}

__inline__ int logCoeff(int lvl) {
	if (lvl == -1)
		return logMsg();
	else if (lvl < depth())
		return logCoeffMax()-lvl*logCoeffCut();
	else if (lvl == depth())
		return logCoeffMin()-logCrtPrime();
	else {
		cout<<"Error: lvl cannot be more than depth!"<<endl;
		exit(-1);
	}
}
__inline__ int wordsCoeff(int lvl) { return (logCoeff(lvl)+31)/32 > 1 ? (logCoeff(lvl)+31)/32 : 1;}
__inline__ int numEvalKey(int lvl) { return (logCoeff(lvl)+logRelin()-1)/logRelin();}
__inline__ int getLevel(int logq) {
	if (logq >= logCoeffMin())
		return (logCoeffMax()-logq)/logCoeffCut();
	else
		return -1; // plaintext
}
*/


} // end cuHE
