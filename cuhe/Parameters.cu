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

#include "Parameters.h"
#include <NTL/ZZ.h>
NTL_CLIENT

namespace cuHE {

static Param* param = NULL;

Param globalParam() {
	return *param;
}

void setParam(int d, int p, int w, int min, int cut, int m) {
	if (param != NULL)
		delete param;
	param = new Param(d, p, w, min, cut, m);
}

void resetParam() {
	param->~Param();
	if (param != NULL)
		delete param;
	param = NULL;
}

static ZZ euler_toient(ZZ x) {
	if (x < to_ZZ("3"))
		return x;
	ZZ res = x;
	ZZ t = to_ZZ("2");
	bool is = false;
	while (x != to_ZZ("1")) {
		while (GCD(x,t) == t) {
			x=x/t;
			is = true;
		}
		if (is)
			res = res*(t-1)/t;
		is = false;
		t = NextPrime(t+1);
	}
	return res;
}

Param::Param() {}

Param::Param(int d, int p, int w, int min, int cut, int m) {
	depth = d;
	modMsg = p;
	logRelin = w;
	logCoeffMin = min;
	logCoeffCut = cut;
	mSize = m;

	logCoeffMax = logCoeffMin+logCoeffCut*(depth-1);
	modLen = to_long(euler_toient(to_ZZ(mSize)));
	modLen2 = 0x1<<NumBits(to_ZZ(modLen)-1);
	if (modLen2 < 8192)
		modLen2 = 8192;
	rawLen = modLen2;
	crtLen = modLen2;
	nttLen = 2*modLen2;

	logMsg = NumBits(to_ZZ(modMsg)-1);
	wordsMsg = (logMsg+31)/32;

	numEvalKey = (logCoeffMax+logRelin-1)/logRelin;

	// use as large and as few # of crt primes as possible
	logCrtPrime = NumBits(SqrRoot(to_ZZ(0xffffffff00000001)/modLen));
	numCrtPrime = (logCoeffMin+logCrtPrime-1)/logCrtPrime;
	logCrtPrime = 0;
	while (logCrtPrime*numCrtPrime < logCoeffMin)
		logCrtPrime ++;
	numCrtPrime += depth-1;
}

Param::~Param() {
	mSize = 0;
	modLen = 0;
	modLen2 = 0;
	rawLen = 0;
	crtLen = 0;
	nttLen = 0;
	logCoeffMax = 0;
	logCoeffMin = 0;
	logCoeffCut = 0;
	depth = 0;
	modMsg = 0;
	logMsg = 0;
	wordsMsg = 0;
	logRelin = 0;
	numEvalKey = 0;
	logCrtPrime = 0;
	numCrtPrime = 0;
}

int Param::_numCrtPrime(int lvl) {
	if (lvl == -1)
		return 1;
	else if (lvl >= depth) {
		cout<<"Error: numCrtPrime(lvl) has lvl: "<<lvl<<endl;
		exit(0);
	}
	else
		return numCrtPrime-lvl;
}

int Param::_logCoeff(int lvl) {
	if (lvl == -1)
		return logMsg;
	else if (lvl < depth)
		return logCoeffMax-lvl*logCoeffCut;
	else if (lvl == depth)
		return logCoeffMin-logCrtPrime;
	else {
		cout<<"Error: lvl cannot be more than depth!"<<endl;
		exit(-1);
	}
}

int Param::_wordsCoeff(int lvl) {
	int t = (_logCoeff(lvl)+31)/32;
	return t > 1 ? t : 1;
}

int Param::_numEvalKey(int lvl) {
	return (_logCoeff(lvl)+logRelin-1)/logRelin;
}

int Param::_getLevel(int logq) {
	if (logq >= logCoeffMin)
		return (logCoeffMax-logq)/logCoeffCut;
	else
		return -1; // plaintext
}

} // end cuHE
