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

#include "Parameters.h"
#include <NTL/ZZ.h>
NTL_CLIENT

namespace cuHE {

// Parameter values are here
GlobalParameters param;

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

void setParam(int d, int p, int w, int min, int cut, int m) {
	param.depth = d;
	param.modMsg = p;
	param.logRelin = w;
	param.logCoeffMin = min;
	param.logCoeffCut = cut;
	param.mSize = m;

	param.logCoeffMax = param.logCoeffMin+param.logCoeffCut*(param.depth-1);
	param.modLen = to_long(euler_toient(to_ZZ(param.mSize)));
	param.modLen2 = 0x1<<NumBits(to_ZZ(param.modLen)-1);
	if (param.modLen2 < 8192)
		param.modLen2 = 8192;
	param.rawLen = param.modLen2;
	param.crtLen = param.modLen2;
	param.nttLen = 2*param.modLen2;

	param.logMsg = NumBits(to_ZZ(param.modMsg)-1);
	param.wordsMsg = (param.logMsg+31)/32;

	if (param.logRelin != 0)
		param.numEvalKey = (param.logCoeffMax+param.logRelin-1)/param.logRelin;
	else
		param.numEvalKey = 0;

	// use as large and as few # of crt primes as possible
	param.logCrtPrime = NumBits(SqrRoot(to_ZZ(0xffffffff00000001)/param.modLen));
	param.numCrtPrime = (param.logCoeffMin+param.logCrtPrime-1)/param.logCrtPrime;
	param.logCrtPrime = 0;
	while (param.logCrtPrime*param.numCrtPrime < param.logCoeffMin)
		param.logCrtPrime ++;
	param.numCrtPrime += param.depth-1;
}

void resetParam() {
	param.mSize = 0;
	param.modLen = 0;
	param.modLen2 = 0;
	param.rawLen = 0;
	param.crtLen = 0;
	param.nttLen = 0;
	param.logCoeffMax = 0;
	param.logCoeffMin = 0;
	param.logCoeffCut = 0;
	param.depth = 0;
	param.modMsg = 0;
	param.logMsg = 0;
	param.wordsMsg = 0;
	param.logRelin = 0;
	param.numEvalKey = 0;
	param.logCrtPrime = 0;
	param.numCrtPrime = 0;
}

int GlobalParameters::_numCrtPrime(int lvl) {
	if (lvl == -1)
		return 1;
	else if (lvl >= depth) {
		cout<<"Error: numCrtPrime(lvl) has lvl: "<<lvl<<endl;
		exit(0);
	}
	else
		return numCrtPrime-lvl;
}

int GlobalParameters::_logCoeff(int lvl) {
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

int GlobalParameters::_wordsCoeff(int lvl) {
	int t = (_logCoeff(lvl)+31)/32;
	return t > 1 ? t : 1;
}

int GlobalParameters::_numEvalKey(int lvl) {
	return (_logCoeff(lvl)+logRelin-1)/logRelin;
}

int GlobalParameters::_getLevel(int logq) {
	if (logq >= logCoeffMin)
		return (logCoeffMax-logq)/logCoeffCut;
	else
		return -1; // plaintext
}

} // namespace cuHE
