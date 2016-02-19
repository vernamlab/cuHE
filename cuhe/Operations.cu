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
#include "Operations.h"
#include "Parameters.h"
#include "DeviceManager.h"
#include "Debug.h"
#include "Base.h"
#include "CuHE.h"

namespace cuHE {
///////////////////////////////////////////////////////////////////////////////
//// Pre-computation //////////////////////////////////////////////////////////
static ZZ* crtPrime; // decreasing?
static ZZ* coeffModulus; // decreasing
void genCrtPrimes() {
	int pnum = param.numCrtPrime;
	crtPrime = new ZZ[pnum];
	unsigned* h_p = new unsigned[pnum];
	int logmid = param.logCoeffMin-(pnum-param.depth)*param.logCrtPrime;
	// after cutting, fairly larger primes
	ZZ temp = to_ZZ(0x1<<param.logCrtPrime)-1;
	for (int i=0; i<=pnum-param.depth-1; i++) {
		while (!ProbPrime(temp, 10))
			temp --;
		conv(h_p[i], temp);
		crtPrime[i] = temp;
		temp --;
	}

	// mid
	ZZ tmid;
	if (logmid != param.logCrtPrime)
		tmid = to_ZZ(0x1<<logmid)-1;
	else
		tmid = temp;
	while (!ProbPrime(tmid, 10))
		tmid --;
	conv(h_p[pnum-param.depth], tmid);
	crtPrime[pnum-param.depth] = tmid;

	// for cutting
	if (param.logCoeffCut == logmid)
		temp = tmid-1;
	else if (param.logCoeffCut == param.logCrtPrime)
		temp --;
	else
		temp = to_ZZ(0x1<<param.logCoeffCut)-1;
	for (int i=pnum-param.depth+1; i<pnum; i++) {
		while (!ProbPrime(temp, 10) || temp%to_ZZ(param.modMsg) != 1)
			temp --;
		conv(h_p[i], temp);
		crtPrime[i] = temp;
		temp --;
	}

	preload_crt_p(h_p, pnum);
	delete [] h_p;
};
void genCoeffModuli() {
	int d = param.depth;
	int pnum = param.numCrtPrime;
	coeffModulus = new ZZ[d];
	for (int i=0; i<d; i++) {
		coeffModulus[i] = 1;
		for (int j=0; j<pnum-i; j++)
			coeffModulus[i] *= crtPrime[j];
	}
}
void genCrtInvPrimes() {
	int pnum = param.numCrtPrime;
	uint32 *h_pinv = new uint32[pnum*(pnum-1)/2];
	ZZ temp;
	for (int i=1; i<pnum; i++)
		for (int j=0; j<i; j++)
			conv(h_pinv[i*(i-1)/2+j], InvMod(crtPrime[i]%crtPrime[j], crtPrime[j]));
	preload_crt_invp(h_pinv, pnum*(pnum-1)/2);
	delete [] h_pinv;
}
static int* icrtLevel; // one int for each device
static struct IcrtConst {
	uint32 *q;
	uint32 *qp;
	uint32 *qpinv;
} **icrtConst;
void genIcrtByLevel(int lvl) {
	int pnum = param._numCrtPrime(lvl);
	int words_q = param._wordsCoeff(lvl);
	int words_qp = param._wordsCoeff(lvl+1);
	for (int dev=0; dev<numDevices(); dev++) {
		CSC(cudaSetDevice(dev));
		CSC(cudaMallocHost(&icrtConst[dev][lvl].q,
				words_q*sizeof(uint32)));
		CSC(cudaMallocHost(&icrtConst[dev][lvl].qp,
				pnum*words_qp*sizeof(uint32)));
		CSC(cudaMallocHost(&icrtConst[dev][lvl].qpinv,
				pnum*sizeof(uint32)));
	}
	ZZ *z_qp = new ZZ[pnum];
	for (int i=0; i<pnum; i++)
		z_qp[i] = coeffModulus[lvl]/crtPrime[i];
	for (int dev=0; dev<numDevices(); dev++) {
		BytesFromZZ((uint8 *)icrtConst[dev][lvl].q,
				coeffModulus[lvl], words_q*sizeof(uint32));
		for (int i=0; i<pnum; i++) {
			BytesFromZZ((uint8 *)(&icrtConst[dev][lvl].qp[words_qp*i]),
					z_qp[i], words_qp*sizeof(uint32));
			conv(icrtConst[dev][lvl].qpinv[i],
					InvMod(z_qp[i]%crtPrime[i], crtPrime[i]));
		}
	}
	delete [] z_qp;
};
void genIcrt() {
	icrtConst = new IcrtConst *[numDevices()];
	icrtLevel = new int[numDevices()];
	for (int dev=0; dev<numDevices(); dev++) {
		icrtConst[dev] = new IcrtConst[param.depth];
		icrtLevel[dev] = -1;
	}
	for (int i=0; i<param.depth; i++)
		genIcrtByLevel(i);
};
void loadIcrtConst(int lvl, int dev, cudaStream_t st) {
	if (icrtLevel[dev] != lvl) {
		int pnum = param._numCrtPrime(lvl);
		int words_q = param._wordsCoeff(lvl);
		int words_qp = param._wordsCoeff(lvl+1);
		CSC(cudaSetDevice(dev));
		load_icrt_M(icrtConst[dev][lvl].q, words_q, dev, st);
		load_icrt_mi(icrtConst[dev][lvl].qp, words_qp*pnum, dev, st);
		load_icrt_bi(icrtConst[dev][lvl].qpinv, pnum, dev, st);
		icrtLevel[dev] = lvl;
	}
};
void getCoeffModuli(ZZ* dst) {
	for (int i=0; i<param.depth; i++)
		dst[i] = coeffModulus[i];
}
void initCrt(ZZ* coeffModulus) {
	genCrtPrimes();
	genCoeffModuli();
	genCrtInvPrimes();
	genIcrt();
	for (int dev=0; dev<numDevices(); dev++)
		loadIcrtConst(0, dev);
	getCoeffModuli(coeffModulus);
}
///////////////////////////////////////////////////////////////////////////////
static uint64 **d_swap; // conversion buffer
static uint32 **d_hold; // intt result buffer
void initNtt() {
	preload_ntt(param.nttLen);
	// temporary result allocation
	d_swap = new uint64 *[numDevices()];
	d_hold = new uint32 *[numDevices()];
	for (int dev=0; dev<numDevices(); dev++) {
		cudaSetDevice(dev);
		CSC(cudaMalloc(&d_swap[dev], param.nttLen*sizeof(uint64)));
		CSC(cudaMalloc(&d_hold[dev], param.numCrtPrime*param.nttLen*sizeof(uint32)));
	}
}
uint32 *inttResult(int dev) {
	return ptrNttHold(dev);
}
uint64 **ptrNttSwap() { return d_swap;}
uint32 **ptrNttHold() {	return d_hold;}
uint64 *ptrNttSwap(int dev) { return d_swap[dev];}
uint32 *ptrNttHold(int dev) { return d_hold[dev];}
///////////////////////////////////////////////////////////////////////////////
uint64 **d_barrett_ntt;
uint32 **d_barrett_crt;
uint32 **d_barrett_src;
void createBarrettTemporySpace() {
	d_barrett_crt = new uint32*[numDevices()];
	d_barrett_ntt = new uint64*[numDevices()];
	d_barrett_src = new uint32*[numDevices()];
	for (int dev=0; dev<numDevices(); dev++) {
		cudaSetDevice(dev);
		CSC(cudaMalloc(&d_barrett_crt[dev], param.numCrtPrime*param.nttLen*sizeof(uint32)));
		CSC(cudaMalloc(&d_barrett_ntt[dev], param.numCrtPrime*param.nttLen*sizeof(uint64)));
		CSC(cudaMalloc(&d_barrett_src[dev], param.numCrtPrime*param.nttLen*sizeof(uint32)));
	}
}
static uint32 *ptrBarrettCrt(int dev) { return d_barrett_crt[dev];}
static uint64 *ptrBarrettNtt(int dev) { return d_barrett_ntt[dev];}
static uint32 *ptrBarrettSrc(int dev) { return d_barrett_src[dev];}
void setPolyModulus(ZZX m) {
	// compute NTL type zm, zu
	ZZ zq = coeffModulus[0];
	ZZX zm = m;
	ZZX zu;
	SetCoeff(zu, 2*param.modLen-1, 1);
	zu /= zm;
	for (int i=0; i<=deg(zm); i++)
		SetCoeff(zm, i, coeff(zm, i)%zq);
	for (int i=0; i<=deg(zu); i++)
		SetCoeff(zu, i, coeff(zu, i)%zq);
	SetCoeff(zm, param.modLen, 0);
	// prep m
	CuCtxt c;
	c.setLevel(0, 0, zm);
	c.x2c();
	preload_barrett_m_c(c.cRep(), param.numCrtPrime*param.crtLen*sizeof(uint32));
	c.x2n();
	preload_barrett_m_n(c.nRep(), param.numCrtPrime*param.nttLen*sizeof(uint64));	
	// prep u
	CuCtxt cc;
	cc.setLevel(0, 0, zu);
	cc.x2n();
	preload_barrett_u_n(cc.nRep(), param.numCrtPrime*param.nttLen*sizeof(uint64));
};
void initBarrett(ZZX m) {
	setPolyModulus(m);
	createBarrettTemporySpace();
}

///////////////////////////////////////////////////////////////////////////////
//// Operations ///////////////////////////////////////////////////////////////
void crt(uint32 *dst, uint32 *src, int logq, int dev, cudaStream_t st) {
	int lvl = param._getLevel(logq);
	cudaSetDevice(dev);
	crt<<<(param.modLen+63)/64, 64, param._wordsCoeff(lvl)*sizeof(uint32)*64, st>>>
			(dst, src, param._numCrtPrime(lvl), param._wordsCoeff(lvl), param.modLen, param.crtLen);
	CCE();
}
void icrt(uint32 *dst, uint32 *src, int logq, int dev, cudaStream_t st) {
	int lvl = param._getLevel(logq);
	loadIcrtConst(lvl, dev, st);
	CSC(cudaStreamSynchronize(st));
	CSC(cudaSetDevice(dev));
	icrt<<<(param.modLen+63)/64, 64, 0, st>>>(dst, src, param._numCrtPrime(lvl),
			param._wordsCoeff(lvl), param._wordsCoeff(lvl+1), param.modLen, param.crtLen);
	CCE();
}
void crtAdd(uint32 *sum, uint32 *x, uint32 *y, int logq, int dev, cudaStream_t st) {
	int lvl = param._getLevel(logq);
	cudaSetDevice(dev);
	crt_add<<<(param.modLen+63)/64, 64, 0, st>>>(sum, x, y, param._numCrtPrime(lvl), param.modLen, param.crtLen);
	CCE();
}
void crtAddInt(uint32 *sum, uint32 *x, unsigned a, int logq, int dev, cudaStream_t st) {
	int lvl = param._getLevel(logq);
	cudaSetDevice(dev);
	crt_add_int<<<(param._numCrtPrime(lvl)+63)/64, 64, 0, st>>>(sum, x, a, param._numCrtPrime(lvl), param.crtLen);
	CCE();
}
void crtAddNX1(uint32 *sum, uint32 *x, uint32 *scalar, int logq, int dev, cudaStream_t st) {
	int lvl = param._getLevel(logq);
	cudaSetDevice(dev);
	crt_add_nx1<<<(param.modLen+63)/64, 64, 0, st>>>(sum, x, scalar, param._numCrtPrime(lvl), param.modLen, param.crtLen);
	CCE();
}
void crtMulInt(uint32 *prod, uint32 *x, int a, int logq, int dev, cudaStream_t st) {
	int lvl = param._getLevel(logq);
	cudaSetDevice(dev);
	crt_mul_int<<<(param.numCrtPrime-lvl+63)/64, 64, 0, st>>>(prod, x, a, param._numCrtPrime(lvl), param.crtLen);
	CCE();
}
void crtModSwitch(uint32 *dst, uint32 *src, int logq, int dev, cudaStream_t st) {
	int lvl = param._getLevel(logq);
	cudaSetDevice(dev);
	modswitch<<<(param.modLen+63)/64, 64, 0, st>>>(dst, src, param._numCrtPrime(lvl),
			param.modLen, param.crtLen, param.modMsg);
	CCE();
}

//// single crt polynomial
void _ntt(uint64 *X, uint32 *x, int dev, cudaStream_t st) {
	if (param.nttLen == 16384) {
		ntt_1_16k_ext<<<param.nttLen/512, 64, 0, st>>>(ptrNttSwap(dev), x);
		CCE();
		ntt_2_16k<<<param.nttLen/512, 64, 0, st>>>(ptrNttSwap(dev));
		CCE();
		ntt_3_16k<<<param.nttLen/512, 64, 0, st>>>(X, ptrNttSwap(dev));
		CCE();
	}
	else if (param.nttLen == 32768) {
		ntt_1_32k_ext<<<param.nttLen/512, 64, 0, st>>>(ptrNttSwap(dev), x);
		CCE();
		ntt_2_32k<<<param.nttLen/512, 64, 0, st>>>(ptrNttSwap(dev));
		CCE();
		ntt_3_32k<<<param.nttLen/512, 64, 0, st>>>(X, ptrNttSwap(dev));
		CCE();
	}
	else if (param.nttLen == 65536) {
		ntt_1_64k_ext<<<param.nttLen/512, 64, 0, st>>>(ptrNttSwap(dev), x);
		CCE();
		ntt_2_64k<<<param.nttLen/512, 64, 0, st>>>(ptrNttSwap(dev));
		CCE();
		ntt_3_64k<<<param.nttLen/512, 64, 0, st>>>(X, ptrNttSwap(dev));
		CCE();
	}
}
void _nttw(uint64 *X, uint32 *x, int coeffwords, int relinIdx, int dev, cudaStream_t st) {
	if (param.nttLen == 16384) {
		ntt_1_16k_ext_block<<<param.nttLen/512, 64, 0, st>>>(ptrNttSwap(dev), x, param.logRelin, relinIdx, coeffwords);
		CCE();
		ntt_2_16k<<<param.nttLen/512, 64, 0, st>>>(ptrNttSwap(dev));
		CCE();
		ntt_3_16k<<<param.nttLen/512, 64, 0, st>>>(X, ptrNttSwap(dev));
		CCE();
	}
	else if (param.nttLen == 32768) {
		ntt_1_32k_ext_block<<<param.nttLen/512, 64, 0, st>>>(ptrNttSwap(dev), x, param.logRelin,relinIdx, coeffwords);
		CCE();
		ntt_2_32k<<<param.nttLen/512, 64, 0, st>>>(ptrNttSwap(dev));
		CCE();
		ntt_3_32k<<<param.nttLen/512, 64, 0, st>>>(X, ptrNttSwap(dev));
		CCE();
	}
	else if (param.nttLen == 65536) {
		ntt_1_64k_ext_block<<<param.nttLen/512, 64, 0, st>>>(ptrNttSwap(dev), x, param.logRelin,relinIdx, coeffwords);
		CCE();
		ntt_2_64k<<<param.nttLen/512, 64, 0, st>>>(ptrNttSwap(dev));
		CCE();
		ntt_3_64k<<<param.nttLen/512, 64, 0, st>>>(X, ptrNttSwap(dev));
		CCE();
	}
}
// !!! x has length of param.nttLen
void _intt(uint32 *x, uint64 *X, int crtidx, int dev, cudaStream_t st) {
	if (param.nttLen == 16384) {
		intt_1_16k<<<param.nttLen/512, 64, 0, st>>>(ptrNttSwap(dev), X);
		CCE();
		ntt_2_16k<<<param.nttLen/512, 64, 0, st>>>(ptrNttSwap(dev));
		CCE();
		intt_3_16k_modcrt<<<param.nttLen/512, 64, 0, st>>>(x, ptrNttSwap(dev), crtidx);
		CCE();
	}
	else if (param.nttLen == 32768) {
		intt_1_32k<<<param.nttLen/512, 64, 0, st>>>(ptrNttSwap(dev), X);
		CCE();
		ntt_2_32k<<<param.nttLen/512, 64, 0, st>>>(ptrNttSwap(dev));
		CCE();
		intt_3_32k_modcrt<<<param.nttLen/512, 64, 0, st>>>(x, ptrNttSwap(dev), crtidx);
		CCE();
	}
	else if (param.nttLen == 65536) {
		intt_1_64k<<<param.nttLen/512, 64, 0, st>>>(ptrNttSwap(dev), X);
		CCE();
		ntt_2_64k<<<param.nttLen/512, 64, 0, st>>>(ptrNttSwap(dev));
		CCE();
		intt_3_64k_modcrt<<<param.nttLen/512, 64, 0, st>>>(x, ptrNttSwap(dev), crtidx);
		CCE();
	}
}
//// all crt polynomials
// ntt
void ntt(uint64 *X, uint32 *x, int logq, int dev, cudaStream_t st) {
	int lvl = param._getLevel(logq);
	for (int i=0; i<param._numCrtPrime(lvl); i++)
		_ntt(X+i*param.nttLen, x+i*param.crtLen, dev, st);
}
void nttw(uint64 *X, uint32 *x, int logq, int dev, cudaStream_t st) {
	int lvl = param._getLevel(logq);
	for (int i=0; i<param._numEvalKey(lvl); i++)
		_nttw(X+i*param.nttLen, x, param._wordsCoeff(lvl), i, dev, st);
}
// intt holding result
void inttHold(uint64 *X, int logq, int dev, cudaStream_t st) {
	int lvl = param._getLevel(logq);
	for (int i=0; i<param._numCrtPrime(lvl); i++)
		_intt(ptrNttHold(dev)+i*param.nttLen, X+i*param.nttLen, i, dev, st);
}
// intt without barrett copy result, x has param.nttLen
void inttDoubleDeg(uint32 *x, uint64 *X, int logq, int dev, cudaStream_t st) {
	int lvl = param._getLevel(logq);
	for (int i=0; i<param._numCrtPrime(lvl); i++)
		_intt(ptrNttHold(dev)+i*param.nttLen, X+i*param.nttLen, i, dev, st);
	CSC(cudaMemcpyAsync(x, ptrNttHold(dev),
			param._numCrtPrime(lvl)*param.nttLen*sizeof(uint32), cudaMemcpyDeviceToDevice, st));
}
// intt without barrett copy result, x has param.crtLen
void intt(uint32 *x, uint64 *X, int logq, int dev, cudaStream_t st) {
	int lvl = param._getLevel(logq);
	for (int i=0; i<param._numCrtPrime(lvl); i++) {
		_intt(ptrNttHold(dev)+i*param.nttLen, X+i*param.nttLen, i, dev, st);
		CSC(cudaMemcpyAsync(x+i*param.crtLen, ptrNttHold(dev)+i*param.nttLen,
			param.crtLen*sizeof(uint32), cudaMemcpyDeviceToDevice, st));
	}
}
// intt with barrett, x has param.crtLen
void inttMod(uint32 *x, uint64 *X, int logq, int dev, cudaStream_t st) {
	int lvl = param._getLevel(logq);
	for (int i=0; i<param._numCrtPrime(lvl); i++)
		_intt(ptrNttHold(dev)+i*param.nttLen, X+i*param.nttLen, i, dev, st);
	barrett(x, lvl, dev, st);
}
void nttMul(uint64 *z, uint64 *y, uint64 *x, int logq, int dev, cudaStream_t st) {
	int lvl = param._getLevel(logq);
	ntt_mul<<<(param.nttLen+63)/64, 64, 0, st>>>(z, y, x, param._numCrtPrime(lvl), param.nttLen);
}
void nttMulNX1(uint64 *z, uint64 *x, uint64 *scalar, int logq, int dev, cudaStream_t st) {
	int lvl = param._getLevel(logq);
	ntt_mul_nx1<<<(param.nttLen+63)/64, 64, 0, st>>>(z, x, scalar, param._numCrtPrime(lvl), param.nttLen);
}
void nttAdd(uint64 *z, uint64 *y, uint64 *x, int logq, int dev, cudaStream_t st) {
	int lvl = param._getLevel(logq);
	ntt_add<<<(param.nttLen+63)/64, 64, 0, st>>>(z, x, y, param._numCrtPrime(lvl), param.nttLen);
}
void nttAddNX1(uint64 *z, uint64 *x, uint64 *scalar, int logq, int dev, cudaStream_t st) {
	int lvl = param._getLevel(logq);
	ntt_add_nx1<<<(param.nttLen+63)/64, 64, 0, st>>>(z, x, scalar, param._numCrtPrime(lvl), param.nttLen);
}

void barrett(uint32 *dst, uint32 *src, int lvl, int dev, cudaStream_t st) {
	cudaSetDevice(dev);
	uint32 *ptrCrt = ptrBarrettCrt(dev);
	uint64 *ptrNtt = ptrBarrettNtt(dev);
	uint32 *ptrSrc = ptrBarrettSrc(dev);
	CSC(cudaMemcpyAsync(ptrSrc, src, param._numCrtPrime(lvl)*param.nttLen*sizeof(uint32),
				cudaMemcpyDeviceToDevice, st));
	// ptrSrc = f, deg = 2n-2
	for (int i=0; i<param._numCrtPrime(lvl); i++)
		_ntt(ptrNtt+i*param.nttLen, ptrSrc+i*param.nttLen+param.modLen-1, dev, st);
	// ptrNtt = f>>(n-1), deg = n-1
	barrett_mul_un<<<(param.nttLen+63)/64, 64, 0, st>>>
			(ptrNtt, param._numCrtPrime(lvl), param.nttLen);
	inttDoubleDeg(ptrCrt, ptrNtt, param._logCoeff(lvl), dev, st);
	// ptrCrt = u * f>>(n-1), deg = 2n-2
	for (int i=0; i<param._numCrtPrime(lvl); i++)
		CSC(cudaMemsetAsync(ptrCrt+i*param.nttLen, 0, param.modLen*sizeof(uint32), st));
	// ptrCrt = u*f>>(2n-1)<<n
	for (int i=0; i<param._numCrtPrime(lvl); i++)
		_ntt(ptrNtt+i*param.nttLen, ptrCrt+i*param.nttLen+param.modLen, dev, st);
	// ptrNtt = (u * f>>(n-1))>>n = u*f>>(2n-1), deg = n-2
	barrett_mul_mn<<<(param.nttLen+63)/64, 64, 0, st>>>
			(ptrNtt, param._numCrtPrime(lvl), param.nttLen);
	// ptrNtt = (m-x^n) * (u * f>>(n-1))>>n, deg = 2n-3
	barrett_sub_1<<<(param.modLen+63)/64, 64, 0, st>>>
			(ptrSrc, ptrCrt, param._numCrtPrime(lvl), param.modLen, param.nttLen);
	// ptrSrc = f - (u*f>>(2n-1))<<n
	inttDoubleDeg(ptrCrt, ptrNtt, param._logCoeff(lvl), dev, st);
	// ptrCrt = (m-x^n) * (u * f>>(2n-1)), deg = 2n-3
	barrett_sub_2<<<(param.nttLen+63)/64, 64, 0, st>>>
			(ptrSrc, ptrCrt, param._numCrtPrime(lvl), param.nttLen);
	// ptrSrc = f - (m*u*f)>>(2n-1), deg = n
	barrett_sub_mc<<<(param.nttLen+63)/64, 64, param._numCrtPrime(lvl)*sizeof(uint32), st>>>
			(ptrSrc, param._numCrtPrime(lvl), param.modLen, param.crtLen, param.nttLen);
	// ptrSrc = ptrSrc - m, deg = n-1
	for (int i=0; i<param._numCrtPrime(lvl); i++)
		CSC(cudaMemcpyAsync(dst+i*param.crtLen, ptrSrc+i*param.nttLen,
				param.crtLen*sizeof(uint32), cudaMemcpyDeviceToDevice, st));
}
void barrett(uint32 *dst, int lvl, int dev, cudaStream_t st) {
	barrett(dst, inttResult(dev), lvl, dev, st);
}
} // end cuHE
