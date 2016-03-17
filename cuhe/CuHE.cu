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

#include "CuHE.h"
#include "Debug.h"
#include "Operations.h"
#include "DeviceManager.h"
#include "Relinearization.h"

namespace cuHE {

// Initailization
static uint32 **dhBuffer_; // pinned memory for ZZX-Raw conversions

void initCuHE(ZZ *coeffMod_, ZZX modulus) {
	dhBuffer_ = new uint32 *[numDevices()];
	for (int i=0; i<numDevices(); i++) {
		CSC(cudaSetDevice(i));
		CSC(cudaMallocHost(&dhBuffer_[i],
				param.rawLen*param._wordsCoeff(0)*sizeof(uint32)));
		for (int j=0; j<numDevices(); j++) {
			if (i != j)
				CSC(cudaDeviceEnablePeerAccess(j, 0));
		}
	}
	initNtt();
	initCrt(coeffMod_);
	initBarrett(modulus);
}

void startAllocator() {
	bootDeviceAllocator(param.numCrtPrime*param.nttLen*sizeof(uint64));
}

void stopAllocator() {
	haltDeviceAllocator();
}

void multiGPUs(int num) {
	setNumDevices(num);
}

int numGPUs() {
	return numDevices();
}

void setParameters(int d, int p, int w, int min, int cut, int m) {
	setParam(d, p, w, min, cut, m);
}

void resetParameters() {
	resetParam();
}

void initRelinearization(ZZX* evalkey) {
	initRelin(evalkey);
}

// Operations: CuCtxt & CuPtxt
void copy(CuCtxt &dst, CuCtxt &src, cudaStream_t st) {
	if (&dst != &src) {
		dst.reset();
		dst.setLevel(src.level(), src.domain(), src.device(), st);
		dst.isProd(src.isProd());
		CSC(cudaSetDevice(dst.device()));
		if (dst.domain() == 0)
			dst.zRep(src.zRep());
		else if (dst.domain() == 1)
			CSC(cudaMemcpyAsync(dst.rRep(), src.rRep(),
					dst.rRepSize(), cudaMemcpyDeviceToDevice, st));
		else if (dst.domain() == 2)
			CSC(cudaMemcpyAsync(dst.cRep(), src.cRep(),
					dst.cRepSize(), cudaMemcpyDeviceToDevice, st));
		else if (dst.domain() == 3)
			CSC(cudaMemcpyAsync(dst.nRep(), src.nRep(),
					dst.nRepSize(), cudaMemcpyDeviceToDevice, st));
		CSC(cudaStreamSynchronize(st));
	}
}
void cAnd(CuCtxt &out, CuCtxt &in0, CuCtxt &in1, cudaStream_t st) {
	if (in0.device() != in1.device()) {
		cout<<"Error: Multiplication of different devices!"<<endl;
		terminate();
	}
	if (in0.domain() != 3 || in1.domain() != 3) {
		cout<<"Error: Multiplication of non-NTT domain!"<<endl;
		terminate();
	}
	if (in0.logq() != in1.logq()) {
		cout<<"Error: Multiplication of different levels!"<<endl;
		terminate();
	}
	if (&out != &in0) {
		out.reset();
		out.setLevel(in0.level(), 3, in0.device(), st);
	}
	CSC(cudaSetDevice(out.device()));
	nttMul(out.nRep(), in0.nRep(), in1.nRep(), out.logq(), out.device(), st);
	out.isProd(true);
	CSC(cudaStreamSynchronize(st));
}
void cAnd(CuCtxt &out, CuCtxt &inc, CuPtxt &inp, cudaStream_t st) {
	if (inc.device() != inp.device()) {
		cout<<"Error: Multiplication of different devices!"<<endl;
		terminate();
	}
	if (inc.domain() != 3 || inp.domain() != 3) {
		cout<<"Error: Multiplication of non-NTT domain!"<<endl;
		terminate();
	}
	if (&out != &inc) {
		out.reset();
		out.setLevel(inc.level(), 3, inc.device(), st);
	}
	CSC(cudaSetDevice(out.device()));
	nttMulNX1(out.nRep(), inc.nRep(), inp.nRep(), out.logq(), out.device(), st);
	out.isProd(true);
	CSC(cudaStreamSynchronize(st));
}
void cXor(CuCtxt &out, CuCtxt &in0, CuCtxt &in1, cudaStream_t st) {
	if (in0.device() != in1.device()) {
		cout<<"Error: Addition of different devices!"<<endl;
		terminate();
	}
	if (in0.logq() != in1.logq()) {
		cout<<"Error: Addition of different levels!"<<endl;
		terminate();
	}
	if (in0.domain() == 2 && in1.domain() == 2) {
		if (&out != &in0) {
			out.reset();
			out.setLevel(in0.level(), 2, in0.device(), st);
		}
		CSC(cudaSetDevice(out.device()));
		crtAdd(out.cRep(), in0.cRep(), in1.cRep(), out.logq(), out.device(), st);
		CSC(cudaStreamSynchronize(st));
	}
	else if (in0.domain() == 3 && in1.domain() == 3) {
		if (&out != &in0) {
			out.reset();
			out.setLevel(in0.level(), 3, in0.device(), st);
			out.isProd(in0.isProd()||in1.isProd());
		}
		CSC(cudaSetDevice(out.device()));
		nttAdd(out.nRep(), in0.nRep(), in1.nRep(), out.logq(), out.device(), st);
		CSC(cudaStreamSynchronize(st));
	}
	else {
		cout<<"Error: Addition of non-CRT-nor-NTT domain!"<<endl;
		terminate();
	}
}
void cXor(CuCtxt &out, CuCtxt &in0, CuPtxt &in1, cudaStream_t st) {
	if (in0.device() != in1.device()) {
		cout<<"Error: Addition of different devices!"<<endl;
		terminate();
	}
	if (in0.domain() == 2 && in1.domain() == 2) {
		if (&out != &in0) {
			out.reset();
			out.setLevel(in0.level(), 2, in0.device(), st);
		}
		CSC(cudaSetDevice(out.device()));
		crtAddNX1(out.cRep(), in0.cRep(), in1.cRep(), out.logq(), out.device(), st);
		CSC(cudaStreamSynchronize(st));
	}
	else if (in0.domain() == 3 && in1.domain() == 3) {
		if (&out != &in0) {
			out.reset();
			out.setLevel(in0.level(), 3, in0.device(), st);
			out.isProd(in0.isProd()||in1.isProd());
		}
		CSC(cudaSetDevice(out.device()));
		nttAddNX1(out.nRep(), in0.nRep(), in1.nRep(), out.logq(), out.device(), st);
		CSC(cudaStreamSynchronize(st));
	}
	else {
		cout<<"Error: Addition of non-CRT-nor-NTT domain!"<<endl;
		terminate();
	}
}
void cNot(CuCtxt &out, CuCtxt &in, cudaStream_t st) {
	if (in.domain() != 2) {
		cout<<"Error: cNot of non-CRT domain!"<<endl;
		terminate();
	}
	if (&out != &in) {
		out.reset();
		out.setLevel(in.level(), in.domain(), in.device(), st);
	}
	CSC(cudaSetDevice(out.device()));
	crtAddInt(out.cRep(), in.cRep(), (unsigned)param.modMsg-1, out.logq(),
			out.device(), st);
	CSC(cudaStreamSynchronize(st));
}
void moveTo(CuCtxt &tar, int dstDev, cudaStream_t st) {
	if (dstDev != tar.device()) {
		void *ptr;
		if (tar.domain() == 1) {
			CSC(cudaSetDevice(dstDev));
			ptr = deviceMalloc(tar.rRepSize());
			CSC(cudaSetDevice(tar.device()));
			CSC(cudaMemcpyPeerAsync(ptr, dstDev, tar.rRep(), tar.device(),
					tar.rRepSize(), st));
			tar.rRepFree();
			tar.rRep((uint32 *)ptr);
			CSC(cudaStreamSynchronize(st));
		}
		else if (tar.domain() == 2) {
			CSC(cudaSetDevice(dstDev));
			ptr = deviceMalloc(tar.cRepSize());
			CSC(cudaSetDevice(tar.device()));
			CSC(cudaMemcpyPeerAsync(ptr, dstDev, tar.cRep(), tar.device(),
					tar.cRepSize(), st));
			tar.cRepFree();
			tar.cRep((uint32 *)ptr);
			CSC(cudaStreamSynchronize(st));
		}
		else if (tar.domain() == 3) {
			CSC(cudaSetDevice(dstDev));
			ptr = deviceMalloc(tar.nRepSize());
			CSC(cudaSetDevice(tar.device()));
			CSC(cudaMemcpyPeerAsync(ptr, dstDev, tar.nRep(), tar.device(),
					tar.nRepSize(), st));
			tar.nRepFree();
			tar.nRep((uint64 *)ptr);
			CSC(cudaStreamSynchronize(st));
		}
		tar.device(dstDev);
	}
}
void copyTo(CuCtxt &dst, CuCtxt &src, int dstDev, cudaStream_t st) {
	copy(dst, src, st);
	moveTo(dst, dstDev, st);
}

// NTL Interface
void mulZZX(ZZX &out, ZZX in0, ZZX in1, int lvl, int dev, cudaStream_t st) {
	CuCtxt cin0, cin1;
	cin0.setLevel(lvl, dev, in0);
	cin1.setLevel(lvl, dev, in1);
	cin0.x2n(st);
	cin1.x2n(st);
	cAnd(cin0, cin0, cin1, st);
	cin0.x2z(st);
	out = cin0.zRep();
}

// @class CuPolynomial
// Constructor
CuPolynomial::CuPolynomial() {
	logq_ = -1;
	domain_ = -1;
	device_ = -1;
	clear(zRep_);
	rRep_ = NULL;
	cRep_ = NULL;
	nRep_ = NULL;
	isProd_ = 0;
}
CuPolynomial::~CuPolynomial() {
	reset();
}
void CuPolynomial::reset() {
	clear(zRep_);
	if (rRep_ != NULL)
		rRepFree();
	if (cRep_ != NULL)
		cRepFree();
	if (nRep_ != NULL)
		nRepFree();
	isProd_ = 0;
	logq_ = -1;
	domain_ = -1;
	device_ = -1;
}
// Set Methods
void CuPolynomial::logq(int val) { logq_ = val;}
void CuPolynomial::domain(int val) { domain_ = val;}
void CuPolynomial::device(int val) { device_ = val;}
void CuPolynomial::isProd(bool val) { isProd_ = val;}
void CuPolynomial::zRep(ZZX val) { zRep_ = val;}
void CuPolynomial::rRep(uint32* val) { rRep_ = val;}
void CuPolynomial::cRep(uint32* val) { cRep_ = val;}
void CuPolynomial::nRep(uint64* val) { nRep_ = val;}
// Get Methods
int CuPolynomial::logq() { return logq_;}
int CuPolynomial::device() { return device_;}
int CuPolynomial::domain() { return domain_;}
bool CuPolynomial::isProd() { return isProd_;}
ZZX CuPolynomial::zRep() { return zRep_;}
uint32 * CuPolynomial::rRep() { return rRep_;}
uint32 * CuPolynomial::cRep() { return cRep_;}
uint64 * CuPolynomial::nRep() { return nRep_;}
// Domain Conversions
void CuPolynomial::z2r(cudaStream_t st) {
	if (domain_ != 0) {
		printf("Error: Not in domain ZZX!\n");
		terminate();
	}
	CSC(cudaSetDevice(device_));
	rRepCreate(st);
	for(int i=0; i<param.rawLen; i++)
		BytesFromZZ((uint8 *)(dhBuffer_[device_]+i*coeffWords()),
				coeff(zRep_, i), coeffWords()*sizeof(uint32));
	CSC(cudaMemcpyAsync(rRep_, dhBuffer_[device_], rRepSize(),
				cudaMemcpyHostToDevice, st));
	cudaStreamSynchronize(st);
	clear(zRep_);
	domain_ = 1;
}
void CuPolynomial::r2z(cudaStream_t st) {
	if (domain_ != 1) {
		printf("Error: Not in domain RAW!\n");
		terminate();
	}
	CSC(cudaSetDevice(device_));
	CSC(cudaMemcpyAsync(dhBuffer_[device_], rRep_, rRepSize(),
			cudaMemcpyDeviceToHost, st));
	cudaStreamSynchronize(st);
	clear(zRep_);
	for(int i=0; i<param.modLen; i++)
		SetCoeff( zRep_, i, ZZFromBytes( (uint8 *)(dhBuffer_[device_]
			+i*coeffWords() ), coeffWords()*sizeof(uint32)) );
	rRepFree();
	domain_ = 0;
}
void CuPolynomial::r2c(cudaStream_t st) {
	if (domain_ != 1) {
		printf("Error: Not in domain RAW!\n");
		terminate();
	}
	if (logq_ > param.logCrtPrime) {
		CSC(cudaSetDevice(device_));
		cRepCreate(st);
		crt(cRep_, rRep_, logq_, device_, st);
		rRepFree();
	}
	else {
		cRep_ = rRep_;
		rRep_ = NULL;
	}
	domain_ = 2;
}
void CuPolynomial::c2r(cudaStream_t st) {
	if (domain_ != 2) {
		printf("Error: Not in domain CRT!\n");
		terminate();
	}
	if (logq_ > param.logCrtPrime) {
		CSC(cudaSetDevice(device_));
		rRepCreate(st);
		icrt(rRep_, cRep_, logq_, device_, st);
		cRepFree();
	}
	else {
		rRep_ = cRep_;
		cRep_ = NULL;
	}
	domain_ = 1;
}
void CuPolynomial::c2n(cudaStream_t st) {
	if (domain_ != 2) {
		printf("Error: Not in domain CRT!\n");
		terminate();
	}
	CSC(cudaSetDevice(device_));
	nRepCreate(st);
	ntt(nRep_, cRep_, logq_, device_, st);
	cRepFree();
	domain_ = 3;
}
void CuPolynomial::n2c(cudaStream_t st) {
	if (domain_ != 3) {
		printf("Error: Not in domain NTT!\n");
		terminate();
	}
	CSC(cudaSetDevice(device_));
	cRepCreate(st);
	if (isProd_) {
		inttMod(cRep_, nRep_, logq_, device_, st);
	}
	else {
		intt(cRep_, nRep_, logq_, device_, st);
	}
	isProd_ = false;
	nRepFree();
	domain_ = 2;
}
void CuPolynomial::x2z(cudaStream_t st) {
	if (domain_ == 0)
		return;
	else if (domain_ == 1)
		r2z(st);
	else if (domain_ == 2) {
		c2r(st);
		r2z(st);
	}
	else if (domain_ == 3) {
		n2c(st);
		c2r(st);
		r2z(st);
	}
}
void CuPolynomial::x2r(cudaStream_t st) {
	if (domain_ == 1)
		return;
	else if (domain_ == 0)
		z2r(st);
	else if (domain_ == 2)
		c2r(st);
	else if (domain_ == 3) {
		n2c(st);
		c2r(st);
	}
}
void CuPolynomial::x2c(cudaStream_t st) {
	if (domain_ == 2)
		return;
	else if (domain_ == 0) {
		z2r(st);
		r2c(st);
	}
	else if (domain_ == 1)
		r2c(st);
	else if (domain_ == 3)
		n2c(st);
}
void CuPolynomial::x2n(cudaStream_t st) {
	if (domain_ == 3)
		return;
	else if (domain_ == 0) {
		z2r(st);
		r2c(st);
		c2n(st);
	}
	else if (domain_ == 1) {
		r2c(st);
		c2n(st);
	}
	else if (domain_ == 2)
		c2n(st);	
}
// Memory management
void CuPolynomial::rRepCreate(cudaStream_t st) {
	CSC(cudaSetDevice(device_));
	if (deviceAllocatorIsOn())
		rRep_ = (uint32 *)deviceMalloc(param.numCrtPrime*param.nttLen*sizeof(uint64));
	else
		CSC(cudaMalloc(&rRep_, rRepSize()));
	CSC(cudaMemsetAsync(rRep_, 0, rRepSize(), st));
}
void CuPolynomial::cRepCreate(cudaStream_t st) {
	CSC(cudaSetDevice(device_));
	if (deviceAllocatorIsOn())
		cRep_ = (uint32 *)deviceMalloc(param.numCrtPrime*param.nttLen*sizeof(uint64));
	else
		CSC(cudaMalloc(&cRep_, cRepSize()));
	CSC(cudaMemsetAsync(cRep_, 0, cRepSize(), st));
}
void CuPolynomial::nRepCreate(cudaStream_t st) {
	CSC(cudaSetDevice(device_));
	if (deviceAllocatorIsOn())
		nRep_ = (uint64 *)deviceMalloc(param.numCrtPrime*param.nttLen*sizeof(uint64));
	else
		CSC(cudaMalloc(&nRep_, nRepSize()));
	CSC(cudaMemsetAsync(nRep_, 0, nRepSize(), st));
}
void CuPolynomial::rRepFree() {
	CSC(cudaSetDevice(device_));
	if (deviceAllocatorIsOn())
		deviceFree(rRep_);
	else
		CSC(cudaFree(rRep_));
	rRep_ = NULL;
}
void CuPolynomial::cRepFree() {
	CSC(cudaSetDevice(device_));
	if (deviceAllocatorIsOn())
		deviceFree(cRep_);
	else
		CSC(cudaFree(cRep_));
	cRep_ = NULL;
}
void CuPolynomial::nRepFree() {
	CSC(cudaSetDevice(device_));
	if (deviceAllocatorIsOn())
		deviceFree(nRep_);
	else
		CSC(cudaFree(nRep_));
	nRep_ = NULL;
}
// Utilities
int CuPolynomial::coeffWords() { return (logq_+31)/32;}
size_t CuPolynomial::rRepSize() { return param.rawLen*coeffWords()*sizeof(uint32);}

// @class CuCtxt
// Get Methods
void CuCtxt::setLevel(int lvl, int domain, int device, cudaStream_t st) {
	level_ = lvl;
	logq_ = param._logCoeff(lvl);
	domain_ = domain;
	device_ = device;
	if (domain_ == 0)
		clear (zRep_);
	else if (domain_ == 1)
		rRepCreate(st);
	else if (domain_ == 2)
		cRepCreate(st);
	else if (domain_ == 3)
		nRepCreate(st);
}
void CuCtxt::setLevel(int lvl, int device, ZZX val) {
	level_ = lvl;
	logq_ = param._logCoeff(lvl);
	domain_ = 0;
	device_ = device;
	zRep_ = val;
}
int CuCtxt::level() { return level_;}
// Noise Control
void CuCtxt::modSwitch(cudaStream_t st) {
	if (logq_ < param.logCoeffMin+param.logCoeffCut) {
		printf("Error: Cannot do modSwitch on last level!\n");
		terminate();
	}
	x2c();
	CSC(cudaSetDevice(device_));
	crtModSwitch(cRep_, cRep_, logq_, device_, st);
	CSC(cudaStreamSynchronize(st));
	logq_ -= param.logCoeffCut;
	level_ ++;
}
void CuCtxt::modSwitch(int lvl, cudaStream_t st) {
	if (lvl < level_ || lvl >= param.depth) {
		printf("Error: ModSwitch to unavailable level!\n");
		terminate();
	}
	else if (lvl == level_)
		return;
	x2c();
	CSC(cudaSetDevice(device_));
	while (lvl > level_) {
		crtModSwitch(cRep_, cRep_, logq_, device_, st);
		logq_ -= param.logCoeffCut;
		level_ ++;
	}
	CSC(cudaStreamSynchronize(st));
}
void CuCtxt::relin(cudaStream_t st) {
	CSC(cudaSetDevice(device_));
	x2r();
	nRepCreate(st);
	relinearization(nRep_, rRep_, level_, device_, st);
	CSC(cudaStreamSynchronize(st));
	rRepFree();
	isProd_ = true;
	domain_ = 3;
	n2c();
	CSC(cudaStreamSynchronize(st));
}
size_t CuCtxt::cRepSize() { return param._numCrtPrime(level_)*param.crtLen*sizeof(uint32);}
size_t CuCtxt::nRepSize() { return param._numCrtPrime(level_)*param.nttLen*sizeof(uint64);}

// @class CuPtxt
void CuPtxt::setLogq(int logq, int domain, int device, cudaStream_t st) {
	logq_ = logq;
	domain_ = domain;
	device_ = device;
	if (domain_ == 0)
		clear (zRep_);
	else if (domain_ == 1)
		rRepCreate(st);
	else if (domain_ == 2)
		cRepCreate(st);
	else if (domain_ == 3)
		nRepCreate(st);
}
void CuPtxt::setLogq(int logq, int device, ZZX val) {
	logq_ = logq;
	domain_ = 0;
	device_ = device;
	zRep_ = val;
}
size_t CuPtxt::cRepSize() { return param.crtLen*sizeof(uint32);}
size_t CuPtxt::nRepSize() { return param.nttLen*sizeof(uint64);}

} // namespace cuHE
