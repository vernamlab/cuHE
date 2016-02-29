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

// Define relinearization keys, methods.

#include "Relinearization.h"
#include "Parameters.h"
#include "DeviceManager.h"
#include "Debug.h"
#include "Base.h"
#include "CuHE.h"
#include "Operations.h"

namespace cuHE {

static uint64** d_relin; // nttw conversion buffer
static uint64*** d_ek; // buffer for a part of eval keys
static uint64** h_ek; // all eval keys in ntt domain
static int more = 1; // 1 <= more <= param.numCrtPrime

// Pre-computation
void initRelin(ZZX* evalkey) {
	CSC(cudaSetDevice(0));	
	h_ek = new uint64* [param.numCrtPrime];
	for (int i=0; i<param.numCrtPrime; i++)
		CSC(cudaMallocHost(&h_ek[i], param.numEvalKey*param.nttLen*sizeof(uint64)));

	for (int i=0; i<param.numEvalKey; i++) {
		CuCtxt ek;
		ek.setLevel(0, 0, evalkey[i]);
		ek.x2n();
		for (int j=0; j<param.numCrtPrime; j++)
			CSC(cudaMemcpy(h_ek[j]+i*param.nttLen, ek.nRep()+j*param.nttLen,
					param.nttLen*sizeof(uint64), cudaMemcpyDeviceToHost));
	}

	d_relin = new uint64* [numDevices()];
	d_ek = new uint64** [numDevices()];
	for (int dev=0; dev<numDevices(); dev++) {
		CSC(cudaSetDevice(dev));
		CSC(cudaMalloc(&d_relin[dev],
				param.numEvalKey*param.nttLen*sizeof(uint64)));
		d_ek[dev] = new uint64* [more];
		for (int i=0; i<more; i++)
			CSC(cudaMalloc(&d_ek[dev][i],
					param.numEvalKey*param.nttLen*sizeof(uint64)));
//		for (int i=0; i<more; i++)
//			CSC(cudaMemcpy(d_ek[dev]+i*param.numEvalKey*param.nttLen,
//						h_ek[dev*more+i], param.numEvalKey*param.nttLen*sizeof(uint64),
//						cudaMemcpyHostToDevice));
	}
}

// Operations
void relinearization(uint64 *dst, uint32 *src, int lvl, int dev,
		cudaStream_t st) {
	CSC(cudaSetDevice(dev));
	nttw(d_relin[dev], src, param._logCoeff(lvl), dev, st);
	for (int i=0; i<param._numCrtPrime(lvl); i++) {
		CSC(cudaMemcpyAsync(d_ek[dev][i%more], h_ek[i],
				param._numEvalKey(lvl)*param.nttLen*sizeof(uint64),
				cudaMemcpyHostToDevice, st));
		relinMulAddPerCrt<<<(param.nttLen+63)/64, 64, 0, st>>>(dst+i*param.nttLen,
				d_relin[dev], d_ek[dev][i%more], param._numEvalKey(lvl), param.nttLen);
		CCE();
	}
}

} // namespace cuHE
