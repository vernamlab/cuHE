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

#include "AutoComplete.h"
#include "HEScheme.h"
#include "cuHELib/Debug.h" // remove after devicemanger is set
#include "Profiler.h"
#include <cuda_profiler_api.h>
#include "cuHELib/CuHE.h"
using namespace cuHE;
#include <NTL/ZZ.h>
NTL_CLIENT
#include <iostream>
#include <fstream>
using namespace std;

AutoComplete::AutoComplete(ParamID id, int numentries, int numoutchars, int logmsg) {
	he = new CuDHS(id, 1<<logmsg, 0);
	batcher = new myCRT();
	batcher->SetModulus(he->polyMod());
	batcher->ComputeFactors(modLen()/numSlot(), numSlot());
	batcher->CalculateMs();
	batcher->CalculateNs();
	batcher->CalculateMxNs();

	numEntries = numentries;
	numOutBits = numoutchars*8;
	numIn = (numEntries+numSlot()-1)/numSlot();
	numOut = (numOutBits+logMsg()-1)/logMsg();

	zin = new ZZX[numIn];
	zout = new ZZX[numOut];
	zzxdict = new ZZX *[numOut];
	for (int i=0; i<numOut; i++)
		zzxdict[i] = new ZZX[numIn];
	zdict = new ZZ[numEntries];

	in = new CuCtxt *[numDevices()];
	for (int dev=0; dev<numDevices(); dev++)
		in[dev] =  new CuCtxt[numIn];
	dict = new CuPtxt *[numOut];
	for (int i=0; i<numOut; i++)
		dict[i] = new CuPtxt[numIn];
	out = new CuCtxt [numOut];
	temp = new CuCtxt *[numDevices()];
	for (int dev=0; dev<numDevices(); dev++)
		temp[dev] =  new CuCtxt[numIn];

	stream_ = new cudaStream_t[numDevices()];
	for (int dev=0; dev<numDevices(); dev++) {
		CSC(cudaSetDevice(dev));
		CSC(cudaStreamCreate(&stream_[dev]));
	}

	chardict = new char *[numEntries];
	for (int i=0; i<numEntries; i++) {
		chardict[i] = new char[numOutBits/8+1];
		for (int j=0; j<=numOutBits/8; j++)
			chardict[i][j] = 0;
	}

}
AutoComplete::~AutoComplete() {
	for (int dev=0; dev<numDevices(); dev++) {
		CSC(cudaSetDevice(dev));
		CSC(cudaStreamDestroy(stream_[dev]));
	}
	delete [] stream_;
	delete he;
	delete batcher;
	delete [] zdict;
	delete [] zin;
	delete [] zout;
	for (int i=0; i<numOut; i++)
		delete [] zzxdict[i];
	delete [] zzxdict;
	for (int i=0; i<numOut; i++)
		delete [] dict[i];
	delete [] dict;
	for (int dev=0; dev<numDevices(); dev++)
		delete [] in[dev];
	delete [] in;
	delete [] out;
	for (int dev=0; dev<numDevices(); dev++)
		delete [] temp[dev];
	delete [] temp;
	for (int i=0; i<numEntries; i++)
		delete [] chardict[i];
	delete [] chardict;
}

void getLine(char *dst, int numChars, unsigned index) {
	char c[3];
	c[2] = (char)(index%26+65);
	index /= 26;
	if (index > 0) {
		c[1] = (char)(index%26+65);
		index /= 26;
		if (index > 0)
			c[0] = (char)(index%26+65);
		else
			c[0] = (char)95;
	}
	else {
		c[1] = (char)95;
		c[0] = (char)95;
	}
	for (int i=0; i<numChars; i++)
		dst[i] = c[i%3];
}

void AutoComplete::saveDictionary() {
	for (int i=0; i<numEntries; i++)
		getLine(chardict[i], numOutBits/8, i);
	for (int i=0; i<numEntries; i++)
		zdict[i] = ZZFromBytes((unsigned char *)chardict[i], numOutBits/8);//to_ZZ(i);
	for (int i=0; i<numIn; i++) {
		for (int j=0; j<numOut; j++) {
			clear(zzxdict[j][i]);
			for (int k=0; k<numSlot(); k++)
				if (k+i*numSlot() < numEntries)
					SetCoeff(zzxdict[j][i], k, (zdict[k+i*numSlot()]>>(j*logMsg()))%modMsg());
			zzxdict[j][i] = batcher->EncodeMessageMxN(zzxdict[j][i]);
			cout<<zzxdict[j][i]<<endl;
		}
	}
}

void AutoComplete::loadDictionary() {
	for (int i=0; i<numIn; i++)
		for (int j=0; j<numOut; j++)
			cin>>zzxdict[j][i];
	for (int i=0; i<numIn; i++) {
		for (int j=0; j<numOut; j++) {
			dict[j][i].set(logMsg(), j%numDevices(), zzxdict[j][i]);
			dict[j][i].x2n();
		}
	}
}

void AutoComplete::userSend() {
	unsigned index = 15;//to_long(RandomBnd(numEntries));
//	cout<<"Expect:\t"<<zdict[index]<<endl;
//	cout<<"Expect:\t"<<chardict[index]<<endl;
	for (unsigned i=0; i<numEntries; i++) {
		if (i == index)
			SetCoeff(zin[i/numSlot()], i%numSlot(), 1);
		else
			SetCoeff(zin[i/numSlot()], i%numSlot(), 0);
	}
	for (int i=0; i<numIn; i++) {
		zin[i] = batcher->EncodeMessageMxN(zin[i]);
		he->encrypt(zin[i], zin[i], 0);
	}
}

double AutoComplete::server() {
	cudaProfilerStart();
	OmpTimer t;
	t.start();

#pragma omp parallel num_threads(numDevices())
{
	int dev = omp_get_thread_num();
	#pragma omp for nowait
	for (int j=0; j<numIn; j++) {
		in[dev][j].set(logCoeff(0), dev, zin[j]);
		in[dev][j].x2n(stream_[dev]);
		for (int d=0; d<numDevices(); d++)
			if (d != dev)
				copy(in[d][j], in[dev][j], stream_[dev]);
	}
	cudaStreamSynchronize(stream_[dev]);
	#pragma omp barrier
	cout<<"NTT done!"<<endl;
	#pragma omp for nowait
	for (int i=0; i<numOut; i++) {
		for (int j=0; j<numIn; j++) {
			cAnd(temp[dev][j], in[dev][j], dict[i][j], stream_[dev]);
			if (j > 0)
				cXor(temp[dev][0], temp[dev][0], temp[dev][j], stream_[dev]);	// ntt add
		}
		copy(out[i], temp[dev][0], stream_[dev]);
		out[i].x2z(stream_[dev]);
		zout[i] = out[i].zRep();
	}
}

	t.stop();
	cudaProfilerStop();
	return t.t();
}

void AutoComplete::userReceive() {
	int index = -1;
	for (int i=0; i<numOut; i++) {
		he->decrypt(zout[i], zout[i], 0);
		zout[i] = batcher->DecodeMessage(zout[i]);
		if (deg(zout[i]) >= 0) {
			if (index == -1)
				index = deg(zout[i]);
			if (index != deg(zout[i])) {
				printf("**** Wrong message slot: (%d, %ld)! ****\n", index, deg(zout[i]));
				terminate();
			}
		}
	}
	cout<<"Index:\t"<<index<<endl;
	ZZ zwords;
	for (int i=numOut-1; i>=0; i--) {
		zwords <<= logMsg();
		zwords += coeff(zout[i], index);
	}
//	cout<<"Receive:\t"<<zwords<<endl;
	char *words = new char[numOutBits/8+1];
	words[numOutBits/8] = 0;
	BytesFromZZ((unsigned char *)words, zwords, numOutBits/8);
	cout<<"Receive:\t"<<words<<endl;
}

void AutoComplete::run() {
//	saveDictionary();
	loadDictionary();
	userSend(); // ZZX *zin
	startAllocator();
	cout << numEntries << "\t" << numOutBits/8 << "\t"
				<< modMsg() << "\t" << server() << " ms" << endl; // ZZX *zout
	stopAllocator();
	userReceive(); // ZZX *zreceive

}


int main() {
	setNumDevices(1);
	AutoComplete *a = new AutoComplete(AUTOCOMPLETE, 26, 40, 1);
	a->run();
	delete a;
	return 0;
}

/*
void AutoComplete::tempDictionary() {
	for (int i=0; i<numEntries; i++)
		getLine(dictionary[i], numOutBits/8, i);
}
size_t AutoComplete::setDict() {
	// slice dictionary
	ZZ entry;
	unsigned **dictSliced = new unsigned *[numEntries];
	for (int i=0; i<numEntries; i++) {
		dictSliced[i] = new unsigned[numOut];
		ZZ entry = ZZFromBytes((unsigned char *)dictionary[i], numOutBits/8);
		for (int j=0; j<numOut; j++) {
			dictSliced[i][j] = to_long( entry & to_ZZ(modMsg()-1) );
			entry >>= logMsg();
		}
	}
	// precompute dictionary
	for (int i=0; i<numOut; i++) {
		int dev = i%numDevices();
		ZZX *temp = new ZZX[numIn];
		for (int j=0; j<numEntries; j++)
			SetCoeff(temp[j/numSlot()], j%numSlot(), dictSliced[j][i]);
		for (int j=0; j<numIn; j++) {
			temp[j] = batcher->EncodeMessageMxN(temp[j]);
			dict[i][j].set(logMsg(), dev, temp[j]);
			dict[i][j].x2n(stream_[dev]);
		}
		delete [] temp;
	}
	for (int i=0; i<numEntries; i++)
		delete [] dictSliced[i];
	delete [] dictSliced;
	return numIn*numOut*nttLen()*sizeof(uint64);
}
size_t AutoComplete::userUp() {
	unsigned index = to_long(RandomBnd(numEntries));
	cout<<"Expect:\t"<<dictionary[index]<<endl;
	for (unsigned i=0; i<numEntries; i++) {
		if (i == index)
			SetCoeff(zin[i/numSlot()], i%numSlot(), 1);
		else
			SetCoeff(zin[i/numSlot()], i%numSlot(), 0);
	}
	size_t ret = 0;
	for (int i=0; i<numIn; i++) {
		zin[i] = batcher->EncodeMessageMxN(zin[i]);
		he->encrypt(zin[i], zin[i], 0);
		ret += sizeof(zin[i]);
	}
	return ret;
}

double AutoComplete::serverCrtAdd() {
	CuCtxt **in;
	in = new CuCtxt *[numDevices()];
	for (int dev=0; dev<numDevices(); dev++)
		in[dev] =  new CuCtxt[numIn];
	cudaProfilerStart();
	OmpTimer t;
	t.start();
	CuCtxt temp;
#pragma omp parallel num_threads(numDevices())
{
	#pragma omp for nowait
	for (int j=0; j<numIn; j++) {
		int dev = j%numDevices();
		in[dev][j].set(logCoeff(0), dev, zin[j]);
		in[dev][j].x2n(stream_[dev]);
		for (int d=0; d<numDevices(); d++)
			if (d != dev)
				copy(in[d][j], in[dev][j], stream_[dev]);
	}
}
#pragma omp barrier
	CSC(cudaDeviceSynchronize());
#pragma omp parallel num_threads(numDevices())
{
	#pragma omp for nowait
	for (int i=0; i<numOut; i++) {
		int dev = i%numDevices();
		CuCtxt *out = new CuCtxt[numIn];
		for (int j=0; j<numIn; j++) {
			cAnd(out[j], in[dev][j], dict[i][j], stream_[dev]);
			out[j].x2c(stream_[dev]);
		}
		for (int j=1; j<numIn; j++) {
			cXor(out[0], out[0], out[j], stream_[dev]);	// crt add
		}
		out[0].x2z(stream_[dev]);
		zout[i] = out[0].zRep();
		delete [] out;
	}
}
#pragma omp barrier
	t.stop();
	cudaProfilerStop();
	for (int dev=0; dev<numDevices(); dev++)
		delete [] in[dev];
	delete [] in;
	return t.t();
}

double AutoComplete::serverNttAdd() {
	cudaProfilerStart();
	OmpTimer t;
	t.start();
#pragma omp parallel num_threads(numDevices())
{
	int dev = omp_get_thread_num();
	#pragma omp for nowait
	for (int j=0; j<numIn; j++)
		in[dev][j].set(logCoeff(0), dev, zin[j]);
	#pragma omp for nowait
	for (int j=0; j<numIn; j++)
		in[dev][j].x2r(stream_[dev]);
	for (int j=0; j<numIn; j++)
		in[dev][j].x2n(stream_[dev]);
	#pragma omp for nowait
	for (int j=0; j<numIn; j++)
		for (int d=0; d<numDevices(); d++)
			if (d != dev)
				copy(in[d][j], in[dev][j], stream_[dev]);

	cudaStreamSynchronize(stream_[dev]);

	#pragma omp for nowait
	for (int i=0; i<numOut; i++) {
		CuCtxt *out = new CuCtxt[numIn];
		for (int j=0; j<numIn; j++) {
			cAnd(out[j], in[dev][j], dict[i][j], stream_[dev]);
			if (j > 0)
				cXor(out[0], out[0], out[j], stream_[dev]);	// ntt add
		}
		out[0].x2z(stream_[dev]);
		zout[i] = out[0].zRep();
		delete [] out;
	}
}
	t.stop();
	cudaProfilerStop();

	return t.t();
}

size_t AutoComplete::userDown() {
	size_t ret = 0;
	for (int i=0; i<numOut; i++) {
		ret += sizeof(zout[i]);
	}
	int index = -1;
	for (int i=0; i<numOut; i++) {
		he->decrypt(zout[i], zout[i], 0);
		zout[i] = batcher->DecodeMessage(zout[i]);
		if (deg(zout[i]) >= 0) {
			if (index == -1)
				index = deg(zout[i]);
			if (index != deg(zout[i])) {
				printf("**** Wrong message slot: (%d, %ld)! ****\n", index, deg(zout[i]));
				terminate();
			}
		}
	}
	cout<<"Index:\t"<<index<<endl;
	ZZ zwords;
	char *words = new char[numOutBits/8+1];
	words[numOutBits/8] = 0;
	for (int i=numOut-1; i>=0; i--) {
		zwords <<= logMsg();
		zwords += coeff(zout[i], index);
	}
	cout<<"Receive:\t"<<words<<endl;
	return ret;
}

// temporary
void getLine(char *dst, int numChars, unsigned index) {
	char c[3];
	c[2] = (char)(index%26+65);
	index /= 26;
	if (index > 0) {
		c[1] = (char)(index%26+65);
		index /= 26;
		if (index > 0)
			c[0] = (char)(index%26+65);
		else
			c[0] = (char)95;
	}
	else {
		c[1] = (char)95;
		c[0] = (char)95;
	}
	for (int i=0; i<numChars; i++)
		dst[i] = c[i%3];
}
*/
