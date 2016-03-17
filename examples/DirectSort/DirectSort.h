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

// Defines circuit depth and minimum coefficient size for decryption,
// according to a given sorting size.
// Provides a sorting method.

#include "DHS.h"
#include <vector>
using namespace std;
#include "../../cuhe/CuHE.h"
using namespace cuHE;

typedef struct {
	ZZX bit;
	int lvl;
} CtxtBit;

typedef struct {
	CtxtBit bit[32];
	int lvl;
} CtxtInt;

class DirectSort {
public:
	DirectSort();
	DirectSort(int);
	~DirectSort();
	void run();
private:
	CuDHS *cudhs;
	int level; // circuit level tracker
	int sortSize; // number of integers to sort
	int circuitDepth; // circuitDepth
	int minCoeffSize; // minimum bit-size of coefficient to decrypt

	vector<int> randList; // create random integers
	vector<int> trueSortedList; // plaintext sort
	vector<int>	sortedList; // homomorphic sort
	vector<vector<ZZX>> list; // encrypted list

	CuCtxt ***cI;
	CuCtxt ***cM;
	CuCtxt ***cS;
	CuCtxt **cO;
	CuCtxt *cE;
	CuCtxt **cT;

	void heSetup();
	void setList();
	void trueSort();
	void printList (vector<int>);
	void configCircuit(int &, int &);
	void encList();
	void decList();
	void directSort();
	void preAllocation();
	void prepareInput();
	void constructMatrix();
	// return enc(0) if a < b
	void isLess(CuCtxt &res, CuCtxt *a, CuCtxt *b);
	void hammingWeights();
	void calHW(CuCtxt *s, CuCtxt *m);
	void calHW4(CuCtxt *s, CuCtxt *m);
	void calHW8(CuCtxt *s, CuCtxt *m);
	void calHW16(CuCtxt *s, CuCtxt *m);
	void calHW32(CuCtxt *s, CuCtxt *m);
	void calHW64(CuCtxt *s, CuCtxt *m);

//	void prepareOutput();
//	void isEqual(CuCtxt &res, CuCtxt *a, CuCtxt *b);
};
