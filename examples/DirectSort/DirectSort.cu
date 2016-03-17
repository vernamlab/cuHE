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

#include "DirectSort.h"
#include "Timer.h"
#include <algorithm>
#include <omp.h>
#include <time.h>

#define detailedTiming
#define	check_cI
#define	check_cM
#define	check_cS

DirectSort::DirectSort() {
}

DirectSort::DirectSort(int val) {
	sortSize = val;
	configCircuit(circuitDepth, minCoeffSize);
	level = 0;
	srand(time(NULL));
}

DirectSort::~DirectSort() {
	sortSize = 0;
	circuitDepth = 0;
	minCoeffSize = 0;
	level = 0;
	randList.clear();
	trueSortedList.clear();
	sortedList.clear();
	list.clear();
	delete cudhs;

	int nGPUs = numGPUs();
	for (int i=0; i<nGPUs; i++) {
		for (int j=0; j<sortSize; j++) {
			delete [] cI[i][j];
			delete [] cS[i][j];
		}
		delete [] cI[i];
		delete [] cS[i];
		delete [] cO[i];
		delete [] cT[i];
	}
	delete [] cI;
	delete [] cS;
	delete [] cT;
	delete [] cO;
	delete [] cE;
}

void DirectSort::run() {
	cout<<"---------- Precomputation --------"<<endl;
	heSetup();
	cout<<"---------- Encrypt List ----------"<<endl;
	setList();
	encList();
	cout<<"---------- Direct Sort -----------"<<endl;
	preAllocation();
  startAllocator();
	otimer ot;
	ot.start();
	directSort();
	ot.stop();
	ot.show("Sort Time");
	stopAllocator();
	cout<<"---------- Decrypt List ----------"<<endl;
	decList();
	trueSort();
	cout<<"Input:  ";
	printList(randList);
	cout<<"Expect: ";
	printList(trueSortedList);
	cout<<"Output: ";
	printList(sortedList);

}

void DirectSort::configCircuit(int &d, int &min) {
	switch (sortSize) {
		case (4):
			d = 12;
			min = 25;
			return;
		case (8):
			d = 13;
			min = 20;
			return;
		case (16):
			d = 14;
			min = 21;
			return;
		case (32):
			d = 15;
			min = 22;
			return;
	}
	printf("Error: wrong sortSize = %d!\n", sortSize);
	terminate();
}

void DirectSort::heSetup() {
	multiGPUs(1);
	cudhs = new CuDHS(circuitDepth, 2, 16, minCoeffSize, minCoeffSize, 8191);
}

void DirectSort::setList() {
	vector<int> t(sortSize, 0);
	for (auto &i: t)
		i = rand()%1000;
	// TODO: remove fixed inputs
	t[0] = 3;
	t[1] = 1;
	t[2] = 0;
	t[3] = 2;
	randList = t;
}

static bool sort2(int x, int y) {
	return (x<y);
}
void DirectSort::trueSort() {
	trueSortedList = randList;
	sort(trueSortedList.begin(), trueSortedList.end(), sort2);
}

void DirectSort::printList (vector<int> vec) {
	for (auto i: vec)
		cout<<i<<"\t";
	cout<<endl;
}

void DirectSort::encList() {
	ZZX ZERO;
	clear(ZERO);
	vector<vector<ZZX>> t(sortSize, vector<ZZX>(32, ZERO));
  for (int i=0; i<sortSize; i++) {
    for(int j=0; j<32; j++) {
      SetCoeff(t[i][j], 0, (randList[i]>>j)&0x1);
      cudhs->encrypt(t[i][j], t[i][j], level);
    }
  }
  list = t;
}

void DirectSort::decList() {
	vector<int> t(sortSize, 0);
  for (int i=0; i<sortSize; i++) {
    for(int j=0; j<32; j++) {
      cudhs->decrypt(list[i][j], list[i][j], level);
      t[i] += to_long(coeff(list[i][j], 0)<<j);
    }
  }
  sortedList = t;
}

void DirectSort::directSort() {
  prepareInput();
#ifdef check_cI
  cudaDeviceSynchronize();
  cout<<"cI:"<<endl;
  ZZX ti;
  ZZ number;
  for (int i=0; i<sortSize; i++) {
  	number = to_ZZ(0);
    for(int j=0; j<32; j++) {
      cI[0][i][j].x2z();
      cudhs->decrypt(ti, cI[0][i][j].zRep(), level);
      number += coeff(ti, 0)<<j;
      cI[0][i][j].x2c();
		}
		cout<<number<<" ";
	}
	cout<<endl;
#endif

  ZZX cmpZZX;
  CuCtxt cmp;
  for (int i=0; i<sortSize; i++) {
	  isLess(cmp, cI[0][0], cI[0][i]);
		cmp.x2z();
		cudhs->decrypt(cmpZZX, cmp.zRep(), level+6);
		cout<<coeff(cmpZZX, 0)<<endl;
	}
/*  constructMatrix();
#ifdef check_cM
  cudaDeviceSynchronize();
  cout<<"cM:"<<endl;
  ZZX tm;
  for (int i=0; i<sortSize; i++) {
    for(int j=0; j<sortSize; j++) {
      cM[0][i][j].x2z();
      cudhs->decrypt(tm, cM[0][i][j].zRep(), level);
      cout<<coeff(tm, 0);
      cM[0][i][j].x2c();
    }
    cout<<endl;
  }
#endif

	hammingWeights();
#ifdef check_cS
	cudaDeviceSynchronize();
	cout<<"cS:"<<endl;
	ZZ idx;
	ZZX ts;
	for (int i=0; i<sortSize; i++) {
	  idx = to_ZZ(0);
	  for(int j=0; j<8; j++) {
	    cS[0][i][j].x2z();
	    cudhs->decrypt(ts, cS[0][i][j].zRep(), level);
	    idx += coeff(ts, 0)<<j;
	    cS[0][i][j].x2c();
	  }
	  cout<<idx<<endl;
	}
#endif
*/
}

void DirectSort::preAllocation() {
	int nGPUs = numGPUs();
  cI = new CuCtxt **[nGPUs];// input list
  cM = new CuCtxt **[nGPUs];// comparison matrix
  cS = new CuCtxt **[nGPUs];// hammingweights or rankings
  cO = new CuCtxt *[nGPUs];// output list
  cE = new CuCtxt[nGPUs];// temp
  cT = new CuCtxt *[nGPUs];// temp
  for (int dev=0; dev<nGPUs; dev++) {
    cI[dev] = new CuCtxt *[sortSize];
    cM[dev] = new CuCtxt *[sortSize];
    cS[dev] = new CuCtxt *[sortSize];
    for (int i=0; i<sortSize; i++) {
      cI[dev][i] = new CuCtxt[32];
      cM[dev][i] = new CuCtxt[sortSize];
      cS[dev][i] = new CuCtxt[8];
    }
    cO[dev] = new CuCtxt[32];
    cT[dev] = new CuCtxt[8];
  }
}

void DirectSort::prepareInput() {
#ifdef detailedTiming
  otimer ot;
  ot.start();
#endif
  int nGPUs = numGPUs();
  #pragma omp parallel num_threads(nGPUs)
  {
    int nowDev = omp_get_thread_num();
    for (int k=0; k<sortSize*32; k++) {
      int i = k/32, j = k%32;
      if (k%nGPUs == nowDev) {
        cI[nowDev][i][j].setLevel(level, nowDev, list[i][j]);
        cI[nowDev][i][j].x2c();
      }
      else {
        cI[nowDev][i][j].setLevel(level, 2, nowDev);
      }
    }
    #pragma omp barrier
    for (int k=0; k<sortSize*32; k++) {
      int i = k/32, j = k%32;
      if (k%nGPUs == nowDev)
        for (int dev=0; dev<nGPUs; dev++)
          if (dev != nowDev)
          	copyTo(cI[dev][i][j], cI[nowDev][i][j], dev);
    }
  }
#ifdef detailedTiming
  ot.stop();
  ot.show("Input");
#endif
}

void DirectSort::constructMatrix() {
#ifdef detailedTiming
  otimer ot;
  ot.start();
#endif
  int nGPUs = numGPUs();
  #pragma omp parallel num_threads(nGPUs)
  {
    int nowDev = omp_get_thread_num();
    int cnt = 0;
    for (int i=0; i<sortSize; i++) {
      cM[nowDev][i][i].setLevel(level+6, 2, nowDev);
      for (int j=i+1; j<sortSize; j++) {
      	if (cnt%nGPUs == nowDev) {
          isLess(cM[nowDev][j][i], cI[nowDev][i], cI[nowDev][j]);
          cNot(cM[nowDev][i][j], cM[nowDev][j][i]);
        }
        else {
          cM[nowDev][i][j].setLevel(level+6, 2, nowDev);
          cM[nowDev][j][i].setLevel(level+6, 2, nowDev);
        }
        cnt ++;
      }
    }
    #pragma omp barrier
    cnt = 0;
    for (int i=0; i<sortSize; i++) {
      for (int j=i+1; j<sortSize; j++) {
        if (cnt%nGPUs == nowDev) {
          for (int dev=0; dev<nGPUs; dev++) {
            if (dev != nowDev) {
            	copyTo(cM[dev][j][i], cM[nowDev][j][i], dev);
            	copyTo(cM[dev][i][j], cM[nowDev][i][j], dev);
            }
          }
        }
        cnt ++;
      }
    }
  }
  level += 6;
#ifdef detailedTiming
  ot.stop();
  ot.show("Matrix");
#endif
}

void DirectSort::isLess(CuCtxt &res, CuCtxt *a, CuCtxt *b) {
  CuCtxt y;
  CuCtxt m[32], t[32];
  // set & lvl 1
  for (int i=0; i<32; i++) {
  	copy(y, b[i]);
  	cNot(m[i], a[i]);
  	cXor(t[i], m[i], y);
  	m[i].x2n();
    y.x2n();
    cAnd(m[i], m[i], y);
    m[i].relin();
  }
  // lvl 1~5
  for (int i=1; i<6; i++) {
    int e = 0x1<<i;
    for (int j=e; j<32; j+=e) {
      t[j].x2n();
      t[j+e/2].x2n();
      cAnd(t[j], t[j], t[j+e/2]);
      t[j].relin();
    }
    e *= 2;
    // only these values are used later
    for (int j=e/4-1; j<32; j+=e/4)
      m[j].modSwitch();
    for (int j=e/4; j<32; j+=e/2)
      t[j].modSwitch();
    for (int j=e/2; j<32; j+=e/2)
      t[j].modSwitch();

    for (int j=e/4-1; j<32; j+=e/2) {
      m[j].x2n();
      t[j+1].x2n();
      cAnd(m[j], m[j], t[j+1]);
      m[j].relin();
    }
    for (int j=e/2-1; j<32; j+=e/2) {
      m[j].x2c();
      m[j-e/4].x2c();
      cXor(m[j], m[j], m[j-e/4]);
    }
  }
  // lvl 6
  m[31].modSwitch();
  copy(res, m[31]);
}

void DirectSort::hammingWeights() {
#ifdef detailedTiming
  otimer ot;
  ot.start();
#endif
  int tempLevel = level;
  int sortSizeIter = sortSize/4;
  while (sortSizeIter > 0) {
    tempLevel ++;
    sortSizeIter /= 2;
  }
  int nGPUs = numGPUs();
	#pragma omp parallel num_threads(nGPUs)
  {
    int nowDev = omp_get_thread_num();
    for (int i=0; i<sortSize; i++) {
    	cout<<"calHW: "<<i<<endl;
      if (i%nGPUs == nowDev)
        calHW(cS[nowDev][i], cM[nowDev][i]);
      else
        for (int j=0; j<8; j++)
          cS[nowDev][i][j].setLevel(tempLevel, 2, nowDev);
    }
    #pragma omp barrier
    for (int i=0; i<sortSize; i++)
      for (int dev=0; dev<nGPUs; dev++)
        if (i%nGPUs == nowDev)
          if (dev != nowDev)
            for (int j=0; j<8; j++)
            	copyTo(cS[dev][i][j], cS[nowDev][i][j], dev);
  }
  level = tempLevel;
#ifdef detailedTiming
  ot.stop();
  ot.show("HWs");
#endif
}

void DirectSort::calHW(CuCtxt *s, CuCtxt *m) {
	switch (sortSize) {
		case 4:
			calHW4(s, m);
			return;
		case 8:
			calHW8(s, m);
			return;
		case 16:
			calHW16(s, m);
			return;
		case 32:
			calHW32(s, m);
			return;
// TODO: case 64 not tested yet
//		case 64:
//			calHW64(s, m);
//			return;
	}
  cout<<"Error: wrong HammingWeight sortSize"<<endl;
  terminate();
}

void DirectSort::calHW4(CuCtxt *s, CuCtxt *m) {
  int lvl = level;
  for (int i=0; i<4; i++)
    m[i].x2c();
  CuCtxt s1;
  cXor(s1, m[0], m[1]);
  cXor(s1, s1, m[2]);
  cXor(s[0], s1, m[3]);
  // Depth 1
  for (int i=0; i<4; i++)
    m[i].x2n();
  CuCtxt temp;
  CuCtxt c1;
  cAnd(c1, m[0], m[1]);
  cAnd(temp, m[0], m[2]);
  cXor(c1, c1, temp);
  cAnd(temp, m[1], m[2]);
  cXor(c1, c1, temp);
  s1.x2n();
  CuCtxt c2;
  cAnd(c2, s1, m[3]);
  cXor(s[1], c1, c2);
  // Finalize
  s[1].relin();
  s[1].modSwitch();
  lvl ++;
  s[0].modSwitch(lvl);
  for (int i=2; i<8; i++)
    s[i].setLevel(s[0].level(), s[0].domain(), s[0].device());
}

void DirectSort::calHW8(CuCtxt *s, CuCtxt *m) {
}
void DirectSort::calHW16(CuCtxt *s, CuCtxt *m) {
}
void DirectSort::calHW32(CuCtxt *s, CuCtxt *m) {
}
void DirectSort::calHW64(CuCtxt *s, CuCtxt *m) {
}