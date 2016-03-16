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
//#define	check_cI
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
	otimer ot;
	ot.start();
	directSort();
	ot.stop();
	ot.show("Sort Time");
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
	switch(sortSize) {
		case(4):
			d = 12;
			min = 20;
			return;
		case(8):
			d = 13;
			min = 20;
			return;
		case(16):
			d = 14;
			min = 21;
			return;
		case(32):
			d = 15;
			min = 22;
			return;
	}
	printf("Error: wrong sortSize = %d!\n", sortSize);
	terminate();
}

void DirectSort::heSetup() {
	multiGPUs(1);
	cudhs = new CuDHS(circuitDepth, 2, 16, minCoeffSize, minCoeffSize, 17);
}

void DirectSort::setList() {
	vector<int> t(sortSize, 0);
	for (auto &i: t)
		i = rand()%1000;
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
  for (int i=0; i<sortSize; i++) {
    for(int j=0; j<32; j++) {
      cI[0][i][j].x2z();
      list[i][j] = cI[0][i][j].zRep();
		}
	}
	decList();
	printList(sortedList);
  for (int i=0; i<sortSize; i++) {
    for(int j=0; j<32; j++) {
      cI[0][i][j].x2c();
    }
  }
#endif

  constructMatrix();
#ifdef check_cM
  cudaDeviceSynchronize();
  cout<<"cM:"<<endl;
  ZZX temp;
  for (int i=0; i<sortSize; i++) {
    for(int j=0; j<sortSize; j++) {
      cM[0][i][j].x2z();
      cudhs->decrypt(temp, cM[0][i][j].zRep(), level);
      cout<<coeff(temp, 0);
      cM[0][i][j].x2c();
    }
    cout<<endl;
  }
#endif
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
      if (k%nGPUs == nowDev) {
        for (int dev=0; dev<nGPUs; dev++)
          if (dev != nowDev)
          	copyTo(cI[dev][i][j], cI[nowDev][i][j], dev);
      }
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
// TODO:remove
#include "../../cuhe/Debug.h"
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
  // TODO: wrong
  m[31].modSwitch();
  cout<<m[31].domain()<<"\t"<<m[31].cRep()<<endl;
  copy(res, m[31]);
  cout<<res.domain()<<"\t"<<res.cRep()<<endl;
  cout<<m[31].domain()<<"\t"<<m[31].cRep()<<endl;
  CSC(cudaFree(m[31].cRep()));
}





