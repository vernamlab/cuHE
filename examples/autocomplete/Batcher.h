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

/*!	/file AutoComplete.h
 *	/brief Homomorphic AutoComplete algorithm.
 */

#pragma once

#include <iostream>
#include <sstream>
#include <fstream>
#include <NTL/ZZ.h>
#include <NTL/ZZX.h>
#include <NTL/mat_ZZ.h>
#include <NTL/mat_ZZ_p.h>
#include <NTL/ZZ_pX.h>
#include <NTL/ZZ_pE.h>
#include <NTL/ZZ_pXFactoring.h>

using namespace std;
NTL_CLIENT

class myCRT{
public:
	myCRT();
	void SetModulus(ZZX m);
	void ComputeFactors(int f_degree, int f_size);
	void CalculateMs();
	void CalculateNs();
	void CalculateMxNs();
	bool CRTTestNMs();
	ZZX	 EncodeMessageMxN(ZZX &mess);
	ZZX	 DecodeMessage(ZZX &mess);

	ZZX 		num2ZZX(int num);
	vec_ZZ_pX 	ReturnM();
	vec_ZZ_pX 	ReturnN();
	vec_ZZ_pX 	ReturnMxN();
	vec_ZZ_pX 	Returnfactors();
	ZZ_pX	  	Returnmodulus();
	int 	  	Returnsize();

	vec_ZZ_pX	*GetMxNPointer();

	void 		IOReadAll(fstream &fs);
	void 		IOReadModulus(fstream &fs);
	void 		IOReadSize(fstream &fs);
	void 		IOReadFactors(fstream &fs);
	void 		IOReadMs(fstream &fs);
	void 		IOReadNs(fstream &fs);
	void 		IOReadMxNs(fstream &fs);
	~myCRT();
private:
	vec_ZZ_pX 	M;
	vec_ZZ_pX 	N;
	vec_ZZ_pX	MxN;
	vec_ZZ_pX 	factors;
	ZZ_pX		modulus;
	int 		size;
};
