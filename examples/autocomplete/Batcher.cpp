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

#include "Batcher.h"

myCRT::myCRT(){
	ZZ_p::init(to_ZZ(2));
}

void myCRT::SetModulus(ZZX m){
	modulus = to_ZZ_pX(m);
}

void myCRT::ComputeFactors(int f_degree, int f_size){
	factors.SetLength(f_size);
	int s = 1<<f_degree;
	ZZ_pX *list = new ZZ_pX[s];
	for(int i=0; i<s; i++)
		list[i] = to_ZZ_pX(num2ZZX(i+s));

	int j=0;
	ZZ_pX t1 = modulus;
	ZZ_pX comp, remin, quo;
	SetCoeff(comp, 0, 0);
	for(int i=0; i<s; i++){
		DivRem(quo, remin, t1, list[i]);
		if(remin == comp){
			t1 = quo;
			factors[j] = list[i];
			j++;
		}
	}
	size = factors.length();
}

void myCRT::CalculateMs(){
	ZZ_pX temp;
	M.SetLength(size);

	for(int i=0; i<size; i++){
		M[i] = modulus;
		M[i] = M[i] / factors[i];
	}
}
void myCRT::CalculateNs(){
	ZZ_pX mi;
	N.SetLength(size);
	for(int i=0; i<size; i++){
		mi = factors[i];
		ZZ_pE::init(mi);

		ZZ_pE t = to_ZZ_pE((M[i])%mi);
		ZZ_pE ti = inv(t);

		N[i] = rep(ti);
	}
}

void myCRT::CalculateMxNs(){
	MxN.SetLength(size);
	for(int i=0; i<size; i++)
		MxN[i] = (M[i]*N[i])%modulus;
}

bool myCRT::CRTTestNMs(){

	int counter =0;
	ZZ_pX t, realv;

	SetCoeff(realv, 0, 1);

	for(int i=0; i<size; i++){
		t = M[i]*N[i]%factors[i];
			if(t == realv)
				counter++;
	}
	if(counter == size)
		return true;
	else
		return false;
}



ZZX myCRT::EncodeMessageMxN(ZZX &mess){
	ZZ_p::init(to_ZZ("2"));
	ZZ_pX res;
	SetCoeff(res, 0, 0);

	for(int i=0; i<size; i++)
		if(coeff(mess,i) == 1)
			res = res + MxN[i];

	res = res%modulus;
	return to_ZZX(res);
}

ZZX myCRT::DecodeMessage(ZZX &mes){
	ZZ t; ZZX res; ZZ_pX mess_p, tm;

	mess_p = to_ZZ_pX(mes);
	for(int i=0; i<size; i++){
		tm = mess_p%factors[i];
		t = rep(coeff(tm,0));
		t=t%to_ZZ("2");
		SetCoeff(res, i, t);
	}
	return res;
}

/////////////////////////////////////////////////////////

ZZX myCRT::num2ZZX(int num){
	ZZX res;
	SetCoeff(res, 0 , 0);
	if(num == 0)
		return res;
	else{
		for(int i=0; i<32; i++)
			SetCoeff(res, i, (num>>i)%2);
	}
	return res;
}

vec_ZZ_pX myCRT::ReturnM(){
	return M;
}
vec_ZZ_pX myCRT::ReturnN(){
	return N;
}
vec_ZZ_pX myCRT::ReturnMxN(){
	return MxN;
}
vec_ZZ_pX *myCRT::GetMxNPointer(){
	return &MxN;
}
vec_ZZ_pX myCRT::Returnfactors(){
	return factors;
}
ZZ_pX myCRT::Returnmodulus(){
	return modulus;
}
int myCRT::Returnsize(){
	return size;
}

void myCRT::IOReadModulus(fstream &fs){
	fs >> modulus;
}

void myCRT::IOReadSize(fstream &fs){
	fs >> size;
}

void myCRT::IOReadFactors(fstream &fs){
	factors.SetLength(size);
	for(int i=0; i<size; i++)
		fs >> factors[i];
}

void myCRT::IOReadMs(fstream &fs){
	M.SetLength(size);
	for(int i=0; i<size; i++)
		fs >> M[i];
}

void myCRT::IOReadNs(fstream &fs){
	N.SetLength(size);
	for(int i=0; i<size; i++)
		fs >> N[i];
}

void myCRT::IOReadMxNs(fstream &fs){
	MxN.SetLength(size);
	for(int i=0; i<size; i++)
		fs >> MxN[i];
}

void myCRT::IOReadAll(fstream &fs){
	IOReadModulus(fs);
	IOReadSize(fs);
	IOReadFactors(fs);
	IOReadMs(fs);
	IOReadNs(fs);
	IOReadMxNs(fs);
}

myCRT::~myCRT(){}
