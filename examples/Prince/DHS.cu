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

#include "DHS.h"
#include "../../cuhe/CuHE.h"
using namespace cuHE;

///////////////////////////////////////////////////////////////////////////////
// @class CuDHS
///////////////////////////////////////////////////////////////////////////////
//// Constructor //////////////////////////////////////////
CuDHS::CuDHS(int d, int p, int w, int min, int cut, int m) {
	setParameters(d, p, w, min, cut, m);
	coeffMod_ = new ZZ[param.depth];
	pk_ = new ZZX[param.depth];
	sk_ = new ZZX[param.depth];
	if (param.logRelin > 0) {
		ek_ = new ZZX[param.numEvalKey];
	}
	else {
		ek_ = NULL;
	}
	genPolyMod_(); // generate polynomial modulus
	initCuHE(coeffMod_, polyMod_); // create polynomial ring
	B = 1;
	keyGen(); // key generation
	numSlot_ = param.modLen/factorDegree();
	batcher = new Batcher(polyMod_, param.modLen/numSlot_, numSlot_); // setup batching
}
CuDHS::~CuDHS() {
	clear(polyMod_);
	delete [] coeffMod_;
	delete [] pk_;
	delete [] sk_;
	if (ek_ != NULL)
		delete [] ek_;
//	delete batcher;
}
ZZX CuDHS::polyMod() { return polyMod_;};
ZZ* CuDHS::coeffMod() { return coeffMod_;};
int CuDHS::numSlot() { return numSlot_;};
ZZX* CuDHS::ek() { return ek_;};
//// Primitives ///////////////////////////////////////////
void CuDHS::keyGen() {
	genPkSk();
	if (param.logRelin > 0) {
		genEk();
	}
}
void CuDHS::encrypt(ZZX& out, ZZX in, int lvl) {
	ZZX s, e, t;
	s = sample();
	e = sample();
	coeffReduce(s, s, lvl);

	mulZZX(t, pk_[lvl], s, lvl, 0, 0);
//	t = pk_[lvl]*s;
//	t %= polyMod_;
//	coeffReduce(t, t, lvl);

	t += e*param.modMsg+in;
	coeffReduce(t, t, lvl);
	out = t;
}
void CuDHS::decrypt(ZZX& out, ZZX in, int lvl, int maxMulPath) {
	ZZ x;
	ZZX t = in;
	coeffReduce(t, t, lvl);
	if (param.logRelin > 0)
		for (int i=0; i<maxMulPath; i++)
			mulZZX(t, t, sk_[lvl], lvl, 0, 0);
	else
		mulZZX(t, t, sk_[lvl], lvl, 0, 0);
//	ZZX t = sk_[lvl]*in;
//	t %= polyMod_;
//	coeffReduce(t, t, coeffMod_[lvl]);

	clear(out);
	for (int i=0; i<=deg(t); i++) {
		x = coeff(t, i);
		if (x > ((coeffMod_[lvl]-1)/2))
			x -= coeffMod_[lvl];
		SetCoeff(out, i, (x%param.modMsg));
	}
}
void CuDHS::unbalance(ZZX& x, int lvl) {
	ZZ tp, q = coeffMod_[lvl];
	for (int i=0; i<=deg(x); i++) {
		tp = coeff(x, i);
		if (tp < 0)
			tp += q;
		SetCoeff(x, i, tp);
	}
}
void CuDHS::balance(ZZX& x, int lvl) {
	ZZ tp, q = coeffMod_[lvl];
	for (int i=0; i<=deg(x); i++) {
		tp = coeff(x, i);
		if (tp > ((q-1)/2))
			tp -= q;
		SetCoeff(x, i, tp);
	}
}

//// Tools ////////////////////////////////////////////////
int CuDHS::factorDegree() {
	int ret = 1;
	while ( (power(to_ZZ(param.modMsg), ret)-1)%param.mSize != 0 )
		ret++;
	cout<<ret<<endl;
	return ret;
}

void CuDHS::genPolyMod_() {
	int s;
	polyMod_ = 1;
	ZZX *t_vec = new ZZX[param.mSize];
	int *s_vec = new int[param.mSize];
	for (int i=0; i<param.mSize; i++)
		s_vec[i] = 0;
	for (int d=1; d<=param.mSize; d++) {
		if (GCD(d, param.mSize) == d) {
			ZZX t;
			SetCoeff(t, 0 , -1);
			SetCoeff(t, param.mSize/d, 1);
			s = mobuisFunction(d);
			t_vec[d-1] = t;
			s_vec[d-1] = s;
		}
	}
	for (int i=0; i<param.mSize; i++)
		if (s_vec[i] == 1)
			polyMod_ *= t_vec[i];
	for (int i=0; i<param.mSize; i++)
		if (s_vec[i] == -1)
			polyMod_ /=  t_vec[i];
	delete [] t_vec;
	delete [] s_vec;
}
void CuDHS::genPkSk() {
	// sample
	ZZX f, g, ft, f_inv;
	coeffReduce(polyMod_, polyMod_, 0);//
	bool isfound = false;
	while (!isfound) {
		isfound = true;
		ft = sample();
		f = ft*param.modMsg + 1;
		coeffReduce(f, f, 0);//
		findInverse(f_inv, f, coeffMod_[0], isfound);
		coeffReduce(f_inv, f_inv, 0);
	}
	isfound = false;
	g = sample();
	coeffReduce(g, g, 0);
	// sk[0], pk[0] from (f, g, f_inv)
	sk_[0] = f;
	mulZZX(pk_[0], g, f_inv, 0, 0, 0); // pk[0] = g*f_inv, reduce
//	cout<<pk_[0]<<endl;
//	pk_[0] = g*f_inv;
//	pk_[0] %= polyMod_;
//	coeffReduce(pk_[0], pk_[0], 0);

	pk_[0] *= param.modMsg;
	coeffReduce(pk_[0], pk_[0], 0);
	coeffReduce(sk_[0], sk_[0], 0);
	for(int i=1; i<param.depth; i++){
		sk_[i] = sk_[i-1];
		coeffReduce(sk_[i], sk_[i], i);
		pk_[i] = pk_[i-1];
		coeffReduce(pk_[i], pk_[i], i);
	}
}
void CuDHS::genEk() {
	ZZX tk = sk_[0];
	ZZ tw =to_ZZ(1);
	ZZ w = to_ZZ(1)<<param.logRelin;
	ZZX s, e, result, tp;
	for (int i=0; i<param.numEvalKey; i++) {
		tp = tk*tw;
		s = sample();
		e = sample();
		coeffReduce(s, s, 0);
		coeffReduce(e, e, 0);
		coeffReduce(tp, tp, 0);
		mulZZX(ek_[i], pk_[0], s, 0, 0, 0);
		ek_[i] += e*param.modMsg+tp;
//		ek2.key[i] = pk[0]*s + e*p + tp;
//		Arith_PolyReduce(ek2.key[i], ek2.key[i]);

		coeffReduce(ek_[i], ek_[i], 0);
		tw *= w;
	}
	initRelinearization(ek_);
}
void CuDHS::coeffReduce(ZZX& out, ZZX in, int lvl) {
	coeffReduce(out, in, coeffMod_[lvl]);
}
void CuDHS::coeffReduce(ZZX& out, ZZX in, ZZ q) {
	clear(out);
	for (int i=0; i<=deg(in); i++)
		SetCoeff(out, i, coeff(in,i)%q);
}
ZZX CuDHS::sample(){
	ZZX ret;
	for (int i=0; i<param.modLen; i++)
		SetCoeff(ret, i, RandomBnd(to_ZZ(2*B+1))-B);
	return ret;
}
void CuDHS::findInverse(ZZX &f_inv, ZZX &f, ZZ &q, bool &isfound) {
	ZZ_p::init(q);
	ZZ_pX phi;
	phi = to_ZZ_pX(polyMod_);
	ZZ_pE::init(phi);

	ZZ_pE f_, f_inv_;
	f_ = to_ZZ_pE(to_ZZ_pX(f));
	try{ f_inv_ = inv(f_); }
	catch(runtime_error &e)
	{
		isfound = false;
	}
	ZZ_pX tp = rep(f_inv_);
	for(int i=0; i<param.modLen; i++)
		SetCoeff(f_inv, i, rep(coeff(tp, i)));
}
int CuDHS::mobuisFunction(int n) {
	int t, primes;
	primes = 0;

	if (n == 1)
		return 1;
	else {
		for (int i=2; i<=n; i++) {
			if (ProbPrime(i)) {
				if (GCD(i,n) == i) {
					t=n/i;
					primes++;
					if (GCD(i, t) == i)
						return 0;
				}
			}
		}
		if (primes%2 == 0)
			return 1;
		else
			return -1;
	}
}

///////////////////////////////////////////////////////////////////////////////
// @class Batcher
///////////////////////////////////////////////////////////////////////////////
//// Constructor //////////////////////////////////////////
Batcher::Batcher(ZZX polymod, int f_degree, int f_size) {
	if (param.modMsg != 2) {
		cout<<"Error: This Batcher code only supports 1-bit messages."<<endl;
		terminate();
	}
	ZZ_p::init(to_ZZ(2));
	SetModulus(polymod);
	ComputeFactors(f_degree, f_size);
	CalculateMs();
	CalculateNs();
	CalculateMxNs();
}
Batcher::~Batcher() {}

void Batcher::SetModulus(ZZX m) {
	modulus = to_ZZ_pX(m);
}
void Batcher::ComputeFactors(int f_degree, int f_size) {
	factors.SetLength(f_size);
	int s = 1<<f_degree;
	ZZ_pX *list = new ZZ_pX[s];
	for (int i=0; i<s; i++)
		list[i] = to_ZZ_pX(num2ZZX(i+s));

	int j=0;
	ZZ_pX t1 = modulus;
	ZZ_pX comp, remin, quo;
	SetCoeff(comp, 0, 0);
	for (int i=0; i<s; i++) {
		DivRem(quo, remin, t1, list[i]);
		if (remin == comp) {
			t1 = quo;
			factors[j] = list[i];
			j++;
		}
	}
	size = factors.length();
}
void Batcher::CalculateMs() {
	ZZ_pX temp;
	M.SetLength(size);

	for (int i=0; i<size; i++) {
		M[i] = modulus;
		M[i] = M[i] / factors[i];
	}
}
void Batcher::CalculateNs() {
	ZZ_pX mi;
	N.SetLength(size);
	for (int i=0; i<size; i++) {
		mi = factors[i];
		ZZ_pE::init(mi);

		ZZ_pE t = to_ZZ_pE((M[i])%mi);
		ZZ_pE ti = inv(t);

		N[i] = rep(ti);
	}
}
void Batcher::CalculateMxNs() {
	MxN.SetLength(size);
	for (int i=0; i<size; i++)
		MxN[i] = (M[i]*N[i])%modulus;
}
void Batcher::encode(ZZX &poly, ZZX mess) {
	ZZ_p::init(to_ZZ(2));
	ZZ_pX res;
	SetCoeff(res, 0, 0);
	for (int i=0; i<size; i++)
		if (coeff(mess,i) == 1)
			res = res + MxN[i];
	res %= modulus;
	poly = to_ZZX(res);
}
void Batcher::decode(ZZX &mess, ZZX poly) {
	ZZ t;
	ZZ_pX mess_p, tm;
	mess_p = to_ZZ_pX(poly);
	clear(mess);
	for (int i=0; i<size; i++) {
		tm = mess_p%factors[i];
		t = rep(coeff(tm, 0));
		t %= to_ZZ(2);
		SetCoeff(mess, i, t);
	}
}

ZZX Batcher::num2ZZX(int num){
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
