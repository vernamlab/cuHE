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

#include "HEScheme.h"
#include <NTL/ZZ_pE.h>
#include <NTL/ZZ_pX.h>
NTL_CLIENT

///////////////////////////////////////////////////////////////////////////////
// @class CuHEScheme
///////////////////////////////////////////////////////////////////////////////
CuHEScheme::CuHEScheme(ParamID id, int modmsg, int logrelin) {}
CuHEScheme::~CuHEScheme() {}
//// Get methods //////////////////////////////////////
ZZX CuHEScheme::polyMod() { return polyMod_;};
//// Tools ////////////////////////////////////////////////
void CuHEScheme::coeffReduce(ZZX& out, ZZX in, int lvl) {
	coeffReduce(out, in, coeffMod_[lvl]);
}
void CuHEScheme::coeffReduce(ZZX& out, ZZX in, ZZ q) {
	clear(out);
	for (int i=0; i<=deg(in); i++)
		SetCoeff(out, i, coeff(in,i)%q);
}
ZZX CuHEScheme::sample(){
	ZZX ret;
	for (int i=0; i<modLen(); i++)
		SetCoeff(ret, i, RandomBnd(to_ZZ(2*B+1))-B);
	return ret;
}
void CuHEScheme::findInverse(ZZX &f_inv, ZZX &f, ZZ &q, bool &isfound) {
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
	for(int i=0; i<modLen(); i++)
		SetCoeff(f_inv, i, rep(coeff(tp, i)));
}
int CuHEScheme::mobuisFunction(int n) {
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
// @class CuDHS
///////////////////////////////////////////////////////////////////////////////
//// Constructor //////////////////////////////////////////
CuDHS::CuDHS(ParamID id, int modmsg, int logrelin):CuHEScheme(id, modmsg, logrelin) {
	setParam(id, modmsg, logrelin);
	coeffMod_ = new ZZ[depth()];
	pk_ = new ZZX[depth()];
	sk_ = new ZZX[depth()];
	if (logRelin() > 0) {
		ek_ = new ZZX[numEvalKey()];
	}
	else {
		ek_ = NULL;
	}
	genPolyMod_();
	initCuHE(coeffMod_, polyMod_);
	B = 1;
	keyGen();
}
CuDHS::~CuDHS() {
	clear(polyMod_);
	delete [] coeffMod_;
	delete [] pk_;
	delete [] sk_;
	if (ek_ != NULL)
		delete [] ek_;
}

//// Primitives ///////////////////////////////////////////
void CuDHS::keyGen() {
	genPkSk();
	if (logRelin() > 0) {
		genEk();
//		initRelin();
	}
}
void CuDHS::encrypt(ZZX& out, ZZX in, int lvl) {
	ZZX s, e;
	s = sample();
	e = sample();
	coeffReduce(s, s, lvl);

	mulZZX(out, pk_[lvl], s, lvl, 0, 0);
//	out = pk_[lvl]*s;
//	out %= polyMod_;
//	coeffReduce(out, out, lvl);

	out += e*modMsg()+in;
	coeffReduce(out, out, lvl);


}
void CuDHS::decrypt(ZZX& out, ZZX in, int lvl) {
	ZZ x;
	ZZX t;
	coeffReduce(in, in, lvl);

	mulZZX(t, sk_[lvl], in, lvl, 0, 0);
//	t = sk_[lvl]*in;
//	t %= polyMod_;
//	coeffReduce(t, t, coeffMod_[lvl]);

	for (int i=0; i<=deg(t); i++) {
		x = coeff(t, i);
		if (x > ((coeffMod_[lvl]-1)/2))
			x -= coeffMod_[lvl];
		SetCoeff(out, i, (x%modMsg()));
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
void CuDHS::genPolyMod_() {
	int s;
	polyMod_ = 1;
	ZZX t_vec[mSize()];
	int s_vec[mSize()];
	for (int i=0; i<mSize(); i++)
		s_vec[i] = 0;
	for (int d=1; d<=mSize(); d++) {
		if (GCD(d, mSize()) == d) {
			ZZX t;
			SetCoeff(t, 0 , -1);
			SetCoeff(t, mSize()/d, 1);
			s = mobuisFunction(d);
			t_vec[d-1] = t;
			s_vec[d-1] = s;
		}
	}
	for (int i=0; i<mSize(); i++)
		if (s_vec[i] == 1)
			polyMod_ *= t_vec[i];
	for (int i=0; i<mSize(); i++)
		if (s_vec[i] == -1)
			polyMod_ /=  t_vec[i];
}
void CuDHS::genPkSk() {
	// sample
	ZZX f, g, ft, f_inv;
	coeffReduce(polyMod_, polyMod_, 0);//
	bool isfound = false;
	while (!isfound) {
		isfound = true;
		ft = sample();
		f = ft*modMsg() + 1;
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
//	pk_[0] = g*f_inv;
//	pk_[0] %= polyMod_;
//	coeffReduce(pk_[0], pk_[0], 0);

	pk_[0] *= modMsg();
	coeffReduce(pk_[0], pk_[0], 0);
	coeffReduce(sk_[0], sk_[0], 0);
	for(int i=1; i<depth(); i++){
		// whether relin or not
		if (logRelin() > 0) {
			sk_[i] = sk_[i-1];
			coeffReduce(sk_[i], sk_[i], 0);
		}
		else {
			mulZZX(sk_[i], sk_[i-1], sk_[i-1], i-1, 0, 0);
		}
		pk_[i] = pk_[i-1];
		coeffReduce(pk_[i], pk_[i], 0);
	}
}
void CuDHS::genEk() {
	ZZX tk = sk_[0];
	ZZ tw =to_ZZ(1);
	ZZ w = to_ZZ(1)<<logRelin();
	ZZX s, e, result, tp;
	for (int i=0; i<numEvalKey(); i++) {
		tp = tk*tw;
		s = sample();
		e = sample();
		coeffReduce(s, s, 0);
		coeffReduce(e, e, 0);
		coeffReduce(tp, tp, 0);
		mulZZX(ek_[i], pk_[0], s, 0, 0, 0);
		ek_[i] += e*modMsg()+tp;
//		ek2.key[i] = pk[0]*s + e*p + tp;
//		Arith_PolyReduce(ek2.key[i], ek2.key[i]);
		coeffReduce(ek_[i], ek_[i], 0);
		tw *= w;
	}
}
