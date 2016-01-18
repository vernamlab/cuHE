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

/*!	/file HEScheme.h
 *	/brief Homomorphic encryption scheme (DHS as an example).
 */

#pragma once

#include "cuHELib/CuHE.h"
using namespace cuHE;

///////////////////////////////////////////////////////////////////////////////
/** @class CuHEScheme */
/*!	/brief Defines an abstract HE scheme on GPU. */
class CuHEScheme {
public:
	//// Constructor //////////////////////////////////////
	CuHEScheme(ParamID id, int modmsg, int logrelin);
	~CuHEScheme();
	//// Primitives ///////////////////////////////////////
	virtual void keyGen() = 0;
	virtual void encrypt(ZZX& out, ZZX in, int lvl) = 0;
	virtual void decrypt(ZZX& out, ZZX in, int lvl) = 0;
	//// Get methods //////////////////////////////////////
	ZZX polyMod();
protected:
	int B;
	//// Tools ////////////////////////////////////////////
	virtual void genPolyMod_() = 0;
	ZZX sample();
	void findInverse(ZZX& f_inv, ZZX& f, ZZ& q, bool& isfound);
	int mobuisFunction(int n);
	void coeffReduce(ZZX& out, ZZX in, int lvl);
	void coeffReduce(ZZX& out, ZZX in, ZZ q);
	ZZX polyMod_; // x^n-1 usually
	ZZ *coeffMod_; // a decreasing sequence of odd moduli
	ZZX *pk_; // public key
	ZZX *sk_; // secret key
	ZZX *ek_; // evaluation key, not always
}; // end CuHEScheme


///////////////////////////////////////////////////////////////////////////////
/** @class DHS */
/*!	/brief HEScheme by Doroz, Sunar and Hammouri.	*/
class CuDHS: public CuHEScheme {
public:
	//// Constructor //////////////////////////////////////
	CuDHS(ParamID id, int modmsg, int logrelin);
	~CuDHS();
	//// Primitives ///////////////////////////////////////
	void keyGen();
	void encrypt(ZZX& out, ZZX in, int lvl);
	void decrypt(ZZX& out, ZZX in, int lvl);
	
	void balance(ZZX& x, int lvl);
	void unbalance(ZZX& x, int lvl);
protected:
	//// Tools ////////////////////////////////////////////
	void genPkSk();
	void genEk();
	void genPolyMod_();
}; // end DHS
