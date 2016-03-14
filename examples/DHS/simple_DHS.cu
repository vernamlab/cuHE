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

/*
Provide a simple example of homomorphic operations with the DHS scheme.
May also check the functionality and correctness of "CuDHS".

Check:
- and operation
- xor operation
- not operation
- key management
*/

#include "DHS.h"
#include "../../cuhe/CuHE.h"
using namespace cuHE;
#include <iostream>
#include <fstream>
#include <streambuf>
#include <string>
#include <stdio.h>

const int p = 2; // set message space to {0, 1}

CuDHS *dhs;

void checkXor() {
  ZZX x[2], y[2];
  for (int k=0; k<2; k++) {
    for (int i=0; i<dhs->numSlot(); i++)
      SetCoeff(x[k], i, RandomBnd(p));
    dhs->batcher->encode(y[k], x[k]);
    dhs->encrypt(y[k], y[k], 0);
  }

  CuCtxt* cuy = new CuCtxt[2];
  CuCtxt cuz;
  for (int k=0; k<2; k++) {
    cuy[k].setLevel(0, 0, y[k]);
    cuy[k].x2n(); // or x2c() should work as well
  }
  cXor(cuz, cuy[0], cuy[1]); // use cXor
  delete [] cuy;
  // no relin or modswitch!
  cuz.x2z();

  ZZX z;
  dhs->decrypt(z, cuz.zRep(), 0); // decrypt level is 0
  dhs->batcher->decode(z, z);

  ZZX chk;
  clear(chk);
  for (int i=0; i<dhs->numSlot(); i++)
    SetCoeff(chk, i, (coeff(x[0], i)+coeff(x[1], i))%to_ZZ(p));

  cout<<"xor\t";
  if (z == chk)
    cout<<"right"<<endl;
  else
    cout<<"wrong"<<endl;
  return;
}

void checkNot() {
  ZZX x[1], y[1];
  for (int i=0; i<dhs->numSlot(); i++)
    SetCoeff(x[0], i, RandomBnd(p));

  dhs->batcher->encode(y[0], x[0]);
  dhs->encrypt(y[0], y[0], 0);

  CuCtxt cuz;
  cuz.setLevel(0, 0, y[0]);
  cuz.x2c();

  cNot(cuz, cuz); // use cNot
  // no relin or modswitch?
  cuz.x2z();

  ZZX z;
  dhs->decrypt(z, cuz.zRep(), 0); // decrypt level is 0
  dhs->batcher->decode(z, z);

  ZZ ZERO = to_ZZ(0);
  ZZ ONE = to_ZZ(1);

  ZZX chk;
  clear(chk);
  for (int i=0; i<dhs->numSlot(); i++) {
    ZZ c = ZERO;
    if (coeff(x[0], i) == ZERO)
      c = ONE;
    SetCoeff(chk, i, c%to_ZZ(p)); // ~
  }

  cout<<"not\t";
  if (z == chk)
    cout<<"right"<<endl;
  else
    cout<<"wrong"<<endl;
  return;
}

void checkAnd() {
  ZZX x[2], y[2];
  for (int k=0; k<2; k++) {
    for (int i=0; i<dhs->numSlot(); i++)
      SetCoeff(x[k], i, RandomBnd(p));
    dhs->batcher->encode(y[k], x[k]);
    dhs->encrypt(y[k], y[k], 0);
  }

  CuCtxt* cuy = new CuCtxt[2];
  CuCtxt cuz;
  for (int k=0; k<2; k++) {
    cuy[k].setLevel(0, 0, y[k]);
    cuy[k].x2n();
  }
  cAnd(cuz, cuy[0], cuy[1]);
  delete [] cuy;  
  cuz.relin();
  cuz.modSwitch();
  cuz.x2z();


  ZZX z;
  dhs->decrypt(z, cuz.zRep(), 1);
  dhs->batcher->decode(z, z);

  ZZX chk;
  clear(chk);
  for (int i=0; i<dhs->numSlot(); i++)
    SetCoeff(chk, i, coeff(x[0], i)*coeff(x[1], i)%to_ZZ(p));

  cout<<"and\t";
  if (z == chk)
    cout<<"right"<<endl;
  else
    cout<<"wrong"<<endl;
  return;
}

void checkKeys() {
  remove( "private.key" );
  remove( "public.key" );
  ofstream f;
  f.open ("private.key");
  f << dhs->getPrivateKey();
  f.close();
  f.open ("public.key");
  f << dhs->getPublicKey();
  f.close();
  ZZX x, y, z;
  for (int i=0; i<dhs->numSlot(); i++)
    SetCoeff(x, i, RandomBnd(p));

  // dhs --> dhs2(private)
  ifstream priv("private.key", ios::in | ios::binary);
  string privateKey((istreambuf_iterator<char>(priv)),
      istreambuf_iterator<char>());
  dhs->batcher->encode(y, x);
  dhs->encrypt(y, y, 0);
  CuDHS *dhs2 = new CuDHS(privateKey);
  dhs2->decrypt(z, y, 0); // decrypt level is 0
  dhs2->batcher->decode(z, z);
  bool correct = true;
  correct &= (z == x);

  // dhs3(public) --> dhs
  ifstream pub("public.key", ios::in | ios::binary);
  string publicKey((istreambuf_iterator<char>(pub)), istreambuf_iterator<char>());
  CuDHS *dhs3 = new CuDHS(publicKey);
  dhs3->batcher->encode(y, x);
  dhs3->encrypt(y, y, 0);
  dhs->decrypt(z, y, 0); // decrypt level is 0
  dhs->batcher->decode(z, z);
  correct &= (z == x);

  cout<<"keys: \t";
  if (correct)
    cout<<"right"<<endl;
  else
    cout<<"wrong"<<endl;  

  remove( "private.key" );
  remove( "public.key" );
  delete dhs2;
  delete dhs3;  
}


int main() {
  // set the number of GPUs to use
  multiGPUs(1);
  // initialize HE scheme
  dhs = new CuDHS(5, p, 1, 61, 20, 8191);
  // set random seed
  SetSeed(to_ZZ(time(NULL)));

  // test operations
  checkXor();
  checkNot();
  checkAnd();
  checkKeys();

  delete dhs;
  return 0;
}