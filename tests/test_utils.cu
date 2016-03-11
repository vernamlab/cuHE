/*
The MIT License (MIT)

Copyright (c) 2016 Andrea Peruffo

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

#include "../cuhe/Utils.h"
#include <NTL/ZZ.h>
#include <NTL/ZZX.h>
#include <iostream>
#include <string>
#include <vector>
#include <assert.h>

using namespace cuHE_Utils;
NTL_CLIENT

int main() {
  cout << "Start cuHE_Utils Tests" << endl;

  ZZX zzx;
  for (int i=0; i <= 10; i++) {
    SetCoeff(zzx, i, i);
  }
  
  ZZ* zz = new ZZ[11];
  for (int i=10; i >= 0; i--) {
    zz[10-i] = conv<ZZ>(i);
  }

  Picklable* p = new Picklable("k1", zzx);
    
  string sep1 (",");
  p->setSeparator(sep1);

  Picklable* p2 = new Picklable("k2", zz, 11);

  string sep2 ("#");
  p2->setSeparator(sep2);

  assert(p->getKey().compare("k1") == 0);
  assert(p2->getKey().compare("k2") == 0);

  assert(p->getValues().compare("0,1,2,3,4,5,6,7,8,9,10") == 0);
  assert(p2->getValues().compare("10#9#8#7#6#5#4#3#2#1#0") == 0);

  assert(p->pickle().compare("k1,0,1,2,3,4,5,6,7,8,9,10") == 0);
  assert(p2->pickle().compare("k2#10#9#8#7#6#5#4#3#2#1#0") == 0);

  string pickled = p->pickle();

  Picklable* res = new Picklable(pickled);


  assert(res->getKey().compare(p->getKey()) == 0);
  assert(res->getValues().compare(p->getValues()) == 0);

  assert(res->getPoly() == p->getPoly());
  assert(res->getCoeffsLen() == p->getCoeffsLen());

  for (int i = 0; i < res->getCoeffsLen(); i++)
    assert(res->getCoeffs()[i] == p->getCoeffs()[i]);

  p2->setSeparator(",");

  vector<Picklable*> ps;
  ps.push_back(p);
  ps.push_back(p2);

  PicklableMap* pm = new PicklableMap(ps);

  stringstream buffer;
  buffer << p->pickle() << pm->getSeparator() << p2->pickle();
  
  assert(pm->toString().compare(buffer.str()) == 0);

  PicklableMap* pm2 = new PicklableMap(buffer.str());

  cout << pm2->get("k2")->pickle() << endl;
  assert(pm->get("k1")->getPoly() == pm2->get("k1")->getPoly());
  assert(pm->get("k2")->getPoly() == pm2->get("k2")->getPoly());
  
  assert(pm2->get("k1")->pickle().compare("k1,0,1,2,3,4,5,6,7,8,9,10") == 0);
  assert(pm2->get("k2")->pickle().compare("k2,10,9,8,7,6,5,4,3,2,1") == 0);

  cout << "End cuHE_Utils Tests" << endl;
}
