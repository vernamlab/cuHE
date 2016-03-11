#include "DHS.h"
#include "../../cuhe/CuHE.h"
#include <iostream>
#include <fstream>
#include <streambuf>
#include <string>
#include <stdio.h>
using namespace cuHE;

int main() {
  int p = 2;
  multiGPUs(1);
  CuDHS *dhs = new CuDHS(5, p, 1, 61, 20, 8191);

  remove( "private.key" );
  remove( "public.key" );

  ofstream f;
  f.open ("private.key");
  f << dhs->getPrivateKey();
  f.close();

  f.open ("public.key");
  f << dhs->getPublicKey();
  f.close();

  ZZX x, y;
  SetSeed(to_ZZ(time(NULL)));
  for (int i=0; i<dhs->numSlot(); i++)
    SetCoeff(x, i, RandomBnd(p));

  ifstream priv("private.key", ios::in | ios::binary);
  string privateKey((istreambuf_iterator<char>(priv)), istreambuf_iterator<char>());

  CuDHS *dhs2 = new CuDHS(privateKey);

  dhs->batcher->encode(y, x);
  dhs->encrypt(y, y, 0);

  ZZX z;
  dhs2->decrypt(z, y, 0); // decrypt level is 0
  dhs2->batcher->decode(z, z);

  if (z == x)
    cout<<"true"<<endl;
  else
    cout<<"false"<<endl;

  ifstream pub("public.key", ios::in | ios::binary);
  string publicKey((istreambuf_iterator<char>(pub)), istreambuf_iterator<char>());

  CuDHS *dhs3 = new CuDHS(publicKey);

  dhs3->batcher->encode(y, x);
  dhs3->encrypt(y, y, 0);

  dhs->decrypt(z, y, 0); // decrypt level is 0
  dhs->batcher->decode(z, z);

  if (z == x)
    cout<<"true"<<endl;
  else
    cout<<"false"<<endl;

  delete dhs;
  delete dhs2;
  delete dhs3;
  return 0;
}
