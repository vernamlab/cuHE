#include "DHS.h"
#include "../../cuhe/CuHE.h"
using namespace cuHE;

int main() {
  int p = 2;
  multiGPUs(1);
  CuDHS *dhs = new CuDHS(5, p, 1, 61, 20, 8191);

  ZZX x[1], y[1];
  SetSeed(to_ZZ(time(NULL)));
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

  ZZ ZERO = conv<ZZ>("0");
  ZZ ONE = conv<ZZ>("1");

  ZZX chk;
  clear(chk);
  for (int i=0; i<dhs->numSlot(); i++) {
    ZZ c = ZERO;
    if (coeff(x[0], i) <= ZERO)
      c = ONE;

    SetCoeff(chk, i, c%to_ZZ(p)); // ~
  }

  cout<<deg(z)<<" "<<deg(chk)<<" ";
  if (z == chk)
    cout<<"true"<<endl;
  else
    cout<<"false"<<endl;

  delete dhs;
  return 0;
}