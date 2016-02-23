#include "DHS.h"
#include "../../cuhe/CuHE.h"
using namespace cuHE;

int main() {
	int p = 2;
	multiGPUs(1);
	CuDHS *dhs = new CuDHS(5, p, 1, 61, 20, 8191);

	ZZX x[2], y[2];
	SetSeed(to_ZZ(time(NULL)));
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
	cout<<deg(z)<<" "<<deg(chk)<<" ";
	if (z == chk)
		cout<<"true"<<endl;
	else
		cout<<"false"<<endl;

	delete dhs;
	return 0;
}
