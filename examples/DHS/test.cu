#include "DHS.h"
#include "../../cuhe/Debug.h"
int main() {
	setNumDevices(1);
	CuDHS *dhs = new CuDHS(5, 2, 1, 61, 20, 8191);

	ZZX x[2], y[2];
	SetSeed(to_ZZ(time(NULL)));
	for (int k=0; k<2; k++) {
		for (int i=0; i<dhs->numSlot(); i++)
			SetCoeff(x[k], i, RandomBits_ZZ(logMsg()));		
		dhs->batcher->encode(y[k], x[k]);
		dhs->encrypt(y[k], y[k], 0);
	}

	CuCtxt* cuy = new CuCtxt[2];
	CuCtxt cuz;
	for (int k=0; k<2; k++) {
		cuy[k].set(logCoeff(0), 0, y[k]);
		cuy[k].x2n();
	}
	cAnd(cuz, cuy[0], cuy[1]);
	delete [] cuy;	
	cuz.relin();
	cuz.modSwitch();
	cuz.x2z();


	ZZX z;
	dhs->decrypt(z, cuz.zRep(), 1);
	cuz.~CuCtxt();
	dhs->batcher->decode(z, z);

	ZZX chk;
	clear(chk);
	for (int i=0; i<dhs->numSlot(); i++)
		SetCoeff(chk, i, coeff(x[0], i)*coeff(x[1], i)%to_ZZ(modMsg()));
	cout<<deg(z)<<" "<<deg(chk)<<" ";
	if (z == chk)
		cout<<"true"<<endl;
	else
		cout<<"false"<<endl;

	delete dhs;
	return 0;
}