#include "../cuhe/CuHE.h"
using namespace cuHE;

int main() {
	setParam(2, 2, 0, 40, 10, 17);
	ZZX polyMod;
	for (int i=0; i<=modLen(); i++)
		SetCoeff(polyMod, i, 1);
	ZZ *coeffMod = new ZZ[depth()];
	initCuHE(coeffMod, polyMod);
	ZZX a;
	SetCoeff(a, 1, 7);
	SetCoeff(a, 3, 7);
	mulZZX(a, a, a, 0, 0, 0);
	cout<<a<<endl;
	return 0;
}
