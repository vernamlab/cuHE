#include "../cuhe/CuHE.h"
using namespace cuHE;

int main() {
	setParameters(2, 2, 0, 40, 10, 17);
	ZZX polyMod;
	for (int i=0; i<=param.modLen; i++)
		SetCoeff(polyMod, i, 1);
	ZZ *coeffMod = new ZZ[param.depth];
	initCuHE(coeffMod, polyMod);
	ZZX a;
	SetCoeff(a, 1, 7);
	SetCoeff(a, 3, 7);
	mulZZX(a, a, a, 0, 0, 0);
	cout<<a<<endl;
	return 0;
}
