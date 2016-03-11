#include "Prince.h"
#include "Timer.h"
#include <omp.h>
#include "../../cuhe/CuHE.h"
using namespace cuHE;

const int circuitDepth = 25;

int RC[768] = {
	0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	0,0,0,1,0,0,1,1,0,0,0,1,1,0,0,1,1,0,0,0,1,0,1,0,0,0,1,0,1,1,1,0,
	0,0,0,0,0,0,1,1,0,1,1,1,0,0,0,0,0,1,1,1,0,0,1,1,0,1,0,0,0,1,0,0,
	1,0,1,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,1,1,1,0,0,0,0,0,1,0,0,0,1,0,
	0,0,1,0,1,0,0,1,1,0,0,1,1,1,1,1,0,0,1,1,0,0,0,1,1,1,0,1,0,0,0,0,
	0,0,0,0,1,0,0,0,0,0,1,0,1,1,1,0,1,1,1,1,1,0,1,0,1,0,0,1,1,0,0,0,
	1,1,1,0,1,1,0,0,0,1,0,0,1,1,1,0,0,1,1,0,1,1,0,0,1,0,0,0,1,0,0,1,
	0,1,0,0,0,1,0,1,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,1,1,1,1,0,0,1,1,0,
	0,0,1,1,1,0,0,0,1,1,0,1,0,0,0,0,0,0,0,1,0,0,1,1,0,1,1,1,0,1,1,1,
	1,0,1,1,1,1,1,0,0,1,0,1,0,1,0,0,0,1,1,0,0,1,1,0,1,1,0,0,1,1,1,1,
	0,0,1,1,0,1,0,0,1,1,1,0,1,0,0,1,0,0,0,0,1,1,0,0,0,1,1,0,1,1,0,0,
	0,1,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0,1,0,0,1,1,1,1,0,1,1,1,1,0,0,0,
	1,1,1,1,1,1,0,1,1,0,0,1,0,1,0,1,0,1,0,1,1,1,0,0,1,0,1,1,0,0,0,1,
	1,0,0,0,0,1,0,1,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,1,
	1,1,1,1,0,0,0,1,1,0,1,0,1,1,0,0,0,1,0,0,0,0,1,1,1,0,1,0,1,0,1,0,
	1,1,0,0,1,0,0,0,1,0,0,0,0,0,1,0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,1,1,
	0,0,1,0,0,1,0,1,0,0,1,1,0,0,1,0,0,0,1,1,1,1,0,0,0,1,0,1,0,1,0,0,
	0,1,1,0,0,1,0,0,1,0,1,0,0,1,0,1,0,0,0,1,0,0,0,1,1,0,0,1,0,1,0,1,
	1,1,1,0,0,0,0,0,1,1,1,0,0,0,1,1,0,1,1,0,0,0,0,1,0,0,0,0,1,1,0,1,
	1,1,0,1,0,0,1,1,1,0,1,1,0,1,0,1,1,0,1,0,0,0,1,1,1,0,0,1,1,0,0,1,
	1,1,0,0,1,0,1,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,1,1,1,0,0,1,1,0,0,1,
	1,1,0,0,0,0,0,0,1,0,1,0,1,1,0,0,0,0,1,0,1,0,0,1,1,0,1,1,0,1,1,1,
	1,1,0,0,1,0,0,1,0,1,1,1,1,1,0,0,0,1,0,1,0,0,0,0,1,1,0,1,1,1,0,1
};

Prince::Prince() {
	cudhs = NULL;
	level = 0;
	round = 0;
}

Prince::~Prince() {
	delete cudhs;
}

void Prince::heSetup() {
	multiGPUs(1);
	cudhs = new CuDHS(circuitDepth, 2, 16, 25, 25, 21845);
}

void Prince::setMessage(int *m) {
	for	(int i=0; i<64; i++) {
		bits[i] = m[i];
		cudhs->encrypt(bits[i], bits[i], 0);
	}
}

void Prince::setKeys(int *k0, int *k1) {
	for (int i=0; i<64; i++) {
		key0[i] = k0[i];
		key1[i] = k1[i];
		cudhs->encrypt(key0[i], key0[i], 0);
		cudhs->encrypt(key1[i], key1[i], 0);
	}
}

void Prince::run() {
	int A[64],B[64],C[64];
	for (int i=0; i<64; i++) {
		A[i] = 0;
		B[i] = 1;
		C[i] = 0;
	}
	cout<<"---------- Precomputation ----------"<<endl;
	heSetup();
/*	cout<<"---------- Set Message ----------"<<endl;
	setMessage(A);
	cout<<"---------- Set Keys ----------"<<endl;
	setKeys(B, C);
	cout<<"---------- PRINCE ENC ----------"<<endl;

	otimer ot;
	ot.start();
	princeEncrypt(bits, key0, key1);
	ot.stop();
	ot.show("Prince Encryption");

	cout<<"---------- PRINCE DEC ----------"<<endl;
	ZZX res[64];
	for (int i=0; i<64; i++)
		cudhs->decrypt(res[i], bits[i], circuitDepth-1);
	for (int i=0; i<64; i++)
		cout<< coeff(res[i], 0);
	cout<<endl;
	cout<<"1001111110110101000110010011010111111100001111011111010100100100"<<endl;*/
	int p = 2;
	ZZX x[2], y[2];
	SetSeed(to_ZZ(time(NULL)));
	for (int k=0; k<2; k++) {
		for (int i=0; i<cudhs->numSlot(); i++)
			SetCoeff(x[k], i, RandomBnd(p));
		cudhs->batcher->encode(y[k], x[k]);
		cudhs->encrypt(y[k], y[k], 0);
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
	cudhs->decrypt(z, cuz.zRep(), 1);
	cuz.~CuCtxt();
	cudhs->batcher->decode(z, z);

	ZZX chk;
	clear(chk);
	for (int i=0; i<cudhs->numSlot(); i++)
		SetCoeff(chk, i, coeff(x[0], i)*coeff(x[1], i)%to_ZZ(p));
	cout<<deg(z)<<" "<<deg(chk)<<" ";
	if (z == chk)
		cout<<"true"<<endl;
	else
		cout<<"false"<<endl;
}

void Prince::check(int rd) {
	cout<<"Round: "<<rd<<endl;
	for (int i=0; i<64; i++) {
		ZZX chk;
		cudhs->decrypt(chk, bits[i], level);
		cout<<coeff(chk, 0);
	}
	cout<<endl;
	switch(rd) {
	case(0):
			cout<<"0100010001000100010001000100010001000100010001000100010001000100"<<endl<<endl;
			break;
	case(1):
			cout<<"1100000111000101111011011001100010100001001010100010000110111011"<<endl<<endl;
			break;
	case(2):
			cout<<"0001010111110110111001101000001101110010101111110010111100010111"<<endl<<endl;
			break;
	case(3):
			cout<<"0000111110110100100011001100001110111010101010110110101101110000"<<endl<<endl;
			break;
	case(4):
			cout<<"0011100101111101011100000001110101111100101110010111101100111110"<<endl<<endl;
			break;
	case(5):
			cout<<"0110001011001101101111001000001100011000011100100010110011100011"<<endl<<endl;
			break;
	case(6):
			cout<<"1111000000000111010110001001011111100101001011001111001001101110"<<endl<<endl;
			break;
	case(7):
			cout<<"1110011011001010101100101000110011100000011101111010000011101110"<<endl<<endl;
			break;
	case(8):
			cout<<"1111010001000111111011111011110001100001100000001111011100100100"<<endl<<endl;
			break;
	case(9):
			cout<<"0010001000000000000010101101010110110101101010110110011110101111"<<endl<<endl;
			break;
	case(10):
			cout<<"0011101110000011000111101111001010110001111011110111111101111011"<<endl<<endl;
			break;
	case(11):
			cout<<"1010000011100110110011110111110111001010101111100101101000000111"<<endl<<endl;
			break;
	}

}

void Prince::princeEncrypt(ZZX tar[64], ZZX k0[64], ZZX k1[64]) {
//	otimer ot;

	round = 0;
	addRoundKey(tar, k0);
	addRoundKey(tar, k1);
	addRC(tar, round);
	cout<<"start sbox"<<endl;
	for (int i=0; i<5; i++) {
		round++;
//		ot.start();
		SBOX(tar);//
//		ot.stop();
		check(round-1);

		MixColumn(tar);
		addRC(tar, round);
		addRoundKey(tar, k1);
	}

//	ot.start();
	SBOX(tar);//
//	ot.stop();
	check(round);

	M_p(tar);

//	ot.start();
	INV_SBOX(tar);//
//	ot.stop();
	check(round+1);

	for (int i=0; i<5; i++) {
		round++;
		addRoundKey(tar, k1);
		addRC(tar, round);
		inv_MixColumn(tar);

//		ot.start();
		INV_SBOX(tar);//
//		ot.stop();
		check(round+1);
    }
	round++;
	addRC(tar, round);
	addRoundKey(tar, k1);
	KeyExpansion(k0);
	addRoundKey(tar, k0);
	for (int i=0; i<64; i++)
		cudhs->coeffReduce(tar[i], tar[i], circuitDepth-1);

//	ot.show("All Sbox");
}

void Prince::SBOX(ZZX in[64]) {
	for (int i=0; i<64; i++)
		cudhs->coeffReduce(in[i], in[i], level);
//	#pragma omp parallel num_threads(1)
//	#pragma omp parallel num_threads(2)
//	#pragma omp parallel num_threads(3)
	#pragma omp parallel num_threads(numGPUs())
	{
		#pragma omp for nowait
		for (int i=0; i<16; i++)
			_sbox(&in[4*i], omp_get_thread_num(), level);
//			_sbox(&in[4*i], 2, level);
	}
	#pragma omp barrier
	level += 2;
};

void Prince::_sbox(ZZX in[4], int dev, int lvl) {
	CuCtxt a, b, c, d;
	CuCtxt ab, ac, ad, bc, bd, cd;
	CuCtxt abd, acd, bcd, abc;
	CuCtxt out[4];
	///////////////////////////////////
	a.setLevel(lvl, dev, in[0]);
	b.setLevel(lvl, dev, in[1]);
	c.setLevel(lvl, dev, in[2]);
	d.setLevel(lvl, dev, in[3]);
	a.x2n();
	b.x2n();
	c.x2n();
	d.x2n();
	// mul
	cAnd(ab, a, b);
	cAnd(ac, a, c);
	cAnd(ad, a, d);
	cAnd(bc, b, c);
	cAnd(bd, b, d);
	cAnd(cd, c, d);
	// relin
	ab.relin();
	cd.relin();
	// modswitch
	ab.modSwitch();
	ac.modSwitch();
	ad.modSwitch();
	bc.modSwitch();
	bd.modSwitch();
	cd.modSwitch();
	a.modSwitch();
	b.modSwitch();
	c.modSwitch();
	d.modSwitch();
	///////////////////////////////////
	// level up
	// add
	// out[0] = a+c+ab+bc+1;
	cXor(out[0], a, c);
	cXor(out[0], out[0], ab);
	cXor(out[0], out[0], bc);
	cNot(out[0], out[0]);
	// out[1] = a+d+ac+ad+cd;
	cXor(out[1], a, d);
	cXor(out[1], out[1], ac);
	cXor(out[1], out[1], ad);
	cXor(out[1], out[1], cd);
	// out[2] = ac+bc+bd+1;
	cXor(out[2], ac, bc);
	cXor(out[2], out[2], bd);
	cNot(out[2], out[2]);
	// out[3] = a+b+ab+ad+bc+cd+1;
	cXor(out[3], a, b);
	cXor(out[3], out[3], ab);
	cXor(out[3], out[3], ad);
	cXor(out[3], out[3], bc);
	cXor(out[3], out[3], cd);
	cNot(out[3], out[3]);

	// mul
	a.x2n();
	b.x2n();
	c.x2n();
	d.x2n();
	ab.x2n();
	cd.x2n();
	cAnd(abd, ab, d);
	cAnd(acd, cd, a);
	cAnd(bcd, cd, b);
	cAnd(abc, ab, c);
	abd.x2c();
	acd.x2c();
	bcd.x2c();
	abc.x2c();
	// add
	// out[0] += abd+acd+bcd;
	cXor(out[0], out[0], abd);
	cXor(out[0], out[0], acd);
	cXor(out[0], out[0], bcd);
	// out[1] += abc+acd;
	cXor(out[1], out[1], abc);
	cXor(out[1], out[1], acd);
	// out[2] += abc+bcd;
	cXor(out[2], out[2], abc);
	cXor(out[2], out[2], bcd);
	// out[3] += bcd;
	cXor(out[3], out[3], bcd);

	for (int i=0; i<4; i++) {
		out[i].relin();
		out[i].modSwitch();
	}
	// delete
	a.~CuCtxt();
	b.~CuCtxt();
	c.~CuCtxt();
	d.~CuCtxt();
	ab.~CuCtxt();
	ac.~CuCtxt();
	ad.~CuCtxt();
	bc.~CuCtxt();
	bd.~CuCtxt();
	cd.~CuCtxt();
	abd.~CuCtxt();
	acd.~CuCtxt();
	bcd.~CuCtxt();
	abc.~CuCtxt();
	///////////////////////////////////
	// level up
	// output
	for (int i=0; i<4; i++) {
		out[i].x2z();
		in[i] = out[i].zRep();
		out[i].~CuCtxt();
	}
	return;

}

void Prince::INV_SBOX(ZZX in[64]) {
	for (int i=0; i<64; i++)
		cudhs->coeffReduce(in[i], in[i], level);
//	#pragma omp parallel num_threads(1)
//	#pragma omp parallel num_threads(2)
//	#pragma omp parallel num_threads(3)
	#pragma omp parallel num_threads(numGPUs())
	{
		#pragma omp for nowait
		for (int i=0; i<16; i++)
			_inv_sbox(&in[4*i], omp_get_thread_num(), level);
//			_inv_sbox(&in[4*i], 2, level);
	}
	#pragma omp barrier
	level += 2;
};


void Prince::_inv_sbox(ZZX in[4], int dev, int lvl) {
	CuCtxt a, b, c, d;
	CuCtxt ab, ac, ad, bc, bd, cd;
	CuCtxt abd, acd, bcd, abc;
	CuCtxt out[4];
	///////////////////////////////////
	// input
	a.setLevel(lvl, dev, in[0]);
	b.setLevel(lvl, dev, in[1]);
	c.setLevel(lvl, dev, in[2]);
	d.setLevel(lvl, dev, in[3]);
	a.x2n();
	b.x2n();
	c.x2n();
	d.x2n();
	// mul
	cAnd(ab, a, b);
	cAnd(ac, a, c);
	cAnd(ad, a, d);
	cAnd(bc, b, c);
	cAnd(bd, b, d);
	cAnd(cd, c, d);
	// relin
	ab.relin();
	cd.relin();
	// modswitch
	ab.modSwitch();
	ac.modSwitch();
	ad.modSwitch();
	bc.modSwitch();
	bd.modSwitch();
	cd.modSwitch();
	a.modSwitch();
	b.modSwitch();
	c.modSwitch();
	d.modSwitch();
	///////////////////////////////////
	// level up
	// add
	// out[0] = c+d+ab+bc+bd+cd+1;
	cXor(out[0], c, d);
	cXor(out[0], out[0], ab);
	cXor(out[0], out[0], bc);
	cXor(out[0], out[0], bd);
	cXor(out[0], out[0], cd);
	cNot(out[0], out[0]);
	// out[1] = b+d+ac+bc+bd+cd;
	cXor(out[1], b, d);
	cXor(out[1], out[1], ac);
	cXor(out[1], out[1], bc);
	cXor(out[1], out[1], bd);
	cXor(out[1], out[1], cd);
	// out[2] = ab+ac+bc+bd+1;
	cXor(out[2], ab, ac);
	cXor(out[2], out[2], bc);
	cXor(out[2], out[2], bd);
	cNot(out[2], out[2]);
	// out[3] = a+ab+bc+cd+1;
	cXor(out[3], a, ab);
	cXor(out[3], out[3], bc);
	cXor(out[3], out[3], cd);
	cNot(out[3], out[3]);
	// mul
	a.x2n();
	b.x2n();
	c.x2n();
	d.x2n();
	ab.x2n();
	cd.x2n();
	cAnd(abd, ab, d);
	cAnd(acd, cd, a);
	cAnd(bcd, cd, b);
	cAnd(abc, ab, c);
	abd.x2c();
	acd.x2c();
	bcd.x2c();
	abc.x2c();
	// add
	// out[0] += abc+abd+bcd;
	cXor(out[0], out[0], abc);
	cXor(out[0], out[0], abd);
	cXor(out[0], out[0], bcd);
	// out[1] += acd+bcd;
	cXor(out[1], out[1], acd);
	cXor(out[1], out[1], bcd);
	// out[2] += bcd;
	cXor(out[2], out[2], bcd);
	// out[3] += abd+acd;
	cXor(out[3], out[3], abd);
	cXor(out[3], out[3], acd);

	for (int i=0; i<4; i++) {
		// relin
		out[i].relin();
		// modswitch
		out[i].modSwitch();
	}
	// delete
	a.~CuCtxt();
	b.~CuCtxt();
	c.~CuCtxt();
	d.~CuCtxt();
	ab.~CuCtxt();
	ac.~CuCtxt();
	ad.~CuCtxt();
	bc.~CuCtxt();
	bd.~CuCtxt();
	cd.~CuCtxt();
	abd.~CuCtxt();
	acd.~CuCtxt();
	bcd.~CuCtxt();
	abc.~CuCtxt();
	///////////////////////////////////
	// level up
	// output
	for (int i=0; i<4; i++) {
		out[i].x2z();
		in[i] = out[i].zRep();
		out[i].~CuCtxt();
	}
};

void Prince::addRoundKey(ZZX in[64], ZZX key[64]) {
	for (int i=0; i<64; i++)
		in[i] = in[i] + key[i];
};
void Prince::addRC(ZZX in[64], int round) {
	for (int i=0;i<64;i++)
		in[i] = (in[i] + RC[(round*64)+i]);
};
void Prince::MixColumn(ZZX in[64]) {
	M_p(in);
	ShiftRow(in);
};
void Prince::M_p(ZZX in[64]) {
	ZZX in1[64];
	in1[0] = (in[4]+in[8]+in[12]);
	in1[1] = (in[1]+in[9]+in[13]);
	in1[2] = (in[2]+in[6]+in[14]);
	in1[3] = (in[3]+in[7]+in[11]);
	in1[4] = (in[0]+in[4]+in[8]);
	in1[5] = (in[5]+in[9]+in[13]);
	in1[6] = (in[2]+in[10]+in[14]);
	in1[7] = (in[3]+in[7]+in[15]);
	in1[8] = (in[0]+in[4]+in[12]);
	in1[9] = (in[1]+in[5]+in[9]);
	in1[10] = (in[6]+in[10]+in[14]);
	in1[11] = (in[3]+in[11]+in[15]);
	in1[12] = (in[0]+in[8]+in[12]);
	in1[13] = (in[1]+in[5]+in[13]);
	in1[14] = (in[2]+in[6]+in[10]);
	in1[15] = (in[7]+in[11]+in[15]);

	in1[16] = (in[16]+in[20]+in[24]);
	in1[17] = (in[21]+in[25]+in[29]);
	in1[18] = (in[18]+in[26]+in[30]);
	in1[19] = (in[19]+in[23]+in[31]);
	in1[20] = (in[16]+in[20]+in[28]);
	in1[21] = (in[17]+in[21]+in[25]);
	in1[22] = (in[22]+in[26]+in[30]);
	in1[23] = (in[19]+in[27]+in[31]);
	in1[24] = (in[16]+in[24]+in[28]);
	in1[25] = (in[17]+in[21]+in[29]);
	in1[26] = (in[18]+in[22]+in[26]);
	in1[27] = (in[23]+in[27]+in[31]);
	in1[28] = (in[20]+in[24]+in[28]);
	in1[29] = (in[17]+in[25]+in[29]);
	in1[30] = (in[18]+in[22]+in[30]);
	in1[31] = (in[19]+in[23]+in[27]);

	in1[32] = (in[32]+in[36]+in[40]);
	in1[33] = (in[37]+in[41]+in[45]);
	in1[34] = (in[34]+in[42]+in[46]);
	in1[35] = (in[35]+in[39]+in[47]);
	in1[36] = (in[32]+in[36]+in[44]);
	in1[37] = (in[33]+in[37]+in[41]);
	in1[38] = (in[38]+in[42]+in[46]);
	in1[39] = (in[35]+in[43]+in[47]);
	in1[40] = (in[32]+in[40]+in[44]);
	in1[41] = (in[33]+in[37]+in[45]);
	in1[42] = (in[34]+in[38]+in[42]);
	in1[43] = (in[39]+in[43]+in[47]);
	in1[44] = (in[36]+in[40]+in[44]);
	in1[45] = (in[33]+in[41]+in[45]);
	in1[46] = (in[34]+in[38]+in[46]);
	in1[47] = (in[35]+in[39]+in[43]);

	in1[48] = (in[52]+in[56]+in[60]);
	in1[49] = (in[49]+in[57]+in[61]);
	in1[50] = (in[50]+in[54]+in[62]);
	in1[51] = (in[51]+in[55]+in[59]);
	in1[52] = (in[48]+in[52]+in[56]);
	in1[53] = (in[53]+in[57]+in[61]);
	in1[54] = (in[50]+in[58]+in[62]);
	in1[55] = (in[51]+in[55]+in[63]);
	in1[56] = (in[48]+in[52]+in[60]);
	in1[57] = (in[49]+in[53]+in[57]);
	in1[58] = (in[54]+in[58]+in[62]);
	in1[59] = (in[51]+in[59]+in[63]);
	in1[60] = (in[48]+in[56]+in[60]);
	in1[61] = (in[49]+in[53]+in[61]);
	in1[62] = (in[50]+in[54]+in[58]);
	in1[63] = (in[55]+in[59]+in[63]);

	for (int i=0; i<64; i++)
		in[i] = in1[i];
};
void Prince::ShiftRow(ZZX in[64]) {
	int i = 4;
	ZZX temp[16] = {in[i],in[i+1],in[i+2],in[i+3],in[i+16],in[i+17],in[i+18],in[i+19],in[i+32],in[i+33],in[i+34],in[i+35],in[i+48],in[i+49],in[i+50],in[i+51]};
	in[i] = temp[4];
	in[i+1] = temp[5];
	in[i+2] = temp[6];
	in[i+3] = temp[7];
	in[i+16] = temp[8];
	in[i+17] = temp[9];
	in[i+18] = temp[10];
	in[i+19] = temp[11];
	in[i+32] = temp[12];
	in[i+33] = temp[13];
	in[i+34] = temp[14];
	in[i+35] = temp[15];
	in[i+48] = temp[0];
	in[i+49] = temp[1];
	in[i+50] = temp[2];
	in[i+51] = temp[3];

	i = 8;
	ZZX temp1[16] = {in[i],in[i+1],in[i+2],in[i+3],in[i+16],in[i+17],in[i+18],in[i+19],in[i+32],in[i+33],in[i+34],in[i+35],in[i+48],in[i+49],in[i+50],in[i+51]};
	in[i] = temp1[8];
	in[i+1] = temp1[9];
	in[i+2] = temp1[10];
	in[i+3] = temp1[11];
	in[i+16] = temp1[12];
	in[i+17] = temp1[13];
	in[i+18] = temp1[14];
	in[i+19] = temp1[15];
	in[i+32] = temp1[0];
	in[i+33] = temp1[1];
	in[i+34] = temp1[2];
	in[i+35] = temp1[3];
	in[i+48] = temp1[4];
	in[i+49] = temp1[5];
	in[i+50] = temp1[6];
	in[i+51] = temp1[7];

	i = 12;
	ZZX temp2[16] = {in[i],in[i+1],in[i+2],in[i+3],in[i+16],in[i+17],in[i+18],in[i+19],in[i+32],in[i+33],in[i+34],in[i+35],in[i+48],in[i+49],in[i+50],in[i+51]};
	in[i] = temp2[12];
	in[i+1] = temp2[13];
	in[i+2] = temp2[14];
	in[i+3] = temp2[15];
	in[i+16] = temp2[0];
	in[i+17] = temp2[1];
	in[i+18] = temp2[2];
	in[i+19] = temp2[3];
	in[i+32] = temp2[4];
	in[i+33] = temp2[5];
	in[i+34] = temp2[6];
	in[i+35] = temp2[7];
	in[i+48] = temp2[8];
	in[i+49] = temp2[9];
	in[i+50] = temp2[10];
	in[i+51] = temp2[11];
};
void Prince::inv_MixColumn(ZZX in[64]) {
	inv_ShiftRow(in);
	M_p(in);
};
void Prince::inv_ShiftRow(ZZX in[64]) {
	int i = 4;
	ZZX temp2[16] = {in[i],in[i+1],in[i+2],in[i+3],in[i+16],in[i+17],in[i+18],in[i+19],in[i+32],in[i+33],in[i+34],in[i+35],in[i+48],in[i+49],in[i+50],in[i+51]};
	in[i] = temp2[12];
	in[i+1] = temp2[13];
	in[i+2] = temp2[14];
	in[i+3] = temp2[15];
	in[i+16] = temp2[0];
	in[i+17] = temp2[1];
	in[i+18] = temp2[2];
	in[i+19] = temp2[3];
	in[i+32] = temp2[4];
	in[i+33] = temp2[5];
	in[i+34] = temp2[6];
	in[i+35] = temp2[7];
	in[i+48] = temp2[8];
	in[i+49] = temp2[9];
	in[i+50] = temp2[10];
	in[i+51] = temp2[11];

	i = 8;
	ZZX temp1[16] = {in[i],in[i+1],in[i+2],in[i+3],in[i+16],in[i+17],in[i+18],in[i+19],in[i+32],in[i+33],in[i+34],in[i+35],in[i+48],in[i+49],in[i+50],in[i+51]};
	in[i] = temp1[8];
	in[i+1] = temp1[9];
	in[i+2] = temp1[10];
	in[i+3] = temp1[11];
	in[i+16] = temp1[12];
	in[i+17] = temp1[13];
	in[i+18] = temp1[14];
	in[i+19] = temp1[15];
	in[i+32] = temp1[0];
	in[i+33] = temp1[1];
	in[i+34] = temp1[2];
	in[i+35] = temp1[3];
	in[i+48] = temp1[4];
	in[i+49] = temp1[5];
	in[i+50] = temp1[6];
	in[i+51] = temp1[7];

	i = 12;
	ZZX temp[16] = {in[i],in[i+1],in[i+2],in[i+3],in[i+16],in[i+17],in[i+18],in[i+19],in[i+32],in[i+33],in[i+34],in[i+35],in[i+48],in[i+49],in[i+50],in[i+51]};
	in[i] = temp[4];
	in[i+1] = temp[5];
	in[i+2] = temp[6];
	in[i+3] = temp[7];
	in[i+16] = temp[8];
	in[i+17] = temp[9];
	in[i+18] = temp[10];
	in[i+19] = temp[11];
	in[i+32] = temp[12];
	in[i+33] = temp[13];
	in[i+34] = temp[14];
	in[i+35] = temp[15];
	in[i+48] = temp[0];
	in[i+49] = temp[1];
	in[i+50] = temp[2];
	in[i+51] = temp[3];
};
void Prince::KeyExpansion(ZZX key[64]) {
	ZZX temp[64];
	for (int i=0; i<64; i++)
		temp[i] = key[i];
	key[0] = temp[63];
	for (int i=0; i<63; i++)
		key[i+1] = temp[i];
	key[63] = (key[63]+temp[0]);
};
