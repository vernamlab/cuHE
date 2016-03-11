#include "DHS.h"

class Prince {
public:
	Prince();
	~Prince();
	void run();
	void heSetup();
	void setMessage(int *m);
	void setKeys(int *k0, int *k1);

private:
	CuDHS *cudhs;
	int level;
	int round;
	ZZX bits[64];
	ZZX key0[64];
	ZZX key1[64];
	void princeEncrypt(ZZX tar[64], ZZX k0[64], ZZX k1[64]);

	void SBOX(ZZX in[64]);
	void _sbox(ZZX in[4], int dev, int lvl);

	void INV_SBOX(ZZX in[64]);
	void _inv_sbox(ZZX in[4], int dev, int lvl);

	void check(int rd);
	void addRoundKey(ZZX in[64], ZZX key[64]);
	void addRC(ZZX in[64], int round);
	void MixColumn(ZZX in[64]);
	void M_p(ZZX in[64]);
	void ShiftRow(ZZX in[64]);
	void inv_MixColumn(ZZX in[64]);
	void inv_ShiftRow(ZZX in[64]);
	void KeyExpansion(ZZX key[64]);
};
