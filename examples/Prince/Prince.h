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

#include "DHS.h"

class Prince {
public:
	Prince();
	~Prince();
	void run();
	void heSetup();
private:
	CuDHS *cudhs;
	int level;
	int round;
	ZZX bits[64];
	ZZX key0[64];
	ZZX key1[64];
	
	void setMessage(int *m);
	void setKeys(int *k0, int *k1);
	void princeEncrypt(ZZX tar[64], ZZX k0[64], ZZX k1[64]);
	void check(int rd);

	void addRoundKey(ZZX in[64], ZZX key[64]);
	void addRC(ZZX in[64], int round);
	void MixColumn(ZZX in[64]);
	void M_p(ZZX in[64]);
	void ShiftRow(ZZX in[64]);
	void inv_MixColumn(ZZX in[64]);
	void inv_ShiftRow(ZZX in[64]);
	void KeyExpansion(ZZX key[64]);
	void SBOX(ZZX in[64]);
	void _sbox(ZZX in[4], int dev, int lvl);
	void INV_SBOX(ZZX in[64]);
	void _inv_sbox(ZZX in[4], int dev, int lvl);
};
