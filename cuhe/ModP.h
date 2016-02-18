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

// P = 2^64-2^32+1 = 0xffff ffff 0000 0001.
// This file contains all modulo P arithmetic device inline code;
// lays the foundation of fast NTT conversions or NTT domain operations.
// Attention: error occurs when compiled inline

#pragma once
typedef	unsigned int uint32; // 32-bit unsigned integer
typedef unsigned long int	uint64; // 64-bit unsigned integer
#define valP 0xffffffff00000001 // special prime number for NTT conversions
#define uint32Max 0xffffffff // maximum value of 32-bit unsigned integer

namespace cuHE {

// All mod P arithmetci functions are declared below...
// return ( x << l mod P ), where l has the form "3*{0~7}*{0~7}"
__inline__ __device__
uint64 _ls_modP(uint64 x, int l);
// return ( x + y mod P )
__inline__ __device__
uint64 _add_modP(uint64 x, uint64 y);
// return ( x - y mod P )
__inline__ __device__
uint64 _sub_modP(uint64 x, uint64 y);
// return ( x * y mod P )
__inline__ __device__
uint64 _mul_modP(uint64 x, uint64 y);
// return ( x ^ e mod P )
__inline__ __device__
uint64 _pow_modP(uint64 x, int e);
// x[0~1] <-- x[0~2] mod P
__inline__ __device__
void _uint96_modP(uint32 *x);
// x[0~1] <-- x[0~3] mod P
__inline__ __device__
void _uint128_modP(uint32 *x);
// x[0~1] <-- x[0~4] mod P
__inline__ __device__
void _uint160_modP(uint32 *x);
// x[0~1] <-- x[0~5] mod P
__inline__ __device__
void _uint192_modP(uint32 *x);
// x[0~1] <-- x[0~6] mod P
__inline__ __device__
void _uint224_modP(uint32 *x);

// All mod P arithmetic functions are implemented below...
__inline__ __device__
void _uint96_modP(uint32 *x) {
	asm("{\n\t"
			".reg .s32 b;\n\t"
			".reg .pred p;\n\t"
			"add.cc.u32 %1, %1, %2;\n\t"
			"addc.s32 b, 0, 0;\n\t"
			"sub.cc.u32 %0, %0, %2;\n\t"
			"subc.cc.u32 %1, %1, 0;\n\t"
			"subc.s32 b, b, 0;\n\t"
			"setp.eq.s32 p, b, 1;\n\t"
			"@p add.cc.u32 %0, %0, 0xffffffff;\n\t"
			"@p addc.u32 %1, %1, 0;\n\t"
			"}"
			: "+r"(x[0]), "+r"(x[1])
			: "r"(x[2]));
}
__inline__ __device__
void _uint128_modP(uint32 *x) {
	_uint96_modP(x);
	asm("{\n\t"
			".reg .s32 b;\n\t"
			".reg .pred p;\n\t"
			"sub.cc.u32 %0, %0, %2;\n\t"
	  	"subc.cc.u32 %1, %1, 0;\n\t"
	  	"subc.u32 b, 0, 0;\n\t"
			"setp.eq.s32 p, b, -1;\n\t"
			"@p sub.cc.u32 %0, %0, 0xffffffff;\n\t"
			"@p subc.u32 %1, %1, 0;\n\t"
			"}"
	  	: "+r"(x[0]), "+r"(x[1])
	  	: "r"(x[3]));
}
__inline__ __device__
void _uint160_modP(uint32 *x) {
	_uint128_modP(x);
	asm("{\n\t"
			".reg .s32 b;\n\t"
	  	".reg .pred p;\n\t"
			"sub.cc.u32 %1, %1, %2;\n\t"
			"subc.u32 b, 0, 0;\n\t"
			"setp.eq.s32 p, b, -1;\n\t"
			"@p sub.cc.u32 %0, %0, 0xffffffff;\n\t"
			"@p subc.u32 %1, %1, 0;\n\t"
			"}"
	  	: "+r"(x[0]), "+r"(x[1])
	  	: "r"(x[4]));
}
__inline__ __device__
void _uint192_modP(uint32 *x) {
	_uint160_modP(x);
	asm("{\n\t"
			".reg .s32 b;\n\t"
			".reg .pred p;\n\t"
			"add.cc.u32 %0, %0, %2;\n\t"
			"addc.cc.u32 %1, %1, 0;\n\t"
			"addc.u32 b, 0, 0;\n\t"
			"sub.cc.u32 %1, %1, %2;\n\t"
			"subc.u32 b, b, 0;\n\t"
			"setp.eq.s32 p, b, -1;\n\t"
			"@p sub.cc.u32 %0, %0, 0xffffffff;\n\t"
			"@p subc.u32 %1, %1, 0;\n\t"
			"}"
			: "+r"(x[0]), "+r"(x[1])
			: "r"(x[5]));
}
__inline__ __device__
void _uint224_modP(uint32 *x) {
	_uint192_modP(x);
	asm("{\n\t"
			".reg .s32 b;\n\t"
			".reg .pred p;\n\t"
			"add.cc.u32 %0, %0, %2;\n\t"
			"addc.cc.u32 %1, %1, 0;\n\t"
			"addc.u32 b, 0, 0;\n\t"
			"setp.eq.s32 p, b, 1;\n\t"
			"@p add.cc.u32 %0, %0, 0xffffffff;\n\t"
			"@p addc.u32 %1, %1, 0;\n\t"
			"}"
			: "+r"(x[0]), "+r"(x[1])
			: "r"(x[6]));
}
__inline__ __device__
uint64 _ls_modP(uint64 x, int l) {
	register uint64 tx = x;
	register uint32 buff[7];
	switch(l){
		// 2 words
		case (0):
			buff[0] = (uint32)tx;
			buff[1] = (uint32)(tx>>32);
			break;
		// 3 words
		case (3):
		case (6):
		case (9):
		case (12):
		case (15):
		case (18):
		case (21):
		case (24):
		case (27):
		case (30):
			buff[2] = (uint32)(tx>>(64-l));
			buff[1] = (uint32)(tx>>(32-l));
			buff[0] = (uint32)(tx<<l);
			_uint96_modP(buff);
			break;
		// 4 words
		case (36):
		case (42):
		case (45):
		case (48):
		case (54):
		case (60):
		case (63):
			buff[3] = (uint32)(tx>>(96-l));
			buff[2] = (uint32)(tx>>(64-l));
			buff[1] = (uint32)(tx<<(l-32));
			buff[0] = 0;
			_uint128_modP(buff);
			break;
		// 5 words
		case (72):
		case (75):
		case (84):
		case (90):
			buff[4] = (uint32)(tx>>(128-l));
			buff[3] = (uint32)(tx>>(96-l));
			buff[2] = (uint32)(tx<<(l-64));
			buff[1] = 0;
			buff[0] = 0;
			_uint160_modP(buff);
			break;
		// 6 words
		case (105):
		case (108):
		case (126):
			buff[5] = (uint32)(tx>>(160-l));
			buff[4] = (uint32)(tx>>(128-l));
			buff[3] = (uint32)(tx<<(l-96));
			buff[2] = 0;
			buff[1] = 0;
			buff[0] = 0;
			_uint192_modP(buff);
			break;
		// 7 words
		case (147):
			buff[6] = (uint32)(tx>>(192-l));
			buff[5] = (uint32)(tx>>(160-l));
			buff[4] = (uint32)(tx<<(l-128));
			buff[3] = 0;
			buff[2] = 0;
			buff[1] = 0;
			buff[0] = 0;
			_uint224_modP(buff);
			break;
	}
	if (*(uint64 *)buff > valP)
		*(uint64 *)buff -= valP;
	return *(uint64 *)buff;
}
__inline__ __device__
uint64 _add_modP(uint64 x, uint64 y) {
	register uint64 ret;
	ret = x+y;
	if (ret < x)
		ret += uint32Max;
	if (ret >= valP)
		ret -= valP;
	return ret;
}
__inline__ __device__
uint64 _sub_modP(uint64 x, uint64 y) {
	register uint64 ret;
	ret = x-y;
	if (ret > x)
		ret -= uint32Max;
	return ret;
}
__inline__ __device__
uint64 _mul_modP(uint64 x, uint64 y) {
	volatile register uint32 mul[4]; // NEVER REMOVE VOLATILE HERE!!!
	// 128-bit = 64-bit * 64-bit
	asm("mul.lo.u32 %0, %4, %6;\n\t"
      "mul.hi.u32 %1, %4, %6;\n\t"
      "mul.lo.u32 %2, %5, %7;\n\t"
      "mul.hi.u32 %3, %5, %7;\n\t"
      "mad.lo.cc.u32 %1, %4, %7, %1;\n\t"
      "madc.hi.cc.u32 %2, %4, %7, %2;\n\t"
      "addc.u32 %3, %3, 0;\n\t"
      "mad.lo.cc.u32 %1, %5, %6, %1;\n\t"
      "madc.hi.cc.u32 %2, %5, %6, %2;\n\t"
      "addc.u32 %3, %3, 0;\n\t"
      : "+r"(mul[0]), "+r"(mul[1]), "+r"(mul[2]), "+r"(mul[3])
      : "r"(((uint32 *)&x)[0]), "r"(((uint32 *)&x)[1]),
        "r"(((uint32 *)&y)[0]), "r"(((uint32 *)&y)[1]));
	// 128-bit mod P
	asm("{\n\t"
   		".reg .s32 b;\n\t"
			".reg .pred p;\n\t"
			"add.cc.u32 %1, %1, %2;\n\t"
			"addc.s32 b, 0, 0;\n\t"
			"sub.cc.u32 %0, %0, %2;\n\t"
			"subc.cc.u32 %1, %1, 0;\n\t"
			"subc.s32 b, b, 0;\n\t"
			"setp.eq.s32 p, b, 1;\n\t"
			"@p add.cc.u32 %0, %0, 0xffffffff;\n\t"
			"@p addc.cc.u32 %1, %1, 0;\n\t"
			"sub.cc.u32 %0, %0, %3;\n\t"
	  	"subc.cc.u32 %1, %1, 0;\n\t"
	  	"subc.u32 b, 0, 0;\n\t"
			"setp.eq.s32 p, b, -1;\n\t"
			"@p sub.cc.u32 %0, %0, 0xffffffff;\n\t"
			"@p subc.u32 %1, %1, 0;\n\t"
      "}"
      : "+r"(mul[0]), "+r"(mul[1])
      : "r"(mul[2]), "r"(mul[3]));
	if (*(uint64 *)mul > valP)
		*(uint64 *)mul -= valP;
	return *(uint64 *)mul;
}
__inline__ __device__
uint64 _pow_modP(uint64 x, int e) {
	register uint64 ret = 1, tx = x;
	while (e > 0) {
		if (e&0x1 == 0x1) {
			ret = _mul_modP(ret, tx);
		}
		else {
			tx = _mul_modP(tx, tx);
		}
		e >>= 1;
	}
	return ret;
}

} // namespace cuHE
