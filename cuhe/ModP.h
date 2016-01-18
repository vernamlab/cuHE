/* 
 *	The MIT License (MIT)
 *	Copyright (c) 2013-2015 Wei Dai
 *
 *	Permission is hereby granted, free of charge, to any person obtaining a copy
 *	of this software and associated documentation files (the "Software"), to deal
 *	in the Software without restriction, including without limitation the rights
 *	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *	copies of the Software, and to permit persons to whom the Software is
 *	furnished to do so, subject to the following conditions:
 *
 *	The above copyright notice and this permission notice shall be included in
 *	all copies or substantial portions of the Software.
 *
 *	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *	THE SOFTWARE.
 */

/*!	/file ModP.h
 *	/brief	P = 2^64-2^32+1, all device inline modulo P operations.
 *			This provides fast NTT operations.
 */

#pragma once

typedef	unsigned int		uint32;	// 32-bit
typedef unsigned long int	uint64;	// 64-bit

namespace cuHE {

/** All mod P operations are declared below ( P = 2^64-2^32+1 ). */
__device__ __host__ __inline__ void _uint96_modP(uint32 *x);
// 96-bit integer mod P

__device__ __host__ __inline__ void _uint128_modP(uint32 *x);
// 128-bit integer mod P

__device__ __host__ __inline__ void _uint160_modP(uint32 *x);
// 160-bit integer mod P

__device__ __host__ __inline__ void _uint192_modP(uint32 *x);
// 192-bit integer mod P

__device__ __host__ __inline__ void _uint224_modP(uint32 *x);
// 224-bit integer mod P

__device__ __host__ __inline__ uint64 _ls_modP(uint64 x, int l);
// return x<<l mod P, l = [0,7]*[0,7]*3

__device__ __host__ __inline__ uint64 _add_modP(uint64 x, uint64 y);
// return x+y mod P

__device__ __host__ __inline__ uint64 _sub_modP(uint64 x, uint64 y);
// return x-y mod P

__device__ __host__ __inline__ uint64 _mul_modP(uint64 x, uint64 y);
// return x*y mod P

__device__ __host__ __inline__ uint64 _pow_modP(uint64 x, int e);
// return x^e mod P
/** End of declarations */

__device__ __host__ __inline__
void _uint96_modP(uint32 *x) {
	register uint32 bit;
	asm("add.cc.u32 %0,%0,%1;" : "+r"(x[1]) : "r"(x[2]));
	asm("addc.u32 %0,%1,%2;" : "=r"(bit) : "r"(0), "r"(0));
	asm("sub.cc.u32 %0,%0,%1;" : "+r"(x[0]) : "r"(x[2]));
	asm("subc.cc.u32 %0,%0,%1;" : "+r"(x[1]) : "r"(0));
	asm("subc.u32 %0,%0,%1;" : "+r"(bit) : "r"(0));
	uint64 *t = (uint64 *)x;
	if (bit == 0x1) {
		(*t) += 0xffffffff;
	}
	if ((*t) >= 0xffffffff00000001) {
		(*t) -= 0xffffffff00000001;
	}
}

__device__ __host__ __inline__
void _uint128_modP(uint32 *x) {
	register uint32 bit;
	asm("add.cc.u32 %0,%0,%1;" : "+r"(x[1]) : "r"(x[2]));
	asm("addc.u32 %0,%1,%2;" : "=r"(bit) : "r"(0), "r"(0));
	asm("add.cc.u32 %0,%0,%1;" : "+r"(x[2]) : "r"(x[3]));
	asm("addc.u32 %0,%1,%2;" : "=r"(x[3]) : "r"(0), "r"(0));
	asm("sub.cc.u32 %0,%0,%1;" : "+r"(x[0]) : "r"(x[2]));
	asm("subc.cc.u32 %0,%0,%1;" : "+r"(x[1]) : "r"(x[3]));
	asm("subc.u32 %0,%0,%1;" : "+r"(bit) : "r"(0));
	if (bit == 0xffffffff) {
		asm("sub.cc.u32 %0,%0,%1;" : "+r"(x[0]) : "r"(0xffffffff));
		asm("subc.u32 %0,%0,%1;" : "+r"(x[1]) : "r"(0));
		bit = 0;
	}
	uint64 *t = (uint64 *)x;
	if ((*t) >= 0xffffffff00000001) {
		(*t) -= 0xffffffff00000001;
	}
	if (bit == 0x1) {
		(*t) += 0xffffffff;
	}
	if ((*t) >= 0xffffffff00000001) {
		(*t) -= 0xffffffff00000001;
	}
}

__device__ __host__ __inline__
void _uint160_modP(uint32 *x) {
	register uint32 bit;
	asm("add.cc.u32 %0,%0,%1;" : "+r"(x[1]) : "r"(x[2]));
	asm("addc.u32 %0,%1,%2;" : "=r"(bit) : "r"(0), "r"(0));
	asm("add.cc.u32 %0,%0,%1;" : "+r"(x[2]) : "r"(x[3]));
	asm("addc.u32 %0,%1,%2;" : "=r"(x[3]) : "r"(0), "r"(0));
	asm("sub.cc.u32 %0,%0,%1;" : "+r"(x[0]) : "r"(x[2]));
	asm("subc.cc.u32 %0,%0,%1;" : "+r"(x[1]) : "r"(x[3]));
	asm("subc.u32 %0,%0,%1;" : "+r"(bit) : "r"(0));
	if (bit == 0xffffffff) {
		asm("sub.cc.u32 %0,%0,%1;" : "+r"(x[0]) : "r"(0xffffffff));
		asm("subc.u32 %0,%0,%1;" : "+r"(x[1]) : "r"(0));
		bit = 0;
	}
	asm("sub.cc.u32 %0,%0,%1;" : "+r"(x[1]) : "r"(x[4]));
	asm("subc.u32 %0,%0,%1;" : "+r"(bit) : "r"(0));
	if (bit == 0xffffffff) {
		asm("sub.cc.u32 %0,%0,%1;" : "+r"(x[0]) : "r"(0xFFFFFFFF));
		asm("subc.u32 %0,%0,%1;" : "+r"(x[1]) : "r"(0));
		bit = 0;
	}
	uint64 *t = (uint64 *)x;
	if ((*t) >= 0xffffffff00000001) {
		(*t) -= 0xffffffff00000001;
	}
	if (bit == 0x1) {
		(*t) += 0xffffffff;
	}
	if ((*t) >= 0xffffffff00000001) {
		(*t) -= 0xffffffff00000001;
	}
}

__device__ __host__ __inline__
void _uint192_modP(uint32 *x) {
	register uint32 bit;
	asm("add.cc.u32 %0,%0,%1;" : "+r"(x[1]) : "r"(x[2]));
	asm("addc.u32 %0,%1,%2;" : "=r"(bit) : "r"(0), "r"(0));
	asm("add.cc.u32 %0,%0,%1;" : "+r"(x[2]) : "r"(x[3]));
	asm("addc.u32 %0,%1,%2;" : "=r"(x[3]) : "r"(0), "r"(0));
	asm("sub.cc.u32 %0,%0,%1;" : "+r"(x[0]) : "r"(x[2]));
	asm("subc.cc.u32 %0,%0,%1;" : "+r"(x[1]) : "r"(x[3]));
	asm("subc.u32 %0,%0,%1;" : "+r"(bit) : "r"(0));
	if (bit == 0xffffffff) {
		asm("sub.cc.u32 %0,%0,%1;" : "+r"(x[0]) : "r"(0xffffffff));
		asm("subc.u32 %0,%0,%1;" : "+r"(x[1]) : "r"(0));
		bit = 0;
	}
	asm("sub.cc.u32 %0,%0,%1;" : "+r"(x[1]) : "r"(x[4]));
	asm("subc.u32 %0,%0,%1;" : "+r"(bit) : "r"(0));
	if (bit == 0xffffffff) {
		asm("sub.cc.u32 %0,%0,%1;" : "+r"(x[0]) : "r"(0xffffffff));
		asm("subc.u32 %0,%0,%1;" : "+r"(x[1]) : "r"(0));
		bit = 0;
	}
	asm("add.cc.u32 %0,%0,%1;" : "+r"(x[0]) : "r"(x[5]));
	asm("addc.cc.u32 %0,%0,%1;" : "+r"(x[1]) : "r"(0));
	asm("addc.u32 %0,%0,%1;" : "+r"(bit) : "r"(0));
	asm("sub.cc.u32 %0,%0,%1;" : "+r"(x[1]) : "r"(x[5]));
	asm("subc.u32 %0,%0,%1;" : "+r"(bit) : "r"(0));
	if (bit == 0xffffffff) {
		asm("sub.cc.u32 %0,%0,%1;" : "+r"(x[0]) : "r"(0xffffffff));
		asm("subc.u32 %0,%0,%1;" : "+r"(x[1]) : "r"(0));
		bit = 0;
	}
	uint64 *t = (uint64 *)x;
	if ((*t) >= 0xffffffff00000001) {
		(*t) -= 0xffffffff00000001;
	}
	while (bit > 0) {
		(*t) += 0xffffffff;
		if ((*t) >= 0xffffffff00000001) {
			(*t) -= 0xffffffff00000001;
		}
		bit --;
	}
}

__device__ __host__ __inline__
void _uint224_modP(uint32 *x) {
	register uint32 bit;
	asm("add.cc.u32 %0,%0,%1;" : "+r"(x[1]) : "r"(x[2]));
	asm("addc.u32 %0,%1,%2;" : "=r"(bit) : "r"(0), "r"(0));
	asm("add.cc.u32 %0,%0,%1;" : "+r"(x[2]) : "r"(x[3]));
	asm("addc.u32 %0,%1,%2;" : "=r"(x[3]) : "r"(0), "r"(0));
	asm("sub.cc.u32 %0,%0,%1;" : "+r"(x[0]) : "r"(x[2]));
	asm("subc.cc.u32 %0,%0,%1;" : "+r"(x[1]) : "r"(x[3]));
	asm("subc.u32 %0,%0,%1;" : "+r"(bit) : "r"(0));
	if (bit == 0xffffffff) {
		asm("sub.cc.u32 %0,%0,%1;" : "+r"(x[0]) : "r"(0xffffffff));
		asm("subc.u32 %0,%0,%1;" : "+r"(x[1]) : "r"(0));
		bit = 0;
	}
	asm("sub.cc.u32 %0,%0,%1;" : "+r"(x[1]) : "r"(x[4]));
	asm("subc.u32 %0,%0,%1;" : "+r"(bit) : "r"(0));
	if (bit == 0xffffffff) {
		asm("sub.cc.u32 %0,%0,%1;" : "+r"(x[0]) : "r"(0xffffffff));
		asm("subc.u32 %0,%0,%1;" : "+r"(x[1]) : "r"(0));
		bit = 0;
	}
	asm("add.cc.u32 %0,%0,%1;" : "+r"(x[0]) : "r"(x[5]));
	asm("addc.cc.u32 %0,%0,%1;" : "+r"(x[1]) : "r"(0));
	asm("addc.u32 %0,%0,%1;" : "+r"(bit) : "r"(0));
	asm("sub.cc.u32 %0,%0,%1;" : "+r"(x[1]) : "r"(x[5]));
	asm("subc.u32 %0,%0,%1;" : "+r"(bit) : "r"(0));
	if (bit == 0xffffffff) {
		asm("sub.cc.u32 %0,%0,%1;" : "+r"(x[0]) : "r"(0xffffffff));
		asm("subc.u32 %0,%0,%1;" : "+r"(x[1]) : "r"(0));
		bit = 0;
	}
	asm("add.cc.u32 %0,%0,%1;" : "+r"(x[0]) : "r"(x[6]));
	asm("addc.cc.u32 %0,%0,%1;" : "+r"(x[1]) : "r"(0));
	asm("addc.u32 %0,%0,%1;" : "+r"(bit) : "r"(0));
	uint64 *t = (uint64 *)x;
	if ((*t) >= 0xffffffff00000001) {
		(*t) -= 0xffffffff00000001;
	}
	while (bit>0) {
		(*t) += 0xffffffff;
		if ((*t) >= 0xffffffff00000001) {
			(*t) -= 0xffffffff00000001;
		}
		bit--;
	}
}

__device__ __host__ __inline__
uint64 _ls_modP(uint64 x, int l) {
	register uint64 ret, tx = x;
	register uint32 buff[7];
	switch(l){
	case (0):	// 2 words
		ret = tx;
		break;
	case (3):
	case (6):
	case (9):
	case (12):
	case (15):
	case (18):
	case (21):
	case (24):
	case (27):
	case (30):	// 3 words
		buff[2] = (uint32)(tx>>(64-l));
		buff[1] = (uint32)(tx>>(32-l));
		buff[0] = (uint32)(tx<<l);
		_uint96_modP(buff);
		ret = buff[1];
		ret <<= 32;
		ret += buff[0];
		break;
	case (36):
	case (42):
	case (45):
	case (48):
	case (54):
	case (60):
	case (63):	// 4 words
		buff[3] = (uint32)(tx>>(96-l));
		buff[2] = (uint32)(tx>>(64-l));
		buff[1] = (uint32)(tx<<(l-32));
		buff[0] = 0;
		_uint128_modP(buff);
		ret = buff[1];
		ret <<= 32;
		ret += buff[0];
		break;
	case (72):
	case (75):
	case (84):
	case (90):	// 5 words
		buff[4] = (uint32)(tx>>(128-l));
		buff[3] = (uint32)(tx>>(96-l));
		buff[2] = (uint32)(tx<<(l-64));
		buff[1] = 0;
		buff[0] = 0;
		_uint160_modP(buff);
		ret = buff[1];
		ret <<= 32;
		ret += buff[0];
		break;
	case (105):
	case (108):
	case (126):	// 6 words
		buff[5] = (uint32)(tx>>(160-l));
		buff[4] = (uint32)(tx>>(128-l));
		buff[3] = (uint32)(tx<<(l-96));
		buff[2] = 0;
		buff[1] = 0;
		buff[0] = 0;
		_uint192_modP(buff);
		ret = buff[1];
		ret <<= 32;
		ret += buff[0];
		break;
	case (147):	// 7 words
		buff[6] = (uint32)(tx>>(192-l));
		buff[5] = (uint32)(tx>>(160-l));
		buff[4] = (uint32)(tx<<(l-128));
		buff[3] = 0;
		buff[2] = 0;
		buff[1] = 0;
		buff[0] = 0;
		_uint224_modP(buff);
		ret = buff[1];
		ret <<= 32;
		ret += buff[0];
		break;
	}
	return ret;
}

__device__ __host__ __inline__
uint64 _add_modP(uint64 x, uint64 y) {
	register uint64 ret;
	ret = x+y;
	if (ret < x) {
		ret += 0xffffffff;
	}
	if (ret >= 0xffffffff00000001) {
		ret -= 0xffffffff00000001;
	}
	return ret;
}

__device__ __host__ __inline__
uint64 _sub_modP(uint64 x, uint64 y) {
	register uint64 ret;
	ret = x-y;
	if (ret > x) {
		ret -= 0xffffffff;
	}
	return ret;
}

__device__ __host__ __inline__
uint64 _mul_modP(uint64 x, uint64 y) {
	register uint32 mul[4];
	register uint32 a[2], b[2];
	a[0] = *((uint32 *)&x);
	a[1] = *((uint32 *)&x+1);
	b[0] = *((uint32 *)&y);
	b[1] = *((uint32 *)&y+1);
	// 64-bit * 64-bit
	register uint32 temp[2];
	register uint64 ret;
	asm("mul.lo.u32 %0,%1,%2;" : "=r"(mul[0]) : "r"(a[0]), "r"(b[0]));
	asm("mul.hi.u32 %0,%1,%2;" : "=r"(mul[1]) : "r"(a[0]), "r"(b[0]));
	asm("mul.lo.u32 %0,%1,%2;" : "+r"(mul[2]) : "r"(a[1]), "r"(b[1]));
	asm("mul.hi.u32 %0,%1,%2;" : "+r"(mul[3]) : "r"(a[1]), "r"(b[1]));
	asm("mul.lo.u32 %0,%1,%2;" : "=r"(temp[0]) : "r"(a[0]), "r"(b[1]));
	asm("mul.hi.u32 %0,%1,%2;" : "=r"(temp[1]) : "r"(a[0]), "r"(b[1]));
	asm("add.cc.u32 %0,%0,%1;" : "+r"(mul[1]) : "r"(temp[0]));
	asm("addc.cc.u32 %0,%0,%1;" : "+r"(mul[2]) : "r"(temp[1]));
	asm("addc.u32 %0,%0,%1;" : "+r"(mul[3]) : "r"(0));
	asm("mul.lo.u32 %0,%1,%2;" : "+r"(temp[0]) : "r"(a[1]), "r"(b[0]));
	asm("mul.hi.u32 %0,%1,%2;" : "+r"(temp[1]) : "r"(a[1]), "r"(b[0]));
	asm("add.cc.u32 %0,%0,%1;" : "+r"(mul[1]) : "r"(temp[0]));
	asm("addc.cc.u32 %0,%0,%1;" : "+r"(mul[2]) : "r"(temp[1]));
	asm("addc.u32 %0,%0,%1;" : "+r"(mul[3]) : "r"(0));
	// 128-bit mod P
	_uint128_modP(mul);
	ret = mul[1];
	ret <<= 32;
	ret += mul[0];
	return ret;
}

__device__ __host__ __inline__
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

} // end cuHE
