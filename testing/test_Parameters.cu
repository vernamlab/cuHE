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

/*! /file test_ModP.cu
 *	/brief Test all inline device mod P = 2^64-2^32+1 operations.
 */

#include "../cuhe/Parameters.h"
#include <stdio.h>

int main () {
	cuHE::setParam(5, 2, 8, 20, 22, 21845);
	cuHE::Param par = cuHE::globalParam();
	printf("Max Coeff: %d bits.\n", par.logCoeffMax);
	printf("Min Coeff: %d bits.\n", par.logCoeffMin);
	printf("Cut Coeff: %d bits.\n", par.logCoeffCut);
	printf("%d crt primes with %d bits.\n", par.numCrtPrime, par.logCrtPrime);

	cuHE::resetParam();

	cuHE::setParam(10, 2, 16, 20, 15, 8191);
	par = cuHE::globalParam();
	printf("Max Coeff: %d bits.\n", par.logCoeffMax);
	printf("Min Coeff: %d bits.\n", par.logCoeffMin);
	printf("Cut Coeff: %d bits.\n", par.logCoeffCut);
	printf("%d crt primes with %d bits.\n", par.numCrtPrime, par.logCrtPrime);

	return 0;
}