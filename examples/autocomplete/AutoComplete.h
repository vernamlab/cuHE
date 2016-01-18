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

/*!	/file AutoComplete.h
 *	/brief Homomorphic AutoComplete algorithm.
 */

#pragma once

#include "HEScheme.h"
#include "Batcher.h"

class AutoComplete {
public:
	AutoComplete(ParamID id, int numentries, int numoutchars, int logmsg);
	~AutoComplete();
	void run();
	void saveDictionary();
	void loadDictionary();
	void prepDictionary();
	void userSend();
	double server();
	void userReceive();

	ZZX *zin, *zout;
	ZZX **zzxdict;
	ZZ *zdict;

	char **chardict;
	CuPtxt **dict; // [numOut][numIn]
	CuCtxt **in; // [numDev][numIn]
	CuCtxt *out; // [numOut]
	CuCtxt **temp; // [numDev][numIn]

	int numIn, numOut;
	int numEntries, numOutBits;

	CuDHS *he;
	myCRT *batcher;
	cudaStream_t *stream_;

/*
	void loadDictionary();
	void randDictionary();
	void tempDictionary();
	size_t setDict();
	size_t userUp();
	double serverCrtAdd();
	double serverNttAdd();
	size_t userDown();
*/
};

// temp return words
void getLine(char *dst, int numChars, unsigned index);
