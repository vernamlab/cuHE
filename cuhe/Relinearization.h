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

// Define relinearization keys, methods.

#pragma once
#include <NTL/ZZX.h>
NTL_CLIENT

typedef	unsigned int uint32; // 32-bit
typedef unsigned long int uint64; // 64-bit

namespace cuHE {

// Pre-computation
// convert evalkey to ntt domain
// store ntt(evalkey) in CPU memory efficiently
// maybe GPU memory
void initRelin(ZZX* evalkey);

// Operations
// relinearization
void relinearization(uint64 *dst, uint32 *src, int lvl, int dev,
    cudaStream_t st = 0);

} // namespace cuHE