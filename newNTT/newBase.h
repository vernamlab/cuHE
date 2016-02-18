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

// This file declares functions that transfer pre-computed data to GPU device
// memory, i.e. initilize NTT, CRT and Barrett Reduction on GPU.
// It also declares CUDA kernel functions that require pre-computed device
// data, such as NTT, CRT, Barrett Reduction, Relinearization and Modulus
// Switching.

#pragma once
typedef unsigned int uint32; // 32-bit unsigned integer
typedef unsigned long int uint64; // 64-bit unsigned integer

namespace cuHE {

/////////////////////////////////////////////////////
//// Transfer Pre-computed Data to GPU Device(s) ////
/////////////////////////////////////////////////////
// generate & copy twiddle factors to device, bind to texture
// "len": length of NTT (a power of 2)
void preload_ntt(int len);
// free and delete allocated memory space
void cleanup_ntt();
// TODO: CRT
// TODO: Barrett

/////////////////////////////////////////////////////////////////////
//// CUDA Kernel Functions That Require Pre-computed Device Data ////
/////////////////////////////////////////////////////////////////////
// NTT kernels with size: 16384, 32768 or 65536
// Each NTT/invrese-NTT conversion consists of 3 kernels:
//   - NTT:
//      ntt_1 -> ntt_2 ->  ntt_3
//   - inverse-INTT:
//     intt_1 -> ntt_2 -> intt_3.
// NTT conversion either takes an array of 32-bit integers,
// or takes a chunck of each elements in an array of large integers
// as input.
// Inverse-NTT conversion gives an array of 32-bit integers (modulo CRT prime
// numbers) as output.
// NTT 16384 kernels
__global__ void ntt_1_16k_ext(uint64 *dst, uint32 *src);
__global__ void ntt_2_16k(uint64 *src);
__global__ void ntt_3_16k(uint64 *dst, uint64 *src);
// TODO: chunk
//__global__ void intt_1_16k(uint64 *dst, uint64 *src);
//__global__ void intt_3_16k_modcrt(uint32 *dst, uint64 *src, int crtidx);
// NTT 32768 kernels
__global__ void ntt_1_32k_ext(uint64 *dst, uint32 *src);
__global__ void ntt_2_32k(uint64 *src);
__global__ void ntt_3_32k(uint64 *dst, uint64 *src);
// NTT 65536 kernels
__global__ void ntt_1_64k_ext(uint64 *dst, uint32 *src);
__global__ void ntt_2_64k(uint64 *src);
__global__ void ntt_3_64k(uint64 *dst, uint64 *src);


// TODO: CRT
// TODO: ModSwitch
// TODO: Barrett
// TODO: Relinearization
// TODO: NTT Domain Operations
// TODO: CRT Domain Operations

} // namespace cuHE
