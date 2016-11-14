cuHE: Homomorphic and fast
==========================
CUDA Homomorphic Encryption Library (cuHE) is a GPU-accelerated library for [homomorphic encryption][6] (HE) schemes and homomorphic algorithms defined over polynomial rings. cuHE yields an astonishing performance while providing a simple interface that greatly enhances programmer **productivity**. It features both algebraic techniques for homomorphic evalution of circuits and highly optimized code for single-GPU or multi-GPU machines. Develop **high-performance** applications rapidly with cuHE!

The cuHE library is distributed under the terms of the The MIT License (MIT). It is currently created for research purpose only. Several algorithms are implemented as examples and more will follow. Feedback and collaboration of any kind are welcomed.

Features
--------
The library pushes performance to a limit. A number of optimizations such as algebraic techniques for efficient evaluation, memory minimization techniques, memory and stream scheduling and low level CUDA hand-tuned assembly optimizations are included to take full advantage of the mass parallelism and high memory bandwidth GPUs offer. The arithmetic functions constructed to handle very large polynomial operands adopt the Chinese remainder theorem (CRT), the number-theoretic transform (NTT) and Barrett reduction based methods. A few algorithms and routines of the library is described in [this paper][3], along with a performance analysis. More details on arithmetic methods and optimizations regarding HE are explained in our previous papers listed below.

1. Dai, Wei, and Berk Sunar. "cuHE: A Homomorphic Encryption Accelerator Library." Cryptography and Information Security in the Balkans. Springer International Publishing, 2015. 169-186. [[draft][3]] [[Springer][15]]

2. Dai, Wei, Yarkın Doröz, and Berk Sunar. "Accelerating NTRU based homomorphic encryption using GPUs." High Performance Extreme Computing Conference (HPEC), 2014 IEEE. IEEE, 2014. [[draft][4]] [[IEEE Xplore][16]]

3. Dai, Wei, Yarkın Doröz, and Berk Sunar. "Accelerating SWHE Based PIRs Using GPUs." Financial Cryptography and Data Security: FC 2015 International Workshops, BITCOIN, WAHC, and Wearable, San Juan, Puerto Rico, January 30, 2015, Revised Selected Papers. Vol. 8976. Springer, 2015. [[draft][5]] [[Springer][17]]

Examples
--------
Currently available is an implementation of the [Doröz-Hu-Sunar][1] (DHS) somewhat homomorphic encryption (SHE) scheme based on the [Lopez-Tromer-Vaikuntanathan][2] (LTV) scheme.
Several homomorphic applications built on DHS are implemented on GPUs and are included as examples, such as the Prince block cipher and a sorting algorithm.
These examples give an idea of how to program with the cuHE library.

System Requirements
-------------------
1. NVIDIA [CUDA-Enabled GPUs][7] with computation compability 3.0 or higher
2. [NTL][8]: A Library for doing Number Theory 9.3.0 (requires C++11) **NOTE:** to avoid random crashes compile it running ```./configure NTL_EXCEPTIONS=on```
3. The [OpenMP][9] API

Compile
-------

	cd cuhe
	cmake ./
	make

options to cmake command defaults are:

	-DGPU_ARCH:STRING=50
	-DGCC_CUDA_VERSION:STRING=gcc-4.9

Notes for Mac OS X
------------------
On Mac you must use clang instead of gcc. You need to install a version compatible with OpenMP. With brew you can

	brew install clang-omp

Then you must tell Cmake and Cuda that you are using clang-omp

	cd cuhe
	CC=clang-omp CXX=clang-omp++ cmake -DGCC_CUDA_VERSION=clang-omp ./
	make

A Short Tutorial
----------------
To design/implement a homomorphic application/circuit, e.x. the AND of 8 bits. First of all, we need to decide which homomorphic encryption scheme to adopt and set parameters (polynomial ring degree, coefficient sizes in each level of circuit, relinearization strategy) according to some noise analysis process. Let's say we decide to adopt the DHS HE scheme.
```c++
#include "cuHE.h"
void setParameters(int d, int p, int w, int min, int cut, int m);//in "CuHE.h", set parameters
void initCuHE(ZZ *coeffMod_, ZZX modulus); //in "CuHE.h", start pre-computation on GPUs
```

Then we may process some pre-computation of the circuit. When it is time to run the circuit, we suggest to turn on our virtual allocator. Do not turn if off until the circuit is completely done.
```c++
void startAllocator(); //in "CuHE.h", start virtual allocator
void stopAllocator(); //in "CuHE.h", stop virtual allocator
```

The program by default uses a single GPU (device ID 0). To adopt multiple devices, call the function below.
```c++
void multiGPUs(int num); //adopt 'num' GPUs
```

Those are all the initialization steps. To implement any HE scheme or circuit, please check out the provided examples.

Developer
---------
The cuHE library is developed and maintained by [Wei Dai][13] from the [Vernam Group][14] at Worcester Polytechnic Institute.

Acknowledgment
--------------
Funding for this research was in part provided by the US National Science Foundation
CNS Award #1117590 and #1319130.

We want to acknowledge [Andrea Peruffo][18] for improving and debugging the code.

[1]: http://eprint.iacr.org/2014/233/ "DHS14"
[2]: http://eprint.iacr.org/2013/094/ "LTV13"
[3]: http://eprint.iacr.org/2015/818/ "cuHEePrint"
[4]: http://eprint.iacr.org/2014/389/ "Dai2014ePrint"
[5]: http://eprint.iacr.org/2015/462/ "Dai2015ePrint"
[6]: http://en.wikipedia.org/wiki/Homomorphic_encryption "HE"
[7]: http://developer.nvidia.com/cuda-gpus/ "CUDAGPUs"
[8]: http://www.shoup.net/ntl/ "NTL"
[9]: http://openmp.org/wp/ "OpenMP"
[10]: http://en.wikipedia.org/wiki/Chinese_remainder_theorem/ "CRT"
[11]: http://en.wikipedia.org/wiki/Discrete_Fourier_transform_(general)/ "NTT"
[12]: http://en.wikipedia.org/wiki/Barrett_reduction/ "Barrett"
[13]: http://github.com/Veirday "WeiDaiGitHub"
[14]: http://v.wpi.edu/ "VernamGroup"
[15]: http://link.springer.com/chapter/10.1007%2F978-3-319-29172-7_11 "cuHESpringer"
[16]: http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=7041001&tag=1 "Dai2014IEEE"
[17]: http://link.springer.com/chapter/10.1007%2F978-3-662-48051-9_12 "Dai2015aSpringer"
[18]: https://github.com/andreaTP "AndreaPeruffoGitHub"
