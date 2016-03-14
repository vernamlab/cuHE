#define	M_SIZE					8191
#define	polynomialDegree		8190
#define sortSize	32
#if sortSize == 4
	#define circuitDepth  12	//(11+1) size : 4
	#define	numBitsPerPrime			20
#elif sortSize == 8
	#define circuitDepth  13	//(12+1) size : 8
	#define	numBitsPerPrime			20
#elif sortSize == 16
	#define circuitDepth  14	//(13+1) size : 16
	#define	numBitsPerPrime			21
#elif sortSize == 32
	#define circuitDepth  15	//(14+1) size : 32
	#define	numBitsPerPrime			22
#elif sortSize == 64
	#define circuitDepth  16	//(15+1) size : 64
#endif
#define	numPrimesPerLevel		1
#define	relinBits				16

/**	should put in circuit files !!!
* 0: all devices have a copy of ekNtt
* 1: 1 device compute and store, 1 device purely for storage
*/
#define	relinScheme	0
