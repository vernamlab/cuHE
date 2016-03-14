#include "Timer.h"
#include <omp.h>

otimer::otimer() {
	startTime = 0;
	stopTime = 0;
	t = 0;
}
void otimer::start() {
	startTime = omp_get_wtime();
}
void otimer::stop() {
	stopTime = omp_get_wtime();
	t += (stopTime-startTime)/1e-3;
}
void otimer::reset() {
	startTime = 0;
	t = 0;
	stopTime = 0;
}
void otimer::show(string s) {
	cout<<s<<":\t"<<t<<" ms"<<endl;
}
otimer::~otimer() {
}
