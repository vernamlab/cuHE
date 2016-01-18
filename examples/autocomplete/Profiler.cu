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

#include "Profiler.h"

namespace cuHE {

// OmpTimer
OmpTimer::OmpTimer() {
	start_ = 0;
	stop_ = 0;
	t_ = 0;
}
OmpTimer::~OmpTimer() { reset();}
void OmpTimer::start() { start_ = omp_get_wtime();}
void OmpTimer::stop() {
	stop_ = omp_get_wtime();
	t_ += (stop_-start_)/1e-3;
}
void OmpTimer::reset() {
	start_ = 0;
	t_ = 0;
	stop_ = 0;
}
double OmpTimer::t() { return t_;}
void OmpTimer::show(string s) {	cout<<s<<":\t"<<t_<<" ms"<<endl;}

} // end cuHE