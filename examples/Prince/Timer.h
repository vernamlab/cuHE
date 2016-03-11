
#include <iostream>
using namespace std;

// openMP timer
class otimer{
public:
	otimer();
	void start();
	void stop();
	void reset();
	void show(string s);
	~otimer();
private:
	double startTime;
	double stopTime;
	double t;
};
