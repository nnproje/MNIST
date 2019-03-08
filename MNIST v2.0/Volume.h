#pragma once
#ifndef VOLUME_HEADER
#define VOLUME_HEADER
#include <vector>
#include "Matrix.h"
#include <stdlib.h>
typedef matrix<float> Matrix;

class Volume
{
private:
	vector<Matrix*>* vol; //Each entry of *vol holds an address of a Matrix
	int SIZE;
public:
	Volume();
	Volume(int n);
	~Volume();
	void DELETE();
	Matrix* & operator [](int i);
	void resize(int n);
	void pushMat(Matrix* Mat);
	int size();
	void print();
};


#endif // !VOLUME_HEADER
