#pragma once
#ifndef VOLUME_HEADER
#define VOLUME_HEADER
#include <vector>
#include "Matrix.h"

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
	Matrix* & operator [](int i);
	void resize(int n);
	void pushMat(Matrix* Mat);
	int size();
	void print();
};
////////////////////////////////////
Volume::Volume()
{
	vol = NULL;
	SIZE = 0;
}
////////////////////////////////////
Volume::Volume(int n)
{

	SIZE = n;
	vol = new vector<Matrix*>(n);
}
////////////////////////////////////
Volume::~Volume()
{}
////////////////////////////////////
void Volume::pushMat(Matrix* Mat)
{
	vol->push_back(Mat);
	SIZE++;
}
////////////////////////////////////
Matrix* & Volume::operator[](int i)
{
	return (*vol)[i];
}
////////////////////////////////////
void Volume::resize(int n)
{
	SIZE = n;
	vol = new vector<Matrix*>(n);
}
////////////////////////////////////
int Volume::size()
{
	return SIZE;
}
////////////////////////////////////
void Volume::print()
{
	cout << "no. 2D matrices = " << SIZE << endl;
	for (int i = 0; i < SIZE; i++)
	{
		cout << endl;
		(*vol)[i]->print();
		cout << endl;
	}
}
#endif // !VOLUME_HEADER
