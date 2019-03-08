#include "Volume.h"
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
{
    delete vol;
}
////////////////////////////////////
void Volume::DELETE()
{
    delete vol;
}
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
////////////////////////////////////
