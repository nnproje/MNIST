#include "Activations.h"
float ActivPar=0.8;
////////////////////////////////////////////
Matrix* softmax(Matrix* z)
{
    float MAX=z->MaxElement();
    Matrix* z_stable = z->sub(MAX);
    float epsi=1e-10;
    int r=z->Rows();
    int c=z->Columns();
    Matrix* s = new Matrix(1,c);  //summation of exponents
    Matrix* A = new Matrix(r,c);
    for(int i=0;i<r;i++)
    {
        for(int j=0;j<c;j++)
        {
            A->access(i,j) = exp(z_stable->access(i,j));
        }
    }
    s=A->SUM("row");
    for(int ii=0;ii<r;ii++)
    {
        for(int jj=0;jj<c;jj++)
        {
            A->access(ii,jj) = A->access(ii,jj)/(s->access(0,jj)+epsi);
        }
    }
    z_stable->DELETE();
    s->DELETE();
    return A;
}
////////////////////////////////////////////
Matrix* sigmoid(Matrix* z)
{
    int r=z->Rows();
    int c=z->Columns();
    Matrix* A=new Matrix(r,c);
    for(int i=0;i<r;i++)
    {
        for(int j=0;j<c;j++)
        {
            A->access(i,j) = 1/(1+exp(-1*z->access(i,j)));
        }
    }
    return A;
}
///////////////////////////////////////////
Matrix* mytanh(Matrix* z)
{
    int r=z->Rows();
    int c=z->Columns();
    Matrix* A=new Matrix(r,c);
    for(int i=0;i<r;i++)
    {
        for(int j=0;j<c;j++)
        {
            A->access(i,j)=tanh(z->access(i,j));
        }
    }
    return A;
}
////////////////////////////////////////
Matrix* relu(Matrix* z)
{
    int r=z->Rows();
    int c=z->Columns();
    Matrix* A=new Matrix(r,c);
    for(int i=0;i<r;i++)
    {
        for(int j=0;j<c;j++)
        {
            if(z->access(i,j)>=0)
            A->access(i,j)=z->access(i,j);
            else
            A->access(i,j)=0;
        }
    }
    return A;
}
//////////////////////////////////////////
Matrix* leakyRelu(Matrix* z)
{
    int r=z->Rows();
    int c=z->Columns();
    Matrix* A=new Matrix(r,c);
    for(int i=0;i<r;i++)
    {
        for(int j=0;j<c;j++)
        {
            if(z->access(i,j)>=0)
            A->access(i,j)=z->access(i,j);
            else
            A->access(i,j)=0.01*z->access(i,j);
        }
    }
    return A;
}
//////////////////////////////////////////
Matrix* Linear(Matrix* z)
{
    return z;
}
//////////////////////////////////////////
Matrix* satLinear(Matrix* z)
{
    int r=z->Rows();
    int c=z->Columns();
    Matrix* A=new Matrix(r,c);
    for(int i=0; i<r; i++)
    {
        for(int j=0; j<c; j++)
        {
            if(z->access(i,j)<=-1)
                 A->access(i,j)=-1;
            else if(z->access(i,j)>-1 && z->access(i,j)<1)
                 A->access(i,j)=1*z->access(i,j);
            else
                 A->access(i,j)=1;
        }
    }
    return A;
}
//////////////////////////////////////////
Matrix* satLinear2(Matrix* z, float maxErr)
{
    float slope;

    if(maxErr>=ActivPar)
        slope=1;
    else
        slope=2;

    int r=z->Rows();
    int c=z->Columns();
    Matrix* A=new Matrix(r,c);

    for(int i=0; i<r; i++)
    {
        for(int j=0; j<c; j++)
        {
            if(z->access(i,j)<=-1/slope)
                 A->access(i,j)=-1;
            else if(z->access(i,j)>-1/slope && z->access(i,j)<1/slope)
                 A->access(i,j)=slope*z->access(i,j);
            else
                 A->access(i,j)=1;
        }
    }

    return A;
}
//////////////////////////////////////////
Matrix* satLinear3(Matrix* z,float maxErr)
{
    float slope;

    if(maxErr>=ActivPar)
        slope=1;
    else
        slope=2;

    int r=z->Rows();
    int c=z->Columns();
    Matrix* A=new Matrix(r,c);

    for(int i=0; i<r; i++)
    {
        for(int j=0; j<c; j++)
        {
            if(z->access(i,j)<=-1/slope)
                 A->access(i,j)=-1;
            else if(z->access(i,j)>-1/slope && z->access(i,j)<1/slope)
                 A->access(i,j)=slope*z->access(i,j);
            else
                 A->access(i,j)=1;
        }
    }

    return A;
}
//////////////////////////////////////////
//////////////////////////////////////////
Matrix* dsoftmax(Matrix* z)                        //da=a(1-a)
{
    int r=z->Rows();
    int c=z->Columns();
    Matrix* a=softmax(z);
    Matrix* da=new Matrix(r,c);
    da=a->mul(-1);
    da=da->add(1);
    da=da->mul(a);
    return da;
}

//////////////////////////////////////////
Matrix* dsigmoid(Matrix* z)                      //da=a(1-a)
{
    int r=z->Rows();
    int c=z->Columns();
    Matrix* a=sigmoid(z);
    Matrix* da=new Matrix(r,c);
    da=a->mul(-1);
    da=da->add(1);
    da=da->mul(a);
    return da;
}
/////////////////////////////////////////
Matrix* dtanh(Matrix* z)                                 //da=1-a^2
{
    int r=z->Rows();
    int c=z->Columns();
    Matrix* a=mytanh(z);
    Matrix* da=new Matrix(r,c);
    da=a->mul(a);
    da=da->mul(-1);
    da=da->add(1);
    return da;
}
/////////////////////////////////////////
Matrix* drelu(Matrix* z)
{
    int r=z->Rows();
    int c=z->Columns();
    Matrix* da=new Matrix(r,c);
    for(int i=0;i<r;i++)
    {
        for (int j=0;j<c;j++)
        {
            if(z->access(i,j)<0)
            da->access(i,j)=0;
            else
            da->access(i,j)=1;
        }
    }
    return da;
}
//////////////////////////////////////////
Matrix* dleakyRelu(Matrix* z)
{
    int r=z->Rows();
    int c=z->Columns();
    Matrix* da=new Matrix(r,c);
    for(int i=0;i<r;i++)
    {
        for (int j=0;j<c;j++)
        {
            if(z->access(i,j)<0)
            da->access(i,j)=0.01;
            else
            da->access(i,j)=1;
        }
    }
    return da;
}

////////////////////////////////////////
Matrix* dLinear(Matrix* z)
{
    int r=z->Rows();
    int c=z->Columns();
    Matrix* da=new Matrix(r,c,1);
    return da;
}
/////////////////////////////////////////
Matrix* dsatLinear(Matrix* z)
{
    int r=z->Rows();
    int c=z->Columns();
    Matrix* da=new Matrix(r,c);
    for(int i=0;i<r;i++)
    {
        for (int j=0;j<c;j++)
        {
            if(z->access(i,j)<=-1)
                 da->access(i,j)=0;
            else if(z->access(i,j)>-1 && z->access(i,j)<1)
                 da->access(i,j)=1;
            else
                 da->access(i,j)=0;
        }
    }
    return da;
}
//////////////////////////////////////////////////
Matrix* dsatLinear2(Matrix* z,float maxErr)
{
    float slope;

    if(maxErr>=ActivPar)
        slope=1;
    else
        slope=2;

    int r=z->Rows();
    int c=z->Columns();
    Matrix* da=new Matrix(r,c);
    for(int i=0;i<r;i++)
    {
        for (int j=0;j<c;j++)
        {
            if(z->access(i,j)<=-1/slope)
                 da->access(i,j)=0;

            else if(z->access(i,j)>-1/slope && z->access(i,j)<1/slope)
                 da->access(i,j)=slope;

            else
                 da->access(i,j)=0;
        }
    }
    return da;
}
//////////////////////////////////////////////////
Matrix* dsatLinear3(Matrix* z, float maxErr)
{
    float slope;

    if(maxErr>=ActivPar)
        slope=1;
    else
        slope=2;

    int r=z->Rows();
    int c=z->Columns();
    Matrix* da=new Matrix(r,c);
    for(int i=0;i<r;i++)
    {
        for (int j=0;j<c;j++)
        {
            if(z->access(i,j)<=-1/slope)
                 da->access(i,j)=0;

            else if(z->access(i,j)>-1/slope && z->access(i,j)<1/slope)
                 da->access(i,j)=slope;

            else
                 da->access(i,j)=0;
        }
    }
    return da;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* activ(Matrix*z,string activation)
{
    Matrix* A;
    if (activation == "relu")
        A = relu(z);
    else if (activation == "leakyRelu")
        A = leakyRelu(z);
    else if (activation == "tanh")
        A = mytanh(z);
    else if (activation == "sigmoid")
        A = sigmoid(z);
    else if (activation == "softmax")
        A = softmax(z);
    else if (activation == "satLinear")
        A = satLinear(z);
    else if (activation == "Linear")
        A = Linear(z);

    return A;
}
/////////////////////////////////////////////////////////////////
Matrix* dactiv(Matrix*z,string activation)
{
    Matrix* df;
    if (activation == "softmax")
        df = dsoftmax(z);
    if (activation == "sigmoid")
        df = dsigmoid(z);
    else if (activation == "tanh")
        df = dtanh(z);
    else if (activation == "relu")
        df = drelu(z);
    else if (activation == "leakyRelu")
        df = dleakyRelu(z);
    else if (activation == "satLinear")
        df = dsatLinear(z);
    else if (activation == "Linear")
        df = dLinear(z);

     return df;
}
