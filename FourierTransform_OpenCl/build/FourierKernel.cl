__kernel void fourier_transformation(const int N,
                                    __global const float *Yn, __global float *Ck)
{   

    //printf("%d\n", get_global_id(0));

    float K = ((float) get_global_id(0)/N) * 10.0f;
    float SumX = 0;
    float SumY = 0;
        
    for (int i = 0; i < N; i++)
    {
        float Ns = (float)i/N;
        float Theta = 2*3.14159265359*K*Ns;
        float X = sin(Theta) * Yn[get_global_id(0)];
        float Y = cos(Theta) * Yn[get_global_id(0)];
        SumX += X;
        SumY += Y;
    }
    
    Ck[get_global_id(0)] = SumX;
}