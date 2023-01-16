__kernel void fourier_transformation(const int N,
                                    __global const float *Yn, __global float *Ck)
{   
    float k = ((float) get_global_id(0)/N) * 10.0f;
    float sum = 0;
        
    for (int i = 0; i < N; i++)
    {
        float nS = (float)i/N;
        float phi = 2*3.14159265359*k*nS;
        float X = sin(phi) * Yn[get_global_id(0)];
        float Y = cos(phi) * Yn[get_global_id(0)];
        sum += X;
    }

    Ck[get_global_id(0)] = sum;
}