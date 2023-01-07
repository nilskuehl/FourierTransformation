__kernel void fourier_transformation(__global const int N, __global const int *k,
                                    __global const float *Yn, __global const int *Ck//what else goes here?)
{
    //make sum local var
    for (int Ki = 0; Ki < N; Ki++)
    {
        float K = ((float)Ki/N) * 10.0f;
        float SumX = 0;
        float SumY = 0;
        for (int Ni = 0; Ni < N; Ni++)
        {
            float Ns = (float)Ni/N;
            float Theta = 2*PI*K*Ns;
            float Magnitude = Yn;
            float X = sinf(Theta) * Yn;
            float Y = cosf(Theta) * Yn;
            SumX += X;
            SumY += Y;
        }
        Ck = SumX;
    }

}