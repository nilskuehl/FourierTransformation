__kernel void fourier_transformation(__global const int N, __global const int *k,
                                    __global const float *Yn, //what else goes here?)
{
    //make sum local var
    for (int Ki = 0; Ki < N; Ki++)
    {
        float K = ((f32)Ki/N) * 10.0f;
        float SumX = 0;
        float SumY = 0;
        for (int Ni = 0; Ni < N; Ni++)
        {
            f32 Sample = TimeDomain[Ni];
            f32 Ns = (f32)Ni/N;
            f32 Theta = 2*PI*K*Ns;
            f32 Magnitude = Sample;
            f32 X = sinf(Theta) * Magnitude;
            f32 Y = cosf(Theta) * Magnitude;
            SumX += X;
            SumY += Y;
        }
        FreqDomain[Ki] = SumX;
    }

}