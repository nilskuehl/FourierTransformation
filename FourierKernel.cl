__kernel void fourier_transformation(const int N,
                                    __global const float *Yn, __global float *Ck)
{
    if(get_global_id(0) >= N)
        return;
    for (int Ki = 0; Ki < N; Ki++)
    {
        float K = (Ki/N) * 10.0f;
        float SumX = 0;
        float SumY = 0;
        for (int Ni = 0; Ni < N; Ni++)
        {
            float Ns = (float) (Ni/get_global_id(0));
            float Theta = 2*3.14*K*Ns;
            float X = sin(Theta) * Yn[get_global_id(0)];
            float Y = cos(Theta) * Yn[get_global_id(0)];
            SumX += X;
            SumY += Y;
        }
        Ck[get_global_id(0)] = SumX;
    }

}