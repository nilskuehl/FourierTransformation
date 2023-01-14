__kernel void fourier_transformation(const int N,
                                    __global const float *Yn, __global float *Ck)
{   

        float K = ((float)get_global_id(0)/N) * 10.0f;
        float SumX = 0;
        float SumY = 0;
        
        for (int Ni = 0; Ni < N; Ni++)
        {
            float sample = Yn[get_global_id(0)];
            float Ns = (float)Ni/N;
            if(get_global_id(0) == 4){
            printf("4444444 :::::: %f", sample);
        }
            float Theta = 2*3.14159265359*K*Ns;
            float Magnitude = sample;
            float X = sinf(Theta) * Magnitude;
            float Y = cosf(Theta) * Magnitude;
            SumX += X;
            SumY += Y;
        }
        Ck[get_global_id(0)] = (float) ((1/N) * SumX);


    /*if(get_global_id(0) > N) {
        return;
    }
    float SumX = 0;
    for (int Ni = 0; Ni < N; Ni++)
    {
        float Theta = (float) ((2*3.14159265359*get_global_id(0)*Ni) / N);
        float X = (float) sin(Theta) * Yn[get_global_id(0)];
        SumX += X;
    }
    Ck[get_global_id(0)] = SumX;*/
}