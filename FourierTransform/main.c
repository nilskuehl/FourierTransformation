#include <stdio.h>
#include <math.h>
#include "raylib.h"
#define RAYGUI_IMPLEMENTATION
#include "include/raygui.h"
#include <sys/time.h>


#define NUM_SAMPLES 1024
#define PI 3.14159265359
#define min(X,Y) ((X) < (Y) ? (X) : (Y))
#define max(X,Y) ((X) > (Y) ? (X) : (Y))

typedef struct state
{
    float Signal[NUM_SAMPLES];
    float FourierTransform[NUM_SAMPLES];
} state;


SlowFourierTransform(float *TimeDomain, float *FreqDomain, int Size)
{
    double sum = 0;
    // Start measuring time
    struct timeval begin, end;
    gettimeofday(&begin, 0);

    // 0Hz - 10Hz.
    for (int Ki = 0; Ki < Size; Ki++)
    {
        float K = ((float)Ki/Size) * 10.0f;
        float SumX = 0;
        float SumY = 0;
        
        for (int N = 0; N < Size; N++)
        {
            float Sample = TimeDomain[N];
            float Ns = (float)N/Size;
            float Theta = 2*PI*K*Ns;
            float Magnitude = Sample;
            float X = sinf(Theta) * Magnitude;
            float Y = cosf(Theta) * Magnitude;
            SumX += X;
            SumY += Y;
        }
        FreqDomain[Ki] = SumX;
    }
    // Stop measuring time and calculate the elapsed time
    gettimeofday(&end, 0);
    long seconds = end.tv_sec - begin.tv_sec;
    long microseconds = end.tv_usec - begin.tv_usec;
    double elapsed = seconds + microseconds*1e-6;
    
    printf("Result: %.20f\n", sum);
    
    printf("Time measured: %.3f seconds.\n", elapsed);
    
}

Update(state *State)
{
    static int Frequenz1 = 4;
    static bool Freqeunz1_Edit = false;
    static float Amp1 = 1;
    
    static int Frequenz2 = 0;
    static bool Frequenz2_Edit = false;
    static float Amp2 = 1;
    
    static int Frequenz3 = 0;
    static float Amp3 = 1;
    static bool Frequenz3_Edit = false;

    if (GuiSpinner((Rectangle){ 600, 15, 80, 30 }, NULL, &Frequenz1, 0, 10, Freqeunz1_Edit)) 
        Freqeunz1_Edit = !Freqeunz1_Edit;
    if (GuiSpinner((Rectangle){ 700, 15, 80, 30 }, NULL, &Frequenz2, 0, 10, Frequenz2_Edit)) 
        Frequenz2_Edit = !Frequenz2_Edit;
    if (GuiSpinner((Rectangle){ 800, 15, 80, 30 }, NULL, &Frequenz3, 0, 10, Frequenz3_Edit)) 
        Frequenz3_Edit = !Frequenz3_Edit;
    
    for (int N = 0; N < NUM_SAMPLES; N++)
    {
        State->Signal[N] = 0;
        State->Signal[N] += Amp1 * sinf(2 * PI * Frequenz1 * ((float)N / NUM_SAMPLES));
        State->Signal[N] += Amp2 * sinf(2 * PI * Frequenz2 * ((float)N / NUM_SAMPLES));
        State->Signal[N] += Amp3 * sinf(2 * PI * Frequenz3 * ((float)N / NUM_SAMPLES));
        State->Signal[N] /= 3;
    }
}


Draw(state *State)
{
    Color SignalColor = ColorAlpha(RED, 0.7f);
    Color OverlapColor = ColorAlpha(BLUE, 0.7f);
    Color BaselineColor = ColorAlpha(WHITE, 0.7f);
    Color CutColor = ColorAlpha(GREEN, 0.9f);
    Color Sum2DColor = ColorAlpha(YELLOW, 0.9f);
    float MouseX = (float)GetMouseX();
    printf("MOUSE VALUE: %f",MouseX);
    
    // Draw the signal
    int SignalBaselineY = 150;
    int SignalHeight = 50;
    DrawLine(0, SignalBaselineY, NUM_SAMPLES, SignalBaselineY, BaselineColor);
    for (int N = 0; N < NUM_SAMPLES; N++)
    {
        float Sample = State->Signal[N];
        DrawPixel(N, SignalBaselineY + Sample*SignalHeight, SignalColor);
    }
    
    // Cut Frequency
    int FrequencyLineY = 720;
    float MaxCutFrequency = 10.0f;
    float CutFrequency = (MouseX/NUM_SAMPLES) * MaxCutFrequency;
    
    // Draw Frequency Line
    DrawLine(0, FrequencyLineY, NUM_SAMPLES, FrequencyLineY, BaselineColor);


    int MaxCutFrequencyLines = NUM_SAMPLES/(int)MaxCutFrequency;
    for(int N = 0; N <= NUM_SAMPLES; N+=MaxCutFrequencyLines){
        DrawLine(N, 
            FrequencyLineY - SignalHeight, 
            N, 
            FrequencyLineY, 
            WHITE);
        
        if(N>=102){
            DrawText(TextFormat("%dHz", N/100),
             N-10,
             730,
             15,
             WHITE);
        }
    }

    if(MouseX >= 0 && MouseX<= NUM_SAMPLES){
        int CutX = 0;
        int CutStride = max((NUM_SAMPLES/CutFrequency), 1);
        float Summation[NUM_SAMPLES] = {0};

        while (CutStride > 0 && CutX < NUM_SAMPLES)
        {
        DrawLine(CutX, 
                 SignalBaselineY - SignalHeight, 
                 CutX, 
                 SignalBaselineY + SignalHeight, 
                 CutColor);
        // Draw overlap.
        for (int N = 0; (N < CutStride) && (N+CutX < NUM_SAMPLES); N++)
        {
            float Sample = State->Signal[N + CutX];
            DrawPixel(CutStride + N,
                      SignalBaselineY + Sample*SignalHeight,
                      SignalColor);
            Summation[N] += Sample;
        }
        CutX += CutStride;
        }

        // Draw Summation.
        for (int N = 0; N < min(CutStride, NUM_SAMPLES); N++)
        {
            float Sample = Summation[N];
            DrawPixel(CutStride + N,
                    SignalBaselineY + Sample*SignalHeight,
                    OverlapColor);
        }
    }
   
    SlowFourierTransform(&State->Signal[0], &State->FourierTransform[0], NUM_SAMPLES);
    
    // Draw the frequency domain
    for (int N = 0; N < NUM_SAMPLES; N++)
    {
        float F = State->FourierTransform[N];
        DrawPixel(N,
                  650 - (F * 1),
                  PURPLE);
    }
    if(0 <= CutFrequency && CutFrequency <= 10){
        DrawText(TextFormat("Cut Frequency: %.2fHz", CutFrequency),
             15,
             15,
             20,
             WHITE);
    }else{
        DrawText(TextFormat("Out of range"),
             15,
             15,
             20,
             WHITE);
    }
    
}



int main() 
{

    const float screen_width = NUM_SAMPLES;
    const float screen_height = 768;
    InitWindow(screen_width, screen_height, "FourierTransform");
    SetTargetFPS(60);
    InitAudioDevice();
    
    state *State = malloc(sizeof(state));
    for (int N = 0; N < NUM_SAMPLES; N++)
    {
        State->FourierTransform[N] = 0;
    }
    
    while(!WindowShouldClose())
    {
        Update(State);
        BeginDrawing();
        ClearBackground(BLACK);
        Draw(State);
        EndDrawing();
    }
    
    CloseAudioDevice();
    CloseWindow();
    return 0;

}