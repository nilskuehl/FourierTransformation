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
    static int Freq1 = 4;
    static bool Freq1_Edit = false;
    static float Amp1 = 1;
    
    static int Freq2 = 0;
    static bool Freq2_Edit = false;
    static float Amp2 = 1;
    
    static int Freq3 = 0;
    static float Amp3 = 1;
    static bool Freq3_Edit = false;

    if (GuiSpinner((Rectangle){ 600, 15, 80, 30 }, NULL, &Freq1, 0, 10, Freq1_Edit)) 
        Freq1_Edit = !Freq1_Edit;
    if (GuiSpinner((Rectangle){ 700, 15, 80, 30 }, NULL, &Freq2, 0, 10, Freq2_Edit)) 
        Freq2_Edit = !Freq2_Edit;
    if (GuiSpinner((Rectangle){ 800, 15, 80, 30 }, NULL, &Freq3, 0, 10, Freq3_Edit)) 
        Freq3_Edit = !Freq3_Edit;
    
    for (int N = 0; N < NUM_SAMPLES; N++)
    {
        State->Signal[N] = 0;
        State->Signal[N] += Amp1 * sinf(2 * PI * Freq1 * ((float)N / NUM_SAMPLES));
        State->Signal[N] += Amp2 * sinf(2 * PI * Freq2 * ((float)N / NUM_SAMPLES));
        State->Signal[N] += Amp3 * sinf(2 * PI * Freq3 * ((float)N / NUM_SAMPLES));
        State->Signal[N] /= 3;
    }
}


Draw(state *State)
{
    Color SignalCol = ColorAlpha(RED, 0.7f);
    Color OverlapCol = ColorAlpha(BLUE, 0.7f);
    Color BaselineCol = ColorAlpha(WHITE, 0.7f);
    Color CutCol = ColorAlpha(GREEN, 0.9f);
    Color Sum2DCol = ColorAlpha(YELLOW, 0.9f);
    float MouseX = (float)GetMouseX();
    
    // Draw the signal
    int SignalBaselineY = 150;
    int SignalHeight = 50;
    DrawLine(0, SignalBaselineY, NUM_SAMPLES, SignalBaselineY, BaselineCol);
    for (int N = 0; N < NUM_SAMPLES; N++)
    {
        float Sample = State->Signal[N];
        DrawPixel(N, SignalBaselineY + Sample*SignalHeight, SignalCol);
    }
    
    // Cut Frequency
    int FrequencyLineY = 720;
    float MaxCutFreq = 10.0f;
    float CutFreq = (MouseX/NUM_SAMPLES) * MaxCutFreq;
    
    // Draw Frequency Line
    DrawLine(0, FrequencyLineY, NUM_SAMPLES, FrequencyLineY, BaselineCol);

   //TODO:
    int CutX = 0;
    int CutStride = max((NUM_SAMPLES/CutFreq), 1);
    float Summation[NUM_SAMPLES] = {0};
    int MaxCutFreqLines = NUM_SAMPLES/(int)MaxCutFreq;
    printf("%d", CutStride);
    for(int N = 0; N <= NUM_SAMPLES; N+=MaxCutFreqLines){
        DrawLine(N, 
            FrequencyLineY - SignalHeight, 
            N, 
            FrequencyLineY + SignalHeight, 
            WHITE);

        DrawText(TextFormat("%.2fHz", N),
             N,
             730,
             20,
             WHITE);
    }
    
    

    // Draw Cuts
    //int CutX = 0;
    //int CutStride = max((NUM_SAMPLES/CutFreq), 1);
    //float Summation[NUM_SAMPLES] = {0};
    while (CutStride > 0 && CutX < NUM_SAMPLES)
    {
        DrawLine(CutX, 
                 SignalBaselineY - SignalHeight, 
                 CutX, 
                 SignalBaselineY + SignalHeight, 
                 CutCol);
        // Draw overlap.
        for (int N = 0; (N < CutStride) && (N+CutX < NUM_SAMPLES); N++)
        {
            float Sample = State->Signal[N + CutX];
            DrawPixel(CutStride + N,
                      SignalBaselineY + Sample*SignalHeight,
                      SignalCol);
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
                  OverlapCol);
    }
    
    SlowFourierTransform(&State->Signal[0], &State->FourierTransform[0], NUM_SAMPLES);
    
    // Draw the frequency domain
    for (int N = 0; N < NUM_SAMPLES; N++)
    {
        float F = State->FourierTransform[N];
        DrawPixel(N,
                  650 - (F * 1),
                  Sum2DCol);
    }
    
    DrawText(TextFormat("Cut Frequency: %.2fHz", CutFreq),
             15,
             15,
             20,
             WHITE);
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