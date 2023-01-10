#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
//#include "platform.h"
//#include "raylib.h"
//#include "raygui.h"
#include <sys/time.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define CL_PROGRAM_FILE "FourierKernel.cl"
#define KERNEL_NAME "fourier_transformation"
#define N 120
#define T 20
#define PI 3.14159265359

#define NUM_SAMPLES 1024
#define min(X,Y) ((X) < (Y) ? (X) : (Y))
#define max(X,Y) ((X) > (Y) ? (X) : (Y))
#define RAYGUI_SUPPORT_ICONS
#define RAYGUI_IMPLEMENTATION
/*
typedef struct state
{
    f32 Signal[NUM_SAMPLES];
    f32 FourierTransform[NUM_SAMPLES];
} state;*/

struct benchmark_result {
    double transfer_time;
    double calc_time;
    int errors;
};

//Calculate the value of a singal at a given point
float calculateSignal(int x) {
    return (float) sin(x) + cos(x);
} 

int transform(cl_device_id device, char *program_text, char *kernel_name, struct benchmark_result *result) {
    
    //Context
    cl_context context;
    context = clCreateContext(0, 1, &device, NULL, NULL, NULL);
    if (!context) {
        fprintf(stderr, "Failed to create context!\n");
        return -1;
    }

    //Command Queue
    cl_command_queue commands = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, NULL);
    if (!commands) {
        fprintf(stderr, "Failed to createg queue!\n");
        return -1;
    }

    //generate clProgram Object
    int err;
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&program_text, NULL, &err);
    if (err < 0) {
        fprintf(stderr, "Failed to create cl program!\n");
        return -1;
    }

    //build Program with headers used for calculation
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];

        fprintf(stderr, "Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        fprintf(stderr, "%s\n", buffer);
        return -1;
    }

    //Create Kernels
    cl_kernel kernel = clCreateKernel(program, kernel_name, &err);
    if (!kernel || err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create kernel!\n");
        return -1;
    }

    //Prepare signal data on Host machine
    float *h_Yn = malloc(N*sizeof(float));
    float *h_Ck = malloc(N*sizeof(float));
    for (size_t i = 0; i < N - 1; i++) {
        h_Yn[i] = calculateSignal(i);
        h_Ck[i] = -1;
    }

    //create array buffer in the device memory
    
    cl_mem d_Y = clCreateBuffer(context, CL_MEM_READ_ONLY, N*sizeof(float), NULL, &err);
    if (!d_Y) {
        fprintf(stderr, "Failed to allocate device memory\n");
        return -1;
    }
    //create array buffer in the device memory
    cl_mem d_Ck = clCreateBuffer(context, CL_MEM_READ_WRITE, N*sizeof(float), NULL, NULL);
    if (!d_Ck) {
        fprintf(stderr, "Failed to allocate device memory\n");
        return -1;
    }

    // Copy data to device
    double transfer_sec = 0.0;
    cl_event prof_event;

    err = clEnqueueWriteBuffer(commands, d_Y, CL_TRUE, 0, N, h_Yn, 0, NULL, &prof_event);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to write to device!\n");
        return -1;
    }
    // Wait for transfer to finish
    clFinish(commands);

    // Set the arguments to our compute kernel
    int n = N;
    int k = T;
    err = clSetKernelArg(kernel, 0, sizeof(int), &n);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_Y);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_Ck);
    //... what else goes herre has to be added when .cl file is thought out
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to set kernel arguments!\n");
        return -1;
    }

    // Execute the kernel
    size_t global_size[] = {N};
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, global_size, NULL, 0, NULL, &prof_event);
    if (err) {
        fprintf(stderr, "Failed to execute kernel!\n");
        return -1;
    }
    clFinish(commands);

    // Copy Result Data Back
    err = clEnqueueReadBuffer(commands, d_Ck, CL_TRUE, 0, N, h_Ck, 0, NULL, &prof_event);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to read result!\n");
        return -1;
    }
    clFinish(commands);

    result->transfer_time = transfer_sec;
    result->calc_time = 0;
    result->errors = 0;

    //release memory on device and host
    //free Program, context, etc.
    /*clReleaseMemObject(d_Y);
    clReleaseMemObject(d_Ck);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
    free(h_Yn);
    free(h_Ck);*/
    return 0;

}

/*
internal void 
SlowFourierTransform(f32 *TimeDomain, f32 *FreqDomain, i32 Size)
{
    double sum = 0;
    // Start measuring time
    struct timeval begin, end;
    gettimeofday(&begin, 0);

    // 0Hz - 10Hz.
    for (i32 Ki = 0; Ki < Size; Ki++)
    {
        f32 K = ((f32)Ki/Size) * 10.0f;
        f32 SumX = 0;
        f32 SumY = 0;
        for (i32 N = 0; N < Size; N++)
        {
            f32 Sample = TimeDomain[N];
            f32 Ns = (f32)N/Size;
            f32 Theta = 2*PI*K*Ns;
            f32 Magnitude = Sample;
            f32 X = sinf(Theta) * Magnitude;
            f32 Y = cosf(Theta) * Magnitude;
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

internal void 
Update(state *State)
{
    static i32 Freq1 = 4;
    static bool Freq1_Edit = false;
    static f32 Amp1 = 1;
    
    static i32 Freq2 = 0;
    static bool Freq2_Edit = false;
    static f32 Amp2 = 1;
    
    static i32 Freq3 = 0;
    static f32 Amp3 = 1;
    static bool Freq3_Edit = false;
    
    if (GuiSpinner((Rectangle){ 600, 15, 80, 30 }, NULL, &Freq1, 0, 10, Freq1_Edit)) 
        Freq1_Edit = !Freq1_Edit;
    if (GuiSpinner((Rectangle){ 700, 15, 80, 30 }, NULL, &Freq2, 0, 10, Freq2_Edit)) 
        Freq2_Edit = !Freq2_Edit;
    if (GuiSpinner((Rectangle){ 800, 15, 80, 30 }, NULL, &Freq3, 0, 10, Freq3_Edit)) 
        Freq3_Edit = !Freq3_Edit;
    
    for (i32 N = 0; N < NUM_SAMPLES; N++)
    {
        State->Signal[N] = 0;
        State->Signal[N] += Amp1 * sinf(2 * PI * Freq1 * ((f32)N / NUM_SAMPLES));
        State->Signal[N] += Amp2 * sinf(2 * PI * Freq2 * ((f32)N / NUM_SAMPLES));
        State->Signal[N] += Amp3 * sinf(2 * PI * Freq3 * ((f32)N / NUM_SAMPLES));
        State->Signal[N] /= 3;
    }
}

internal void
Draw(state *State)
{
    Color SignalCol = ColorAlpha(RED, 0.7f);
    Color OverlapCol = ColorAlpha(BLUE, 0.7f);
    Color BaselineCol = ColorAlpha(GRAY, 0.7f);
    Color CutCol = ColorAlpha(GREEN, 0.9f);
    Color Sum2DCol = ColorAlpha(YELLOW, 0.9f);
    f32 MouseX = (f32)GetMouseX();
    
    // Draw the signal
    i32 SignalBaselineY = 150;
    i32 SignalHeight = 50;
    DrawLine(0, SignalBaselineY, NUM_SAMPLES, SignalBaselineY, BaselineCol);
    for (i32 N = 0; N < NUM_SAMPLES; N++)
    {
        f32 Sample = State->Signal[N];
        DrawPixel(N, SignalBaselineY + Sample*SignalHeight, SignalCol);
    }
    
    // Cut Frequency
    f32 MaxCutFreq = 10.0f;
    f32 CutFreq = (MouseX/NUM_SAMPLES) * MaxCutFreq;
    
    // Draw Cuts
    i32 CutX = 0;
    i32 CutStride = max((NUM_SAMPLES/CutFreq), 1);
    f32 Summation[NUM_SAMPLES] = {0};
    while (CutStride > 0 && CutX < NUM_SAMPLES)
    {
        DrawLine(CutX, 
                 SignalBaselineY - SignalHeight, 
                 CutX, 
                 SignalBaselineY + SignalHeight, 
                 CutCol);
        // Draw overlap.
        for (i32 N = 0; (N < CutStride) && (N+CutX < NUM_SAMPLES); N++)
        {
            f32 Sample = State->Signal[N + CutX];
            DrawPixel(CutStride + N,
                      SignalBaselineY + Sample*SignalHeight,
                      SignalCol);
            Summation[N] += Sample;
        }
        CutX += CutStride;
    }
    
    // Draw Summation.
    for (i32 N = 0; N < min(CutStride, NUM_SAMPLES); N++)
    {
        f32 Sample = Summation[N];
        DrawPixel(CutStride + N,
                  SignalBaselineY + Sample*SignalHeight,
                  OverlapCol);
    }
    /*
    // Draw circular wrap.
    i32 CircleOriginX = 512;
    i32 CircleOriginY = 400;
    f32 SumX = 0;
    f32 SumY = 0;
    for (i32 N = 0; N < NUM_SAMPLES; N++)
    {
        f32 Sample = State->Signal[N];
        f32 Ns = (f32)N/NUM_SAMPLES;
        f32 Theta = 2*PI*CutFreq*Ns;
        f32 Magnitude = 200 * Sample;
        f32 X = sinf(Theta) * Magnitude;
        f32 Y = cosf(Theta) * Magnitude;
        
        SumX += X;
        SumY += Y;
        
        DrawPixel(X + CircleOriginX, Y + CircleOriginY, CutCol);
    }
    
    // Draw 2D summation.
    f32 Sum2DScale = 0.01f;
    i32 SumXFinal = (i32)(SumX*Sum2DScale) + CircleOriginX;
    i32 SumYFinal = (i32)(SumY*Sum2DScale) + CircleOriginY;
    DrawCircle(SumXFinal, 
               SumYFinal, 
               5, 
               Sum2DCol);
    DrawLine(CircleOriginX,
             CircleOriginY,
             SumXFinal,
             SumYFinal,
             Sum2DCol);
    */  /*
    SlowFourierTransform(&State->Signal[0], &State->FourierTransform[0], NUM_SAMPLES);
    
    // Draw the frequency domain
    for (i32 N = 0; N < NUM_SAMPLES; N++)
    {
        f32 F = State->FourierTransform[N];
        DrawPixel(N,
                  650 - (F * 1),
                  Sum2DCol);
    }
    
    DrawText(TextFormat("Cut Frequency: %.2fHz", CutFreq),
             15,
             15,
             20,
             RAYWHITE);
}

i32 
main(i32 argc, char **argv)
{
    const i32 screen_width = NUM_SAMPLES;
    const i32 screen_height = 768;
    InitWindow(screen_width, screen_height, "FourierTransform");
    SetTargetFPS(60);
    InitAudioDevice();
    
    state *State = malloc(sizeof(state));
    for (i32 N = 0; N < NUM_SAMPLES; N++)
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
}*/

int main() {
    // Get all devices
    cl_device_id *devices;
    cl_uint n_devices;
    clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, 3, NULL, &n_devices);
    devices = (cl_device_id *) malloc(n_devices * sizeof(cl_device_id));
    clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, n_devices, devices, NULL);
    if(n_devices == 0) {
        fprintf(stderr, "No devices found. Exiting.\n");
        return -1;
    }

    char name[128];
    printf("Devices:\n");
    for(int i=0;i<n_devices;i++) {
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(name), name, NULL);
        printf("[%d]: %s\n", i, name);
    }

    // Load program from file
    FILE *program_file = fopen(CL_PROGRAM_FILE, "r");
    if(program_file == NULL) {
        fprintf(stderr, "Failed to open OpenCL program file\n");
        return -1;
    }
    fseek(program_file, 0, SEEK_END);
    size_t program_size = ftell(program_file);
    rewind(program_file);
    char *program_text = (char *) malloc((program_size + 1) * sizeof(char));
    program_text[program_size] = '\0';
    fread(program_text, sizeof(char), program_size, program_file);
    fclose(program_file);
    

    printf("Device                                        | GFLOP/s w/o transfer | GFLOP/s w/ transfer |      Errors\n");
    printf("--------------------------------------------------------------------------------------------------------\n");
    for(int i=0; i<n_devices;i++) {
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(name), name, NULL);
        printf("[%d]: %40s | ", i, name);

        struct benchmark_result result;

        transform(devices[i], program_text, KERNEL_NAME, &result);

        double gflops_calc = 2.0 * N * N * N / 1e9 / result.calc_time;
        double gflops_transfer = 2.0 * N * N * N / 1e9 / (result.transfer_time + result.calc_time);
        
        printf("          %10.2f |          %10.2f |   %9d\n", gflops_calc, gflops_transfer, result.errors);
    }
      
    
    //free(program_text);
    //free(devices);
    return 0;
}