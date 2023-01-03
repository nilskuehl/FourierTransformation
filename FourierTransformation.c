#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
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

int transform(cl_device_id device, char *program_text, char *kernel_name, _) {
    
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
    err = clBuildProgram(program, 0, NULL, "-I complex.h", NULL, NULL);
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
    float *h_Yn = malloc(N);
    float *h_Ck = malloc(N);
    for (size_t i = 0; i < N - 1; i++) {
        h_Yn[i] = calculateSignal(i);
        h_Ck[i] = -1;
    }

    //create array buffer in the device memory
    cl_mem d_Y = clCreateBuffer(context, CL_MEM_READ_ONLY, N, NULL, NULL);
    if (!d_Y) {
        fprintf(stderr, "Failed to allocate device memory\n");
        return -1;
    }


}

//Calculate the value of a singal at a given point
float calculateSignal(int x) {
    return (float) Sin(x) + Cos(x);
}

int main() {

}