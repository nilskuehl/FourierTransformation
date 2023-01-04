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
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &h_Yn);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &k);
    //... what else goes herre has to be added when .cl file is thought out
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to set kernel arguments!\n");
        return -1;
    }

    // Execute the kernel
    size_t global_size[] = {N, N};
    err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, global_size, NULL, 0, NULL, &prof_event);
    if (err) {
        fprintf(stderr, "Failed to execute kernel!\n");
        return -1;
    }
    clFinish(commands);

    // Copy Result Data Back
    err = clEnqueueReadBuffer(commands, d_Y, CL_TRUE, 0, N, h_Ck, 0, NULL, &prof_event);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to read result!\n");
        return -1;
    }
    clFinish(commands);

    //release memory on device and host
    //free Program, context, etc.
    clReleaseMemObject(d_Y);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
    free(h_Yn);
    free(h_Ck);
    return 0;

}

//Calculate the value of a singal at a given point
float calculateSignal(int x) {
    return (float) Sin(x) + Cos(x);
}

int main() {

}