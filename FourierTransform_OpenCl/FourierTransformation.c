#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <sys/time.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define CL_FILE "FourierKernel.cl"
#define KERNEL_DEF "fourier_transformation"
#define N 1000000
#define FREQ 10
#define T 20
#define PI 3.14159265359

struct speedTest {
    double calculation;
    double transfer;
};

//Calculate the value of a singal at a given point
float calculateSignal(int x) {
    return (float) sinf((PI*x)/3);
} 

int transform(cl_device_id device, char *programText, char *kernelFile, struct speedTest *speedTest) {
    
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
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&programText, NULL, &err);
    if (err < 0) {
        fprintf(stderr, "Failed to create cl program object!\n");
        return -1;
    }

    //build Program with headers used for calculation
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];

        fprintf(stderr, "Failed to build program!\n");
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        fprintf(stderr, "%s\n", buffer);
        return -1;
    }

    //Create Kernels
    cl_kernel kernel = clCreateKernel(program, KERNEL_DEF, &err);
    if (!kernel || err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create kernel!\n");
        return -1;
    }

    //Prepare the signal data on the Host machine
    float h_Ck[N];
    float *h_signal = malloc(N*sizeof(float));
    for (size_t i = 0; i < N; i++) {
        h_signal[i] = sinf(2 * PI * 4 * ((float)i / N));
    }


    //create array buffer in the device memory
    cl_mem d_Y = clCreateBuffer(context, CL_MEM_READ_ONLY, N*sizeof(float), NULL, NULL);
    if (!d_Y) {
        fprintf(stderr, "Failed to allocate memory on device!\n");
        return -1;
    }
    //create array buffer in the device memory
    cl_mem d_Ck = clCreateBuffer(context, CL_MEM_READ_WRITE, N*sizeof(float), NULL, NULL);
    if (!d_Ck) {
        fprintf(stderr, "Failed to allocate memory on device!\n");
        return -1;
    }

    // Copy data to device
    cl_event prof_event;
    size_t bytes;
    cl_long end, start;
    double transferTime = 0.0;

    err = clEnqueueWriteBuffer(commands, d_Y, CL_FALSE, 0, N*sizeof(float), h_signal, 0, NULL, &prof_event);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to write to device memory!\n");
        return -1;
    }
    // Wait for transfer to finish
    clFinish(commands);
    clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, &bytes);
    clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, &bytes);
    transferTime += (double) (end - start) / 1.0e9;

    // Set the arguments to our compute kernel
    int n = N;
    int k = T;
    err = clSetKernelArg(kernel, 0, sizeof(int), &n);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_Y);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_Ck);
    //... what else goes herre has to be added when .cl file is thought out
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to set the kernel arguments!\n");
        return -1;
    }

    // Execute the kernel
    size_t global_size[] = {N};
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, global_size, NULL, 0, NULL, &prof_event);
    if (err) {
        fprintf(stderr, "Failed to execute kernel on device!\n");
        return -1;
    }
    clFinish(commands);

    clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, &bytes);
    clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, &bytes);
    double execTime = (double) (end - start) / 1.0e9;

    // Copy speedTest Data Back
    err = clEnqueueReadBuffer(commands, d_Ck, CL_TRUE, 0, N, h_Ck, 0, NULL, &prof_event);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to copy speedTests back to host!\n");
        return -1;
    }
    clFinish(commands);

    speedTest->transfer = transferTime;
    speedTest->calculation = execTime;

    //release memory
    clReleaseMemObject(d_Y);
    clReleaseMemObject(d_Ck);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
    free(h_signal);
    return 0;

}

int main() {
    // Get all deviceList
    cl_device_id *deviceList;
    cl_uint numberDevices;

    clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, 3, NULL, &numberDevices);
    deviceList = (cl_device_id *) malloc(numberDevices * sizeof(cl_device_id));
    clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, numberDevices, deviceList, NULL);
    if(numberDevices == 0) {
        fprintf(stderr, "No List od devices found!\n");
        return -1;
    }

    char deviceName[128];
    printf("Aviable devices:\n");
    for(int i=0;i<numberDevices;i++) {
        clGetDeviceInfo(deviceList[i], CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
        printf("[%d]: %s\n", i, deviceName);
    }

    // Load program .cl file
    FILE *program_file = fopen(CL_FILE, "r");
    if(program_file == NULL) {
        fprintf(stderr, "Failed to open .cl file\n");
        return -1;
    }
    fseek(program_file, 0, SEEK_END);
    size_t program_size = ftell(program_file);
    rewind(program_file);
    //+ 1 for terminating String
    char *program_text = (char *) malloc((program_size + 1) * sizeof(char));
    //terminate String
    program_text[program_size] = '\0';
    fread(program_text, sizeof(char), program_size, program_file);
    fclose(program_file);
    

    printf("Device                                        | Calc time s | Transfer time s \n");
    printf("--------------------------------------------------------------------------------\n");
    for(int i=0; i < numberDevices;i++) {
        clGetDeviceInfo(deviceList[i], CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
        printf("[%d]: %40s | ", i, deviceName);

        struct speedTest speedTest;

        transform(deviceList[i], program_text, KERNEL_DEF, &speedTest);
        
        printf("     %.4f |           %f\n", speedTest.calculation, speedTest.transfer);

    
    }
      
    
    free(program_text);
    free(deviceList);
    return 0;
}