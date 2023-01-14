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

#define CL_PROGRAM_FILE "FourierKernel.cl"
#define KERNEL_NAME "fourier_transformation"
#define N 10
#define T 20
#define PI 3.14159265359

struct speedTest {
    double transfer_time;
    double calc_time;
    float cK[N];
};

//Calculate the value of a singal at a given point
float calculateSignal(int x) {
    return (float) sinf((PI*x)/3);
} 

int transform(cl_device_id device, char *program_text, char *kernel_name, struct speedTest *result) {
    
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
    float h_Ck[N];
    float *h_signal = malloc(N*sizeof(float));
    for (size_t i = 0; i < N; i++) {
        h_signal[i] = calculateSignal(i);
    }


    //create array buffer in the device memory
    cl_mem d_Y = clCreateBuffer(context, CL_MEM_READ_ONLY, N*sizeof(float), NULL, NULL);
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

    err = clEnqueueWriteBuffer(commands, d_Y, CL_FALSE, 0, N*sizeof(float), h_signal, 0, NULL, &prof_event);
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

    printf("\n%f\n", h_Ck[0 + N - 1]);

    result->transfer_time = transfer_sec;
    result->calc_time = 0;
    for(int i = 0; i < N; i++){
            printf("ck[%d] = %0.100f\n", i, (h_Ck[i]));
        }

    //release memory on device and host
    //free Program, context, etc.
    /*clReleaseMemObject(d_Y);
    clReleaseMemObject(d_Ck);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
    free(h_Yn);*/
    return 0;

}

int main() {
    // Get all devices
    cl_device_id *devices;
    cl_uint n_devices;
    clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, 3, NULL, &n_devices);
    devices = (cl_device_id *) malloc(n_devices * sizeof(cl_device_id));
    clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, n_devices, devices, NULL);
    if(n_devices == 0) {
        fprintf(stderr, "No devices found!\n");
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
    

    printf("Device                                        | Calc time s | Transfer time s \n");
    printf("------------------------------------------------------------------------------\n");
    for(int i=0; i < 1;i++) {
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(name), name, NULL);
        printf("[%d]: %40s | ", i, name);

        struct speedTest result;

        transform(devices[i], program_text, KERNEL_NAME, &result);
        
        printf("     %.4f |           %f\n", result.calc_time, result.transfer_time);

    
    }
      
    
    free(program_text);
    free(devices);
    return 0;
}