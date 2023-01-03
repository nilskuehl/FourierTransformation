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

int transform(cl_device_id device, char *program_text, char *kernel_name, _) {
    
    //Context
    cl_context context;
    context = clCreateContext(0, 1, &device, NULL, NULL, NULL);
    if (!context) {
        fprintf(stderr, "Failed to create context.\n");
        return -1;
    }

    //Command Queue
    cl_command_queue commands = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, NULL);
    if (!commands)
    {
        fprintf(stderr, "Failed to createg queue.\n");
        return -1;
    }
}

int main() {

}