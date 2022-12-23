# MacOs
macos:
		clang -framework Opencl -o FourierTransformation.out FourierTransformation.c

# Windows
		gcc -I"$(CUDA_PATH)/include" -L"$(CUDA_PATH)/lib/x64" -o FourierTransformation.exe FourierTransformation.c -lOpenCL 
