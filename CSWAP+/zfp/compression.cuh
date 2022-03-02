#include <stdio.h>
#include<stdlib.h>
#include<stdint.h>
extern "C"{
void PreGPUcompression(int arraySize, int kernelSize, float* arrayGPU, int* valueIndex,int dimsize,int blocksize,cudaStream_t stream);
void PreAcc(int* valueIndex,int KernelSize,int* num,int dimsize,int blocksize,cudaStream_t stream);
void GPUcompression(int arraySize, int kernelSize, float* arrayGPU, float* compressedList, int* compressedValueIndex, uint32_t* compressedBinIndex,int dimsize,int blocksize,cudaStream_t stream);
void GPUdecompression(float* arrayGPU, float* destiGPU, int arraySize, int kernelSize, uint32_t* gpuDataindex, int* beginIndex,int dimsize,int blocksize);
}