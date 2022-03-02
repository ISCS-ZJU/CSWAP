#include "compression.cuh"
#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include <sys/time.h>
#include<iostream>


__global__ void PreAcc(int valueIndex[],int KernelSize,int num[]){
    
    for(int i=1;i<KernelSize;++i){
        valueIndex[i]+=valueIndex[i-1];
    }
    num[0] = valueIndex[KernelSize-1];
    for(int i=KernelSize-1;i>0;--i){
        valueIndex[i] = valueIndex[i-1];
    }
    valueIndex[0] = 0;
}

/**
__global__ void PreGPUcompression(int arraySize, int kernelSize, float arrayGPU[], int valueIndex[]){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int indexNumber = (arraySize / 32 / kernelSize) ;
    int index_begin = indexNumber*i*32;
    int index_end;
    if(i != kernelSize-1)
        index_end =  index_begin + indexNumber*32;
    else
        index_end =  arraySize;
    int valueFlag = 0;

    for(int cur= index_begin; cur<index_end; cur++){
        if(arrayGPU[cur] != 0)
            valueFlag++;
    }
    valueIndex[i] = valueFlag;
}


__global__ void GPUcompression(int arraySize, int kernelSize, float arrayGPU[], float compressedList[], int compressedValueIndex[], uint32_t compressedBinIndex[]){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int indexNumber = (arraySize/ 32 / kernelSize) ;
    int index_begin = indexNumber*i*32;
    int index_end;
    if(i != kernelSize-1)
        index_end =  index_begin + indexNumber*32;
    else
        index_end =  arraySize;

    int cursor = 0;
    uint32_t  final, indexcut[32];
    uint32_t powerList[32] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 
        2048, 4096, 8192, 16384,32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 
            8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648 };
    int cur;
    int valueindex = compressedValueIndex[i];
    for(cur= index_begin; cur<index_end; cur++){
        
        // record 32 bins index
        if(cursor==32){
            final = 0;
            cursor = 0;
            for (int i = 0; i < 32; i++) {
                if (indexcut[i] == 1)
                    final += powerList[i];
            }
            compressedBinIndex[cur / 32 - 1] = final;
        }
        if (arrayGPU[cur] == 0){
            indexcut[cursor++] = 0;
        }
        else{
            indexcut[cursor++] = 1;
            compressedList[valueindex++] = arrayGPU[cur];
        }
        
    }
    // operate final compressedBinIndex of each processor
    
        final = 0;
        for (int i = 0; i < 32; i++) {
            if (indexcut[i] == 1)
                final += powerList[i];
        }
        compressedBinIndex[index_end / 32 - 1] = final;
}

__global__ void GPUdecompression(float arrayGPU[], float destiGPU[], int arraySize, int kernelSize, uint32_t gpuDataindex[], int beginIndex[])
{   

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // how many index every kernel needs to operate 
    int indexNumber = (arraySize/kernelSize) / 32;
    int index_begin = indexNumber*i;
    int index_end;
    if(i != kernelSize-1)
        index_end =  index_begin + indexNumber;
    else
        index_end =  arraySize / 32;

    // int index_end =  index_begin + indexNumber;
    // first data place of every kernel
    int dataReadnow = beginIndex[i];
    for(int cur= index_begin; cur<index_end; cur++){
        int result = 0;
        uint32_t temp = gpuDataindex[cur];
        // printf("Hello thread %lld \n",gpuDataindex[cur]);
        int all = 0;
        while(temp){
            result = temp % 2;
            temp = temp / 2;

            if(result==1){
                destiGPU[cur * 32 + all] = arrayGPU[dataReadnow];
                dataReadnow++;
            }
            else{
                destiGPU[cur * 32 + all] = 0;
            }

            all++;
        }
    for(int j =all;j<32;j++)
        {
            destiGPU[cur * 32 + j] = 0;
        }
    }
}
**/



__global__ void PreGPUcompression(int arraySize, int kernelSize, float arrayGPU[], int valueIndex[])
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int indexNumber;
    // imitate ceil
    if(arraySize % kernelSize == 0)
        indexNumber = arraySize / kernelSize;
    else
        indexNumber = arraySize / kernelSize + 1;

    int index_begin = indexNumber*i;
    int index_end = index_begin + indexNumber;

    // undivided situation operation
    // Do not operate when index_begin overflow the arraySize
    if(index_begin < arraySize){
        int valueFlag = 0;

        // only operate index_begin --> arraySize
        if(index_end > arraySize)
            index_end = arraySize;

        for(int cur= index_begin; cur<index_end; cur++){

            if(arrayGPU[cur] != 0)
                valueFlag++;
        }
        valueIndex[i] = valueFlag;
    }

    // printf("Hello thread %d  %d  %d  %d  %d--- \n", i,index_begin,index_end,index_end - index_begin,arraySize);
}

// 不能整除的情况
__global__ void GPUcompression(int arraySize, int kernelSize, float arrayGPU[], float compressedList[], int compressedValueIndex[], uint32_t compressedBinIndex[])
{
    __shared__ uint32_t powerList[32] ;
    if( threadIdx.x == 0){
    	powerList[0] = 1;powerList[1] = 2;powerList[2] = 4;powerList[3] = 8;powerList[4] = 16;powerList[5] = 32;powerList[6] = 64;powerList[7] = 128;powerList[8] = 256;powerList[9] = 512;
    	powerList[10] = 1024;powerList[11] = 2048;powerList[12] = 4096;powerList[13] = 8192;powerList[14] = 16384;powerList[15] = 32768;powerList[16] = 65536;powerList[17] = 131072;powerList[18] = 262144;powerList[19] = 524288;
    	powerList[20] = 1048576;powerList[21] = 2097152;powerList[22] = 4194304;powerList[23] = 8388608;powerList[24] = 16777216;powerList[25] = 33554432;powerList[26] = 67108864;powerList[27] = 134217728;powerList[28] = 268435456;powerList[29] = 536870912;
    	powerList[30] = 1073741824;powerList[31] = 2147483648;
    }
    // __syncthreads();
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int indexNumber;
    // imitate ceil
    if(arraySize % kernelSize == 0)
        indexNumber = arraySize / kernelSize;
    else
        indexNumber = arraySize / kernelSize + 1;

    int index_begin = indexNumber*i;
    int index_end = index_begin + indexNumber;
    int eachProcessIndexCount;
    // 每个核要处理数据量的索引量 向上取整
    if(indexNumber % 32 == 0)
        eachProcessIndexCount = indexNumber / 32;
    else
        eachProcessIndexCount = indexNumber / 32 + 1;
    // Do not operate when index_begin overflow the arraySize
    if(index_begin < arraySize){

        int ValueIndex = compressedValueIndex[i];
        int cursor = 0;
        uint32_t  final, indexcut[32];

        // only operate index_begin --> arraySize
        // 将最后一个核的处理量规整到arraySize大小
        if(index_end > arraySize)
            index_end = arraySize;

        // 当前核数据索引的起始位置
        int BinIndexCursor = i * eachProcessIndexCount;
        for(int cur= index_begin; cur<index_end; cur++){
            // record 32 bins index
            if(cursor==32){
                final = 0;
                cursor = 0;
                for (int j = 0; j < 32; j++) {
                    if (indexcut[j] == 1)
                        final += powerList[j];
                }
                compressedBinIndex[BinIndexCursor++] = final;
            }

            if (arrayGPU[cur] == 0){
                indexcut[cursor++] = 0;
            }
            else{
                indexcut[cursor++] = 1;
                compressedList[ValueIndex++] = arrayGPU[cur];
            }
        }
        // operate final compressedBinIndex of each processor
        final = 0;
        // operate ending data of each process unit
        // 操作剩下的数据（不足32个） 若恰好为32倍数 也会进入
        for (int j = 0; j < cursor; j++) {
            if (indexcut[j] == 1)
                final += powerList[j];
        }
        compressedBinIndex[BinIndexCursor] = final;
    }

}

__global__ void GPUdecompression(float arrayGPU[], float destiGPU[], int arraySize, int kernelSize, uint32_t gpuDataindex[], int beginIndex[])
{   

    int i = blockDim.x * blockIdx.x + threadIdx.x;

//-----
    int indexNumber, processOperateNum;
    // imitate ceil
    if(arraySize % kernelSize == 0)
        processOperateNum = arraySize / kernelSize;
    else
        processOperateNum = arraySize / kernelSize + 1;

// 部分核不工作
    if(processOperateNum * i < arraySize){

        if(processOperateNum % 32 ==0)
            indexNumber = processOperateNum / 32;
        else
            indexNumber = processOperateNum / 32 + 1;

        int index_begin = indexNumber * i;
        // int index_end;

        int index_end = index_begin + indexNumber;
        int WriteCur;
        // first data place of every kernel
        int dataReadnow = beginIndex[i];
        uint32_t temp;
        int result;
        int cur;
        // 能整除32 全部处理
        int binFlag = 0;
        for(cur= index_begin; cur<index_end-1; cur++){
            result = 0;
            temp = gpuDataindex[cur];
            // WriteCur = cur * 32;
            WriteCur = (processOperateNum * i) + binFlag * 32;
            binFlag++;

            while(temp){
                result = temp % 2;
                temp = temp / 2;

                if(result==1){
                    destiGPU[WriteCur++] = arrayGPU[dataReadnow];
                    dataReadnow++;
                }
                else{
                    destiGPU[WriteCur++] = 0;
                }
            }
        }
        // 处理最后一个核的数据
        
        int end_index = (i+1)*processOperateNum;
        if(end_index<arraySize)
            end_index = arraySize;

        // threshold = (i+1) * processOperateNum;
        temp = gpuDataindex[cur];
        WriteCur = (processOperateNum * i) + binFlag * 32;
        // 处理尾巴数据
        while(temp){
            if(WriteCur==end_index)
                break;
            result = temp % 2;
            temp = temp / 2;
            if(result==1){
                destiGPU[WriteCur++] = arrayGPU[dataReadnow];
                dataReadnow++;
            }
            else{
                destiGPU[WriteCur++] = 0;
            }
        }
    }
}






double get_cur_time_ms() {
    struct timeval   tv;
    struct timezone  tz;
    double cur_time;
    gettimeofday(&tv, &tz);
    cur_time = tv.tv_sec*1000 + tv.tv_usec / 1000.0;
    return cur_time;
}

void PreGPUcompression(int arraySize, int kernelSize, float* arrayGPU, int* valueIndex,int dimsize,int blocksize,cudaStream_t stream){
    //cudaStream_t stream;
    //cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking);
    dim3 dimGrid(dimsize);
    dim3 dimBlock(blocksize);
    PreGPUcompression<<<dimGrid, dimBlock,0,stream>>>(arraySize, kernelSize, arrayGPU, valueIndex);
    //cudaStreamSynchronize(stream);
}
void PreAcc(int* valueIndex,int KernelSize,int* num,int dimsize,int blocksize,cudaStream_t stream){
    //cudaStream_t stream;
    //cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking);
    dim3 dimGrid(dimsize);
    dim3 dimBlock(blocksize);
    PreAcc<<<dimGrid,dimBlock,0,stream>>>(valueIndex,KernelSize,num);
    //cudaStreamSynchronize(stream);

}
void GPUcompression(int arraySize, int kernelSize, float* arrayGPU, float* compressedList, int* compressedValueIndex, uint32_t* compressedBinIndex,int dimsize,int blocksize,cudaStream_t stream){
    
    //cudaStream_t stream;
    //cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking);
    dim3 dimGrid(dimsize);
    dim3 dimBlock(blocksize);
    //cudaDeviceSynchronize();
    //double start_ = get_cur_time_ms();
    GPUcompression<<<dimGrid, dimBlock,0,stream>>>(arraySize, kernelSize, arrayGPU, compressedList, compressedValueIndex, compressedBinIndex);
    //cudaStreamSynchronize(stream);
    //cudaDeviceSynchronize();
    //double end_ = get_cur_time_ms();
    //std::cout<<arraySize<<":"<<kernelSize;
    //std::cout<<"***"<<end_-start_<<std::endl;
}

void GPUdecompression(float* arrayGPU, float* destiGPU, int arraySize, int kernelSize, uint32_t* gpuDataindex, int* beginIndex,int dimsize,int blocksize){
    //cudaStream_t stream;
    //cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking);
    dim3 dimGrid(dimsize);
    dim3 dimBlock(blocksize);
    GPUdecompression<<<dimGrid, dimBlock>>>(arrayGPU, destiGPU, arraySize, kernelSize, gpuDataindex, beginIndex);
    //cudaStreamSynchronize(stream);
}