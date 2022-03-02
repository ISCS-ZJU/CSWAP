#include<iostream>
#include<fstream>
#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include<string>
#include<vector>
#include<functional>
#include<sys/time.h>
#include<stdio.h>
#include<stdlib.h>
#include<torch/csrc/autograd/generated/Functions.h>
#include <cuda_runtime.h>
#include<c10/core/PolicyMaker.h>
//#include<c10/core/TransferRun.h>
#include<c10/core/CPUAllocator.h>
#include<ATen/core/TransferRun.h>
#include<thread>
#include <ATen/core/grad_mode.h>
#include <ATen/core/zfp.h>

extern "C"{
void PreGPUcompression(int arraySize, int kernelSize, float* arrayGPU, int* valueIndex,int dimsize,int blocksize,cudaStream_t stream);
void PreAcc(int* valueIndex,int KernelSize,int* num,int dimsize,int blocksize,cudaStream_t stream);
void GPUcompression(int arraySize, int kernelSize, float* arrayGPU, float* compressedList, int* compressedValueIndex, uint32_t* compressedBinIndex,int dimsize,int blocksize,cudaStream_t stream);
}
enum Layer_type{
    CONV,
    POOL,
    ACT,
    BN,
    FC,
    MASKFILL
};


class Execution{
public:
    Execution(){ 
    }
    void init(){
        index = -1;
        
    }
    void init_(int n){
        policy.resize(n,0);
        begin_stage.resize(n,0); 
        cudaStreamCreateWithFlags(&stream1,cudaStreamNonBlocking);
        dimSize = 512;
        blockSize = 32;
        process = dimSize*blockSize;
        index_begin = at::cuda::getCUDADeviceAllocator()->raw_allocate(process*sizeof(int));
        num = at::cuda::getCUDADeviceAllocator()->raw_allocate(sizeof(int));
        num_ptr = nullptr;
        cudaMallocHost(&num_ptr,sizeof(int));
        compress_value = nullptr;
        index_ = nullptr;
        buffer = nullptr;      
        c10::ProfilerSwitch::set_enabled(true);
        c10::Passive::set_enabled(true);
        c10::GetProfilerImpl()->init(3);
        setRatio(128);
    }
    void pre(torch::Tensor& pre,Layer_type layer_type=ACT){
        //std::cout<<pre.tensor_id<<":"<<begin_stage[pre.tensor_id]<<std::endl;
        index++;
        //std::cout<<index<<std::endl;
        if((pre.tensor_id>=0)&&(index==begin_stage[pre.tensor_id])&&pre.grad_fn()){
            if((policy[pre.tensor_id]==2)||((layer_type==CONV)&&(policy_>2))){
                if(c10::Passive::is_enabled()){
                    if(c10::Profile4::is_enabled()){
                        gettimeofday(&start_1, NULL);
                    }
                    sizes = pre.nbytes();
                    dataptr_ = at::detail::getCUDAHooks().getPinnedMemoryAllocator()->allocate(sizes);
                    cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
                    if(c10::Profile4::is_enabled()){
                        c10::GetCompressProfile()->add_tensor_size(sizes);
                    }
                    cudaMemcpyAsync(dataptr_.get(),pre.data_ptr(),sizes,cudaMemcpyDeviceToHost,stream1);
                    if(c10::Profile4::is_enabled()){
                        gettimeofday(&start_, NULL);
                    }
                } 
            }else if(policy[pre.tensor_id]==3){
                //std::cout<<"beginzvc"<<std::endl;
                size_t index_size = ceil(ceil((float)(pre.numel())/(float)process)/32.0)*process*sizeof(uint32_t);
                PreGPUcompression(pre.numel(), process, (float*)pre.data_ptr(), (int*)index_begin,dimSize,blockSize,c10::cuda::getCurrentCUDAStream().stream());
                int kernel_size = ceil(float(pre.numel())/ceil(float(pre.numel())/process));
                PreAcc((int*)index_begin,kernel_size,(int*)num,1,1,c10::cuda::getCurrentCUDAStream().stream());
                cudaMemcpy(num_ptr, num, sizeof(int), cudaMemcpyDeviceToHost);
                compress_value = at::cuda::getCUDADeviceAllocator()->raw_allocate((*(int*)num_ptr)*sizeof(float));
                index_ = at::cuda::getCUDADeviceAllocator()->raw_allocate(index_size);
                GPUcompression(pre.numel(), process, (float*)pre.data_ptr(), (float*)compress_value, (int*)index_begin, (uint32_t*)index_,dimSize,blockSize,c10::cuda::getCurrentCUDAStream().stream());
                //dataptr_ = at::detail::getCUDAHooks().getPinnedMemoryAllocator()->allocate((*(int*)num_ptr)*sizeof(float));
                dataptr_ = at::detail::getCUDAHooks().getPinnedMemoryAllocator()->allocate(pre.nbytes());
                dataptr_.set_index(at::detail::getCUDAHooks().getPinnedMemoryAllocator()->raw_allocate(index_size));
                dataptr_.set_index_begin(at::detail::getCUDAHooks().getPinnedMemoryAllocator()->raw_allocate(process*sizeof(int)));
                dataptr_.set_size((*(int*)num_ptr)*sizeof(float));
                dataptr_.set_index_size(index_size);
                dataptr_.set_index_begin_size(process*sizeof(int));  
                //std::cout<<*(int*)num_ptr<<":"<<pre.numel()<<std::endl;
                cudaMemcpyAsync(dataptr_.get(), compress_value,(*(int*)num_ptr)*sizeof(float), cudaMemcpyDeviceToHost,stream1);
                cudaMemcpyAsync(dataptr_.get_index(), index_ ,index_size,cudaMemcpyDeviceToHost,stream1);
                cudaMemcpyAsync(dataptr_.get_index_begin(), index_begin ,process*sizeof(int),cudaMemcpyDeviceToHost,stream1);
            }else if(policy[pre.tensor_id]==6){
                //std::cout<<"begin"<<std::endl;
                if(c10::ProfilerSwitch::is_enabled()){
                    cudaDeviceSynchronize();
                    gettimeofday(&start_, NULL);
                }
                buffer = compress(pre.data_ptr(),pre.numel(),buffersize);
                dataptr_ = at::detail::getCUDAHooks().getPinnedMemoryAllocator()->allocate(buffersize);
                dataptr_.set_size(buffersize);
                if(c10::ProfilerSwitch::is_enabled()){
                    cudaDeviceSynchronize();
                    gettimeofday(&end_, NULL);
                    timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
                    c10::GetProfilerImpl()->add_time(2,timeuse);
                }
                cudaMemcpyAsync(dataptr_.get(),buffer,buffersize,cudaMemcpyDeviceToHost,stream1);
                if(c10::ProfilerSwitch::is_enabled()){
                    //cudaDeviceSynchronize();
                    gettimeofday(&start_, NULL);
                }
            }            
        }
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&start_, NULL);
        }
    }

    void pre(std::vector<torch::Tensor> inputs, Layer_type layer_type=ACT){
        index++;
        int length = inputs.size();
        dataptrs.clear();
        stream.clear();
        start_s.clear();
        start_1s.clear();
        end_1s.clear();
        end_s.clear();
        compress_info.clear();
        buffers.clear();
        buffersizes.clear();
        dataptrs.resize(length);
        stream.resize(length);
        start_s.resize(length);
        start_1s.resize(length);
        end_s.resize(length);
        end_1s.resize(length);
        compress_info.resize(length);
        buffers.resize(length);
        buffersizes.resize(length);
        for(int j=0;j<length;j++){
            if((inputs[j].tensor_id>=0)&&(index==begin_stage[inputs[j].tensor_id])&&inputs[j].grad_fn()){
                if((policy[inputs[j].tensor_id]==2)||((layer_type==CONV)&&(policy_>2))){
                    if(c10::Passive::is_enabled()){
                        if(c10::Profile4::is_enabled()){
                            gettimeofday(&start_1s[j], NULL); 
                        }
                        //sizes = inputs[j].nbytes();
                        //dataptrs[j] = at::detail::getCUDAHooks().getPinnedMemoryAllocator()->allocate(sizes);
                        cudaStreamCreateWithFlags(&stream[j],cudaStreamNonBlocking);
                        if(c10::Profile4::is_enabled()){
                            c10::GetCompressProfile()->add_tensor_size(sizes);
                        }
                        //cudaMemcpyAsync(dataptrs[j].get(),inputs[j].data_ptr(),sizes,cudaMemcpyDeviceToHost,stream[j]);
                        if(c10::Profile4::is_enabled()){
                            gettimeofday(&start_s[j], NULL);
                        }
                    }            
                }else if(policy[inputs[j].tensor_id]==3){
                    compress_info[j].resize(2);
                    size_t index_size = ceil(ceil((float)(inputs[j].numel())/(float)process)/32.0)*process*sizeof(uint32_t);
                    PreGPUcompression(inputs[j].numel(), process, (float*)inputs[j].data_ptr(), (int*)index_begin,dimSize,blockSize,c10::cuda::getCurrentCUDAStream().stream());
                    int kernel_size = ceil(float(inputs[j].numel())/ceil(float(inputs[j].numel())/process));
                    PreAcc((int*)index_begin,kernel_size,(int*)num,1,1,c10::cuda::getCurrentCUDAStream().stream());
                    cudaMemcpy(num_ptr, num, sizeof(int), cudaMemcpyDeviceToHost);

                    //std::cout<<(*(int*)num_ptr)*1.0/inputs[j].numel()<<";";
                    //std::cout<<inputs[j].numel()<<";";

                    compress_info[j][0] = at::cuda::getCUDADeviceAllocator()->raw_allocate((*(int*)num_ptr)*sizeof(float));
                    compress_info[j][1] = at::cuda::getCUDADeviceAllocator()->raw_allocate(index_size);
                    GPUcompression(inputs[j].numel(), process, (float*)inputs[j].data_ptr(), (float*)compress_info[j][0], (int*)index_begin, (uint32_t*)compress_info[j][1],dimSize,blockSize,c10::cuda::getCurrentCUDAStream().stream());
                    //dataptrs[j] = at::detail::getCUDAHooks().getPinnedMemoryAllocator()->allocate((*(int*)num_ptr)*sizeof(float));
                    dataptrs[j] = at::detail::getCUDAHooks().getPinnedMemoryAllocator()->allocate(inputs[j].nbytes());
                    dataptrs[j].set_index(at::detail::getCUDAHooks().getPinnedMemoryAllocator()->raw_allocate(index_size));
                    dataptrs[j].set_index_begin(at::detail::getCUDAHooks().getPinnedMemoryAllocator()->raw_allocate(process*sizeof(int)));
                    dataptrs[j].set_size((*(int*)num_ptr)*sizeof(float));
                    dataptrs[j].set_index_size(index_size);
                    dataptrs[j].set_index_begin_size(process*sizeof(int));     
                    cudaStreamCreateWithFlags(&stream[j],cudaStreamNonBlocking); 
                    cudaMemcpyAsync(dataptrs[j].get(), compress_info[j][0],(*(int*)num_ptr)*sizeof(float), cudaMemcpyDeviceToHost,stream[j]);
                    cudaMemcpyAsync(dataptrs[j].get_index(), compress_info[j][1] ,index_size,cudaMemcpyDeviceToHost,stream[j]);
                    cudaMemcpyAsync(dataptrs[j].get_index_begin(), index_begin ,process*sizeof(int),cudaMemcpyDeviceToHost,stream[j]);
                }else if(policy[inputs[j].tensor_id]==6){
                    cudaStreamCreateWithFlags(&stream[j],cudaStreamNonBlocking); 
                    buffers[j] = compress(inputs[j].data_ptr(),inputs[j].numel(),buffersizes[j]);
                    dataptrs[j] = at::detail::getCUDAHooks().getPinnedMemoryAllocator()->allocate(buffersizes[j]);
                    dataptrs[j].set_size(buffersizes[j]);
                    cudaMemcpyAsync(dataptrs[j].get(),buffers[j],buffersizes[j],cudaMemcpyDeviceToHost,stream[j]);
                }
            }
        }
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&start_, NULL);
        }
    }




    void post(torch::Tensor& pre,torch::Tensor& x, /**std::function<at::Tensor(const at::Tensor&)> func_**/ Layer_type layer_type=ACT,bool isomit=false){   
        //std::cout<<pre.tensor_id<<std::endl;   
        x.tensor_id = index;
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&end_, NULL);
            timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
            c10::GetPolicyMaker()->add_comptime(timeuse);
            if(x.grad_fn()){
                x.grad_fn()->id_ = index;
                x.grad_fn()->ids.push_back(pre.tensor_id);
            }
            //x.grad_fn()->ids.push_back(index);
            c10::GetPolicyMaker()->add_tensor(index,x.nbytes());
            c10::GetPolicyMaker()->add_stepdep({pre.tensor_id,index});
            if(isomit){
                c10::GetPolicyMaker()->add_omitid(x.tensor_id);
            }
        }else{
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            if(c10::Profile3::is_enabled()){
                const  c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
                //std::cout<<"current memory:"<<(stats.allocated_bytes)[0].current<<std::endl;
                c10::GetCompressProfile()->add_memory_load((stats.allocated_bytes)[0].current);
            }
        }
        if((pre.tensor_id>=0)&&((policy[pre.tensor_id]==1)||(policy_==4&&layer_type!=CONV))&&(index==begin_stage[pre.tensor_id])&&pre.grad_fn()){
            pre.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->data_ptr().clear();
            pre.grad_fn()->storage_impl_ = pre.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
            pre.grad_fn()->com_and_tran = 1;
                    //x.grad_fn()->is_recomp = true;     
        }else if((pre.tensor_id>=0)&&((policy[pre.tensor_id]==2)||(layer_type==CONV&&policy_>2))&&(index==begin_stage[pre.tensor_id])&&pre.grad_fn()){
            if(c10::Passive::is_enabled()){
                if(c10::Profile4::is_enabled()){
                    gettimeofday(&end_, NULL);
                    timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
                    c10::GetCompressProfile()->add_compute_time(timeuse);
                }
                //sizes = pre.nbytes();
                //dataptr_ = at::detail::getCUDAHooks().getPinnedMemoryAllocator()->allocate(sizes);
                //cudaMemcpyAsync(dataptr_.get(),pre.data_ptr(),sizes,cudaMemcpyDeviceToHost,stream1);

                cudaStreamSynchronize(stream1);
                pre.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->set_data_ptr(std::move(dataptr_));
                pre.grad_fn()->storage_impl_ = pre.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
                pre.grad_fn()->com_and_tran = 2;      
                if(c10::Profile4::is_enabled()){
                    gettimeofday(&end_1, NULL);
                    timeuse = 1000000 * ( end_1.tv_sec - start_1.tv_sec ) + end_1.tv_usec - start_1.tv_usec;
                    c10::GetCompressProfile()->add_transfer_time(timeuse);
                }
            }else{                
                pre.grad_fn()->com_and_tran = 2;
                pre.grad_fn()->storage_impl_ = pre.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
                at::GetTransferRun()->add_item(pre.nbytes(),pre.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl());
            }
        }else if(pre.tensor_id>=0&&index==begin_stage[pre.tensor_id]&&policy[pre.tensor_id]==3&&pre.grad_fn()){
            //std::cout<<"endzvc"<<std::endl;
            cudaStreamSynchronize(stream1);
            at::cuda::getCUDADeviceAllocator()->raw_deallocate(compress_value);
            at::cuda::getCUDADeviceAllocator()->raw_deallocate(index_);
            pre.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->set_data_ptr(std::move(dataptr_));
            pre.grad_fn()->storage_impl_ = pre.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
            pre.grad_fn()->com_and_tran = 3;
            pre.grad_fn()->dimsize = dimSize;
            pre.grad_fn()->blocksize = blockSize;
        }else if(pre.tensor_id>=0&&index==begin_stage[pre.tensor_id]&&policy[pre.tensor_id]==6&&pre.grad_fn()){
            //std::cout<<"end"<<std::endl;
            if(c10::ProfilerSwitch::is_enabled()){
                gettimeofday(&end_, NULL);
                timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
                c10::GetProfilerImpl()->add_time(0,timeuse);
            }
            cudaStreamSynchronize(stream1);
            if(c10::ProfilerSwitch::is_enabled()){
                gettimeofday(&end_, NULL);
                timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
                c10::GetProfilerImpl()->add_time(1,timeuse);
            }
            at::cuda::getCUDADeviceAllocator()->raw_deallocate(buffer);
            pre.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->set_data_ptr(std::move(dataptr_));
            pre.grad_fn()->storage_impl_ = pre.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
            pre.grad_fn()->com_and_tran = 6;
        }
        if(x.grad_fn()){
            x.grad_fn()->pre_node = pre.grad_fn(); 
            x.grad_fn()->input_tensor = pre.variable_data();
            x.grad_fn()->id_ = index;
        }
        /**else{
            std::cout<<x.tensor_id<<std::endl;
        }**/
        //x.grad_fn()->func_ = func_;
    }
    void post(std::vector<torch::Tensor> inputs,torch::Tensor& x,/**std::function<at::Tensor(const at::Tensor&)> func_**/ Layer_type layer_type=ACT,bool isomit=false){
        int length = inputs.size();
        x.tensor_id = index;
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&end_, NULL);
            timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
            c10::GetPolicyMaker()->add_comptime(timeuse);
            x.grad_fn()->id_ = index;
            for(int j=0;j<length;j++){
                x.grad_fn()->ids.push_back(inputs[j].tensor_id);
            }
            c10::GetPolicyMaker()->add_tensor(index,x.nbytes());
            std::vector<int> ids{index};
            for(int j=0;j<length;j++){
                ids.push_back(inputs[j].tensor_id);
            }
            c10::GetPolicyMaker()->add_stepdep(ids);
            if(isomit){
                c10::GetPolicyMaker()->add_omitid(x.tensor_id);
            }
        }else{
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            if(c10::Profile3::is_enabled()){
                const  c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
                c10::GetCompressProfile()->add_memory_load((stats.allocated_bytes)[0].current);
                        //std::cout<<(stats.allocated_bytes)[0].current<<"-";
            }
        }
        if(x.grad_fn()){
            x.grad_fn()->is_cat = true;
            x.grad_fn()->multiple_input.input_tensors.resize(length);
            x.grad_fn()->multiple_input.pre_nodes.resize(length,nullptr);
            x.grad_fn()->id_ = index;
        }

        for(int j=0;j<length;j++){
            if(x.grad_fn()){
                x.grad_fn()->multiple_input.input_tensors[j] = inputs[j].variable_data();
                x.grad_fn()->multiple_input.pre_nodes[j] = inputs[j].grad_fn();
            }
            if(!inputs[j].grad_fn()) continue;
            if((inputs[j].tensor_id>=0)&&((policy[inputs[j].tensor_id]==1)||(policy_==4&&layer_type!=CONV))&&(index==begin_stage[inputs[j].tensor_id])){
                inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->data_ptr().clear();
                inputs[j].grad_fn()->storage_impl_ = inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
                inputs[j].grad_fn()->com_and_tran = 1;
            }else if((inputs[j].tensor_id>=0)&&((policy[inputs[j].tensor_id]==2)||(layer_type==CONV&&policy_>2))&&(index==begin_stage[inputs[j].tensor_id])){
                if(c10::Passive::is_enabled()){
                    if(c10::Profile4::is_enabled()){
                        gettimeofday(&end_s[j], NULL);
                        timeuse = 1000000 * ( end_s[j].tv_sec - start_s[j].tv_sec ) + end_s[j].tv_usec - start_s[j].tv_usec;
                        c10::GetCompressProfile()->add_compute_time(timeuse);
                    }
                    sizes = inputs[j].nbytes();
                    dataptrs[j] = at::detail::getCUDAHooks().getPinnedMemoryAllocator()->allocate(sizes);
                    cudaMemcpyAsync(dataptrs[j].get(),inputs[j].data_ptr(),sizes,cudaMemcpyDeviceToHost,stream[j]);
                    //cudaDeviceSynchronize();
                    cudaStreamSynchronize(stream[j]);
                    inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->set_data_ptr(std::move(dataptrs[j]));   
                    if(c10::Profile4::is_enabled()){ 
                        gettimeofday(&end_1s[j], NULL);
                        timeuse = 1000000 * ( end_1s[j].tv_sec - start_1s[j].tv_sec ) + end_1s[j].tv_usec - start_1s[j].tv_usec;
                        c10::GetCompressProfile()->add_transfer_time(timeuse);                       
                    }
                }else{
                    at::GetTransferRun()->add_item(inputs[j].nbytes(),inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl());
                }
                inputs[j].grad_fn()->com_and_tran = 2;
                inputs[j].grad_fn()->storage_impl_ = inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();

            }else if(inputs[j].tensor_id>=0&&policy[inputs[j].tensor_id]==3&&index==begin_stage[inputs[j].tensor_id]){
                cudaStreamSynchronize(stream[j]);
                at::cuda::getCUDADeviceAllocator()->raw_deallocate(compress_info[j][0]);
                at::cuda::getCUDADeviceAllocator()->raw_deallocate(compress_info[j][1]);
                inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->set_data_ptr(std::move(dataptrs[j]));
                inputs[j].grad_fn()->storage_impl_ = inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
                inputs[j].grad_fn()->com_and_tran = 3;
                inputs[j].grad_fn()->dimsize = dimSize;
                inputs[j].grad_fn()->blocksize = blockSize;
            }else if(inputs[j].tensor_id>=0&&policy[inputs[j].tensor_id]==6&&index==begin_stage[inputs[j].tensor_id]){
                cudaStreamSynchronize(stream[j]);
                at::cuda::getCUDADeviceAllocator()->raw_deallocate(buffers[j]);
                inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->set_data_ptr(std::move(dataptrs[j]));
                inputs[j].grad_fn()->storage_impl_ = inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
                inputs[j].grad_fn()->com_and_tran = 6;
            }
            
            
        }
        //x.grad_fn()->func_ = func_;
    }

    void preforward(){
        at::GetTransferRun()->init();
        init();
    }
    void postforward(){
        if(c10::Profile::is_enabled()){
            c10::GetPolicyMaker()->set_num(index);
        }
        at::GetTransferRun()->finish_forward();
    }


public:
    int index;
    std::vector<int> policy;
    std::vector<int> begin_stage;
    std::vector<Layer_type> types;
    int policy_;// 1 HOME 2 capuchin 3vdnn 4 superneurons

private:
    struct timeval start_, end_;
    struct timeval start_1, end_1;

    std::vector<struct timeval> start_s,end_s,start_1s,end_1s;

    int timeuse;
    int dimSize;
    int blockSize;
    int process;
    size_t buffersize;
    void* index_begin;
    void* num;
    void* num_ptr;
    void* compress_value;
    void* index_;
    void* buffer;
    size_t sizes;
    cudaStream_t stream1;
    at::DataPtr dataptr_;
    std::vector<cudaStream_t> stream;
    std::vector<c10::DataPtr> dataptrs;
    std::vector<std::vector<void*>> compress_info;
    std::vector<void*> buffers;
    std::vector<size_t> buffersizes; 
};