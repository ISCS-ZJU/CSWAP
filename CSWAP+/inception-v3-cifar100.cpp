#include<iostream>
#include<fstream>
#include <torch/torch.h>
#include <torch/nn/pimpl.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include<string>
#include<vector>
#include<functional>
#include<sys/time.h>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<torch/csrc/autograd/generated/Functions.h>
#include <cuda_runtime.h>
#include <torch/nn/cloneable.h>
#include <torch/expanding_array.h>
#include<c10/core/PolicyMaker.h>
#include<c10/core/CPUAllocator.h>
#include<ATen/core/TransferRun.h>
#include <ATen/core/grad_mode.h>

using namespace std;

void log_transfer_time(torch::Tensor& x){
    /**
    torch::Device device1(torch::kCUDA);
    torch::Device device2(torch::kCPU);
    FILE *f = fopen("resnet-cifar10/transfer_time.txt","a+");
    struct timeval start_, end_;
    gettimeofday(&start_, NULL );
    auto y=x.to(device2);
    gettimeofday(&end_, NULL );
    int timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
    fprintf(f,"%d\n",timeuse);
    gettimeofday(&start_, NULL );
    y.to(device1);
    gettimeofday(&end_, NULL );
    int timeuse2 = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
    fprintf(f,"%d\n",timeuse2);
    fclose(f);
    **/
    FILE *f = fopen("inceptionv3-cifar100/mem.txt","a+");
    fprintf(f,"%zu\n",x.nbytes());
    fclose(f);
}



class CIFAR100 :public torch::data::datasets::Dataset<CIFAR100>{
public:

    torch::data::Example<> get(size_t index) {
        return {images_[index], targets_[index]};
    }
    torch::optional<size_t> size() const {
        return images_.size(0);
    }
    CIFAR100(){}
    CIFAR100(std::string path){
        images_ = torch::empty({50000, 3,32, 32}, torch::kByte);
        torch::Tensor targets_1 =  torch::empty(50000, torch::kByte);
        targets_ =  torch::empty(50000, torch::kByte);
        std::ifstream reader(path,std::ios::binary);
        for(int i=0;i<50000;i++){
            reader.read(reinterpret_cast<char*>(targets_1.data_ptr())+i,1);
            reader.read(reinterpret_cast<char*>(targets_.data_ptr())+i,1);
            reader.read(reinterpret_cast<char*>(images_.data_ptr())+i*3072,3072);
        }
        targets_ = targets_.to(torch::kInt64);
        images_ = images_.to(torch::kFloat32).div_(255);
    }

    void init(std::string path){
        images_ = torch::empty({50000, 3,32, 32}, torch::kByte);
        torch::Tensor targets_1 =  torch::empty(50000, torch::kByte);
        targets_ =  torch::empty(50000, torch::kByte);
        std::ifstream reader(path,std::ios::binary);
        for(int i=0;i<50000;i++){
            reader.read(reinterpret_cast<char*>(targets_1.data_ptr())+i,1);
            reader.read(reinterpret_cast<char*>(targets_.data_ptr())+i,1);
            reader.read(reinterpret_cast<char*>(images_.data_ptr())+i*3072,3072);
        }
        targets_ = targets_.to(torch::kInt64);
        images_ = images_.to(torch::kFloat32).div_(255);
    }

private:
    torch::Tensor images_,targets_;
};


struct Normalize : public torch::data::transforms::TensorTransform<> {
    Normalize(float mean, float stddev)
        : mean_(torch::tensor(mean)), stddev_(torch::tensor(stddev)) {}
    torch::Tensor operator()(torch::Tensor input) {
        input.resize_({3,36,36});
        return input.sub_(mean_).div_(stddev_);
    }
    torch::Tensor mean_, stddev_;
};



/**
struct Net1Impl : public torch::nn::Module{
    Net1Impl(){}
    Net1Impl(int64_t N, int64_t M){
        linear = torch::nn::Linear(N, M);
        register_module("linear",linear);
    }
    torch::nn::Linear linear{nullptr};
};

TORCH_MODULE(Net1);

struct Net2Impl : public torch::nn::Module{

    Net2Impl(){
        n = Net1(5,4);
        register_module("net1",n);
        linear = torch::nn::Linear(512,10);
        register_module("linear2", linear);
    }
    Net1 n;
    torch::nn::Linear linear{nullptr};
};

TORCH_MODULE(Net2);
**/

int index_ = -1;
struct timeval start_, end_;
struct timeval start_1, end_1;
int timeuse;
torch::Tensor pre;
cudaStream_t stream1;
size_t sizes;
at::DataPtr dataptr_;

vector<int> policy(320,0);
vector<int> begin_stage(320,0);

bool flaf = false;

enum Layer_type{
    CONV,
    POOL,
    ACT,
    BN,
    FC
};

bool is_super = false;
bool is_vdnn = false;

struct BasicConv2dImpl: torch::nn::Module{
    BasicConv2dImpl(){}
    BasicConv2dImpl(int in,int out,torch::ExpandingArray<2> kernel_size,torch::ExpandingArray<2> padding=0,torch::ExpandingArray<2> stride=1){
        conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in,out,kernel_size).stride(stride).padding(padding).bias(false));
        bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out));
        register_module("conv",conv);
        register_module("batchnorm",bn);
    }
    torch::Tensor forward(torch::Tensor& input){
       
        pre = input;
        index_++;
        
        if((pre.tensor_id>=0)&&((policy[pre.tensor_id]==2)||is_super||is_vdnn)&&(index_==begin_stage[pre.tensor_id])){
            //std::cout<<"ok;";
            if(c10::Passive::is_enabled()){
                if(c10::Profile2::is_enabled()){
                    gettimeofday(&start_1, NULL);
                }
                sizes = pre.nbytes();
                //dataptr_ = at::getCPUAllocator()->allocate(sizes);
                dataptr_ = at::detail::getCUDAHooks().getPinnedMemoryAllocator()->allocate(sizes);
                cudaStreamCreateWithFlags(&stream1,cudaStreamNonBlocking);
                //cudaDeviceSynchronize();
                cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
                if(c10::BreakdownProfile::is_enabled()){
                    gettimeofday(&start_1, NULL);
                }
                cudaMemcpyAsync(dataptr_.get(),pre.data_ptr(),sizes,cudaMemcpyDeviceToHost,stream1);
                //cudaStreamAddCallback(stream1, c10::MyCallback, (void*)index, 0);
            }              
        }
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&start_, NULL);
        }
        //std::cout<<pre.tensor_id<<"-";
        if(c10::BreakdownProfile::is_enabled()){
            gettimeofday(&start_, NULL);
        }
        torch::Tensor x = conv->forward(pre);
        x.tensor_id = index_;
        
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&end_, NULL);
            timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
            c10::GetPolicyMaker()->add_comptime(timeuse);
            x.grad_fn()->id_ = index_;
            x.grad_fn()->ids.push_back(pre.tensor_id);
            c10::GetPolicyMaker()->add_tensor(index_,x.nbytes());
            c10::GetPolicyMaker()->add_stepdep({pre.tensor_id,index_});
        }else{
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            if(c10::Profile3::is_enabled()){
                const  c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
                c10::GetCompressProfile()->add_memory_load((stats.allocated_bytes)[0].current);
                    //std::cout<<(stats.allocated_bytes)[0].current<<"-";
            }
            if(c10::BreakdownProfile::is_enabled()){
                gettimeofday(&end_, NULL);
                timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
                c10::GetBreakdownProfile()->add_compute_time(timeuse);
            }
        } 
        if((pre.tensor_id>=0)&&(policy[pre.tensor_id]==1)&&(index_==begin_stage[pre.tensor_id])){
            pre.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->data_ptr().clear();
            pre.grad_fn()->storage_impl_ = pre.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
            pre.grad_fn()->com_and_tran = 1;
            //x.grad_fn()->is_recomp = true;     
        }else if((pre.tensor_id>=0)&&((policy[pre.tensor_id]==2)||is_super||is_vdnn)&&(index_==begin_stage[pre.tensor_id])){
            if(c10::Passive::is_enabled()){
                if(c10::BreakdownProfile::is_enabled()){
                    gettimeofday(&start_, NULL);
                }
                cudaStreamSynchronize(stream1);
                if(c10::BreakdownProfile::is_enabled()){
                    gettimeofday(&end_1, NULL);
                    timeuse = 1000000 * ( end_1.tv_sec - start_1.tv_sec ) + end_1.tv_usec - start_1.tv_usec;                      
                    c10::GetBreakdownProfile()->add_transfer_time(timeuse);
                    timeuse = 1000000 * ( end_1.tv_sec - start_.tv_sec ) + end_1.tv_usec - start_.tv_usec;                      
                    c10::GetBreakdownProfile()->add_sync_time(timeuse);
                }
                pre.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->set_data_ptr(std::move(dataptr_));
                pre.grad_fn()->storage_impl_ = pre.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
                pre.grad_fn()->com_and_tran = 2;      
                if(c10::Profile2::is_enabled()){
                    gettimeofday(&end_1, NULL);
                    timeuse = 1000000 * ( end_1.tv_sec - start_1.tv_sec ) + end_1.tv_usec - start_1.tv_usec;
                    c10::GetCompressProfile()->add_transfer_time(timeuse);
                }
            }else{                
                pre.grad_fn()->com_and_tran = 2;
                pre.grad_fn()->storage_impl_ = pre.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
                at::GetTransferRun()->add_item(pre.nbytes(),pre.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl());
            }
        }
        x.grad_fn()->pre_node = pre.grad_fn();    
        x.grad_fn()->func_ = std::bind(&torch::nn::Conv2dImpl::forward,conv,std::placeholders::_1);
        x.grad_fn()->input_tensor = pre.variable_data();
        pre = x;
        index_++;
        
        if((pre.tensor_id>=0)&&(policy[pre.tensor_id]==2)&&(index_==begin_stage[pre.tensor_id])){
            //std::cout<<"ok;";
            if(c10::Passive::is_enabled()){
                if(c10::Profile2::is_enabled()){
                    gettimeofday(&start_1, NULL);
                }
                sizes = pre.nbytes();
                //dataptr_ = at::getCPUAllocator()->allocate(sizes);
                dataptr_ = at::detail::getCUDAHooks().getPinnedMemoryAllocator()->allocate(sizes);
                cudaStreamCreateWithFlags(&stream1,cudaStreamNonBlocking);
                //cudaDeviceSynchronize();
                cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
                if(c10::BreakdownProfile::is_enabled()){
                    gettimeofday(&start_1, NULL);
                }
                cudaMemcpyAsync(dataptr_.get(),pre.data_ptr(),sizes,cudaMemcpyDeviceToHost,stream1);
                //cudaStreamAddCallback(stream1, c10::MyCallback, (void*)index, 0);
            }              
        }
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&start_, NULL);
        }
        //std::cout<<pre.tensor_id<<"-";
        if(c10::BreakdownProfile::is_enabled()){
            gettimeofday(&start_, NULL);
        }
        x = bn->forward(pre);
        x.tensor_id = index_;
        
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&end_, NULL);
            timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
            c10::GetPolicyMaker()->add_comptime(timeuse);
            x.grad_fn()->id_ = index_;
            x.grad_fn()->ids.push_back(pre.tensor_id);
            c10::GetPolicyMaker()->add_tensor(index_,x.nbytes());
            c10::GetPolicyMaker()->add_stepdep({pre.tensor_id,index_});
        }else{
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            if(c10::Profile3::is_enabled()){
                const  c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
                c10::GetCompressProfile()->add_memory_load((stats.allocated_bytes)[0].current);
                    //std::cout<<(stats.allocated_bytes)[0].current<<"-";
            }
            if(c10::BreakdownProfile::is_enabled()){
                gettimeofday(&end_, NULL);
                timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
                c10::GetBreakdownProfile()->add_compute_time(timeuse);
            }
        } 
        if((pre.tensor_id>=0)&&((policy[pre.tensor_id]==1)||is_super)&&(index_==begin_stage[pre.tensor_id])){
            pre.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->data_ptr().clear();
            pre.grad_fn()->storage_impl_ = pre.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
            pre.grad_fn()->com_and_tran = 1;
            //x.grad_fn()->is_recomp = true;     
        }else if((pre.tensor_id>=0)&&(policy[pre.tensor_id]==2)&&(index_==begin_stage[pre.tensor_id])){
            if(c10::Passive::is_enabled()){
                if(c10::BreakdownProfile::is_enabled()){
                    gettimeofday(&start_, NULL);
                }
                cudaStreamSynchronize(stream1);
                if(c10::BreakdownProfile::is_enabled()){
                    gettimeofday(&end_1, NULL);
                    timeuse = 1000000 * ( end_1.tv_sec - start_1.tv_sec ) + end_1.tv_usec - start_1.tv_usec;                      
                    c10::GetBreakdownProfile()->add_transfer_time(timeuse);
                    timeuse = 1000000 * ( end_1.tv_sec - start_.tv_sec ) + end_1.tv_usec - start_.tv_usec;                      
                    c10::GetBreakdownProfile()->add_sync_time(timeuse);
                }
                pre.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->set_data_ptr(std::move(dataptr_));
                pre.grad_fn()->storage_impl_ = pre.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
                pre.grad_fn()->com_and_tran = 2;      
                if(c10::Profile2::is_enabled()){
                    gettimeofday(&end_1, NULL);
                    timeuse = 1000000 * ( end_1.tv_sec - start_1.tv_sec ) + end_1.tv_usec - start_1.tv_usec;
                    c10::GetCompressProfile()->add_transfer_time(timeuse);
                }
            }else{                
                pre.grad_fn()->com_and_tran = 2;
                pre.grad_fn()->storage_impl_ = pre.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
                at::GetTransferRun()->add_item(pre.nbytes(),pre.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl());
            }
        }
        x.grad_fn()->pre_node = pre.grad_fn();    
        x.grad_fn()->func_ = std::bind(&torch::nn::BatchNorm2dImpl::forward,bn,std::placeholders::_1);
        x.grad_fn()->input_tensor = pre.variable_data();
        
        pre = x;
        index_++;
        
        if((pre.tensor_id>=0)&&(policy[pre.tensor_id]==2)&&(index_==begin_stage[pre.tensor_id])){
            //std::cout<<"ok;";
            if(c10::Passive::is_enabled()){
                if(c10::Profile2::is_enabled()){
                    gettimeofday(&start_1, NULL);
                }
                sizes = pre.nbytes();
                //dataptr_ = at::getCPUAllocator()->allocate(sizes);
                dataptr_ = at::detail::getCUDAHooks().getPinnedMemoryAllocator()->allocate(sizes);
                cudaStreamCreateWithFlags(&stream1,cudaStreamNonBlocking);
                //cudaDeviceSynchronize();
                cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
                if(c10::BreakdownProfile::is_enabled()){
                    gettimeofday(&start_1, NULL);
                }
                cudaMemcpyAsync(dataptr_.get(),pre.data_ptr(),sizes,cudaMemcpyDeviceToHost,stream1);
                //cudaStreamAddCallback(stream1, c10::MyCallback, (void*)index, 0);
            }              
        }
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&start_, NULL);
        }
        //std::cout<<pre.tensor_id<<"-";
        if(c10::BreakdownProfile::is_enabled()){
            gettimeofday(&start_, NULL);
        }
        x = torch::relu(pre);
        x.tensor_id = index_;
        
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&end_, NULL);
            timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
            c10::GetPolicyMaker()->add_comptime(timeuse);
            x.grad_fn()->id_ = index_;
            x.grad_fn()->ids.push_back(pre.tensor_id);
            c10::GetPolicyMaker()->add_tensor(index_,x.nbytes());
            c10::GetPolicyMaker()->add_stepdep({pre.tensor_id,index_});
        }else{
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            if(c10::Profile3::is_enabled()){
                const  c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
                c10::GetCompressProfile()->add_memory_load((stats.allocated_bytes)[0].current);
                    //std::cout<<(stats.allocated_bytes)[0].current<<"-";
            }
            if(c10::BreakdownProfile::is_enabled()){
                gettimeofday(&end_, NULL);
                timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
                c10::GetBreakdownProfile()->add_compute_time(timeuse);
            }
        } 
        if((pre.tensor_id>=0)&&((policy[pre.tensor_id]==1)||is_super)&&(index_==begin_stage[pre.tensor_id])){
            pre.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->data_ptr().clear();
            pre.grad_fn()->storage_impl_ = pre.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
            pre.grad_fn()->com_and_tran = 1;
            //x.grad_fn()->is_recomp = true;     
        }else if((pre.tensor_id>=0)&&(policy[pre.tensor_id]==2)&&(index_==begin_stage[pre.tensor_id])){
            if(c10::Passive::is_enabled()){
                if(c10::BreakdownProfile::is_enabled()){
                    gettimeofday(&start_, NULL);
                }
                cudaStreamSynchronize(stream1);
                if(c10::BreakdownProfile::is_enabled()){
                    gettimeofday(&end_1, NULL);
                    timeuse = 1000000 * ( end_1.tv_sec - start_1.tv_sec ) + end_1.tv_usec - start_1.tv_usec;                      
                    c10::GetBreakdownProfile()->add_transfer_time(timeuse);
                    timeuse = 1000000 * ( end_1.tv_sec - start_.tv_sec ) + end_1.tv_usec - start_.tv_usec;                      
                    c10::GetBreakdownProfile()->add_sync_time(timeuse);
                }
                pre.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->set_data_ptr(std::move(dataptr_));
                pre.grad_fn()->storage_impl_ = pre.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
                pre.grad_fn()->com_and_tran = 2;      
                if(c10::Profile2::is_enabled()){
                    gettimeofday(&end_1, NULL);
                    timeuse = 1000000 * ( end_1.tv_sec - start_1.tv_sec ) + end_1.tv_usec - start_1.tv_usec;
                    c10::GetCompressProfile()->add_transfer_time(timeuse);
                }
            }else{                
                pre.grad_fn()->com_and_tran = 2;
                pre.grad_fn()->storage_impl_ = pre.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
                at::GetTransferRun()->add_item(pre.nbytes(),pre.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl());
            }
        }
        x.grad_fn()->pre_node = pre.grad_fn();    
        x.grad_fn()->func_ = torch::relu;
        x.grad_fn()->input_tensor = pre.variable_data();
        
        return x;
    }
    torch::nn::Conv2d conv{nullptr};
    torch::nn::BatchNorm2d bn{nullptr};

};
TORCH_MODULE(BasicConv2d);

struct InceptionAImpl: torch::nn::Module{

    InceptionAImpl(){}
    InceptionAImpl(int in,int out){
        basic1 = BasicConv2d(in,64,1);
        basic2 = BasicConv2d(in,48,1);
        basic3 = BasicConv2d(48,64,5,2);
        basic4 = BasicConv2d(in,64,1);
        basic5 = BasicConv2d(64,96,3,1);
        basic6 = BasicConv2d(96,96,3,1);
        basic7 = BasicConv2d(in,out,3,1);
        avg_pool = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(3).stride(1).padding(1));
        register_module("basicConv2d1",basic1);
        register_module("basicConv2d2",basic2);
        register_module("basicConv2d3",basic3);
        register_module("basicConv2d4",basic4);
        register_module("basicConv2d5",basic5);
        register_module("basicConv2d6",basic6);
        register_module("basicConv2d7",basic7);
    }
    torch::Tensor forward(torch::Tensor& input){
        torch::Tensor x1 = basic1->forward(input);
        torch::Tensor x2 = basic2->forward(input);
        x2 = basic3->forward(x2);
        torch::Tensor x3 = basic4->forward(input);
        x3 = basic5->forward(x3);
        x3 = basic6->forward(x3);


        index_++;
        //std::cout<<input.tensor_id<<"-";
        if((input.tensor_id>=0)&&(policy[input.tensor_id]==2)&&(index_==begin_stage[input.tensor_id])){
            if(c10::Passive::is_enabled()){
                if(c10::Profile2::is_enabled()){
                    gettimeofday(&start_1, NULL);
                }
                sizes = input.nbytes();
                //dataptr_ = at::getCPUAllocator()->allocate(sizes);
                dataptr_ = at::detail::getCUDAHooks().getPinnedMemoryAllocator()->allocate(sizes);
                cudaStreamCreateWithFlags(&stream1,cudaStreamNonBlocking);
                //cudaDeviceSynchronize();
                cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
                if(c10::BreakdownProfile::is_enabled()){
                    gettimeofday(&start_1, NULL);
                }
                cudaMemcpyAsync(dataptr_.get(),input.data_ptr(),sizes,cudaMemcpyDeviceToHost,stream1);
                //cudaStreamAddCallback(stream1, c10::MyCallback, (void*)index, 0);
            }              
        }
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&start_, NULL);
        }
        if(c10::BreakdownProfile::is_enabled()){
            gettimeofday(&start_, NULL);
        }
        torch::Tensor x4 = avg_pool->forward(input);
        x4.tensor_id = index_;

        
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&end_, NULL);
            timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
            c10::GetPolicyMaker()->add_comptime(timeuse);
            x4.grad_fn()->id_ = index_;
            x4.grad_fn()->ids.push_back(input.tensor_id);
            c10::GetPolicyMaker()->add_tensor(index_,x4.nbytes());
            c10::GetPolicyMaker()->add_stepdep({input.tensor_id,index_});
        }else{
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            if(c10::Profile3::is_enabled()){
                const  c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
                c10::GetCompressProfile()->add_memory_load((stats.allocated_bytes)[0].current);
                    //std::cout<<(stats.allocated_bytes)[0].current<<"-";
            }
            if(c10::BreakdownProfile::is_enabled()){
                gettimeofday(&end_, NULL);
                timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
                c10::GetBreakdownProfile()->add_compute_time(timeuse);
            }
        } 
        if((input.tensor_id>=0)&&((policy[input.tensor_id]==1)||is_super)&&(index_==begin_stage[input.tensor_id])){
            input.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->data_ptr().clear();
            input.grad_fn()->storage_impl_ = input.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
            input.grad_fn()->com_and_tran = 1;
            //x4.grad_fn()->is_recomp = true;     
        }else if((input.tensor_id>=0)&&(policy[input.tensor_id]==2)&&(index_==begin_stage[input.tensor_id])){
            if(c10::Passive::is_enabled()){
                if(c10::BreakdownProfile::is_enabled()){
                    gettimeofday(&start_, NULL);
                }
                cudaStreamSynchronize(stream1);
                if(c10::BreakdownProfile::is_enabled()){
                    gettimeofday(&end_1, NULL);
                    timeuse = 1000000 * ( end_1.tv_sec - start_1.tv_sec ) + end_1.tv_usec - start_1.tv_usec;                      
                    c10::GetBreakdownProfile()->add_transfer_time(timeuse);
                    timeuse = 1000000 * ( end_1.tv_sec - start_.tv_sec ) + end_1.tv_usec - start_.tv_usec;                      
                    c10::GetBreakdownProfile()->add_sync_time(timeuse);
                }
                input.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->set_data_ptr(std::move(dataptr_));
                input.grad_fn()->storage_impl_ = input.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
                input.grad_fn()->com_and_tran = 2;      
                if(c10::Profile2::is_enabled()){
                    gettimeofday(&end_1, NULL);
                    timeuse = 1000000 * ( end_1.tv_sec - start_1.tv_sec ) + end_1.tv_usec - start_1.tv_usec;
                    c10::GetCompressProfile()->add_transfer_time(timeuse);
                }
            }else{                
                input.grad_fn()->com_and_tran = 2;
                input.grad_fn()->storage_impl_ = input.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
                at::GetTransferRun()->add_item(input.nbytes(),input.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl());
            }
        }
        x4.grad_fn()->pre_node = input.grad_fn();    
        x4.grad_fn()->func_ = std::bind(&torch::nn::AvgPool2dImpl::forward,avg_pool,std::placeholders::_1);
        x4.grad_fn()->input_tensor = input.variable_data();


      
        x4 = basic7->forward(x4);

        index_++;
        //std::cout<<x1.tensor_id<<";"<<x2.tensor_id<<";"<<x3.tensor_id<<";"<<x4.tensor_id<<"-";
        std::vector<torch::Tensor> inputs{x1,x2,x3,x4};
        std::vector<c10::DataPtr> dataptrs(4);
        cudaStream_t stream[4];

        for(int j=0;j<4;j++){
            if((inputs[j].tensor_id>=0)&&(policy[inputs[j].tensor_id]==2)&&(index_==begin_stage[inputs[j].tensor_id])){
                if(c10::Passive::is_enabled()){
                    //if(c10::Profile2::is_enabled()){
                    //    gettimeofday(&start_1, NULL);
                    //}
                    sizes = inputs[j].nbytes();
                    //dataptr_ = at::getCPUAllocator()->allocate(sizes);
                    dataptrs[j] = at::detail::getCUDAHooks().getPinnedMemoryAllocator()->allocate(sizes);
                    cudaStreamCreateWithFlags(&stream[j],cudaStreamNonBlocking);
                    //cudaDeviceSynchronize();
                    //cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
                    cudaMemcpyAsync(dataptrs[j].get(),inputs[j].data_ptr(),sizes,cudaMemcpyDeviceToHost,stream[j]);
                    //cudaStreamAddCallback(stream1, c10::MyCallback, (void*)index, 0);
                }              
            }
        }
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&start_, NULL);
        }

        torch::Tensor x5 = torch::cat({x1,x2,x3,x4},1);
        x5.tensor_id = index_;

        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&end_, NULL);
            timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
            c10::GetPolicyMaker()->add_comptime(timeuse);
            x5.grad_fn()->id_ = index_;
            x5.grad_fn()->ids.push_back(x1.tensor_id);
            x5.grad_fn()->ids.push_back(x2.tensor_id);
            x5.grad_fn()->ids.push_back(x3.tensor_id);
            x5.grad_fn()->ids.push_back(x4.tensor_id);
            c10::GetPolicyMaker()->add_tensor(index_,x5.nbytes());
            c10::GetPolicyMaker()->add_stepdep({x1.tensor_id,x2.tensor_id,x3.tensor_id,x4.tensor_id,index_});
        }else{
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            if(c10::Profile3::is_enabled()){
                const  c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
                c10::GetCompressProfile()->add_memory_load((stats.allocated_bytes)[0].current);
                    //std::cout<<(stats.allocated_bytes)[0].current<<"-";
            }
        } 
        x5.grad_fn()->is_cat = true;
       // x5.grad_fn()->multiple_input.com_and_trans.resize(4,-1);
        x5.grad_fn()->multiple_input.input_tensors.resize(4);
        x5.grad_fn()->multiple_input.pre_nodes.resize(4,nullptr);
        //x5.grad_fn()->multiple_input.storage_impl_s.resize(4,nullptr);
        //x5.grad_fn()->multiple_input.dataptrs.resize(4);
        
        //int cnt_ = 0;

        for(int j=0;j<4;j++){
            if((inputs[j].tensor_id>=0)&&(policy[inputs[j].tensor_id]==1)&&(index_==begin_stage[inputs[j].tensor_id])){
                inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->data_ptr().clear();
                inputs[j].grad_fn()->storage_impl_ = inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
                inputs[j].grad_fn()->com_and_tran = 1;
                

            }else if((inputs[j].tensor_id>=0)&&(policy[inputs[j].tensor_id]==2)&&(index_==begin_stage[inputs[j].tensor_id])){
                //cnt_++;
                if(c10::Passive::is_enabled()){
                    cudaStreamSynchronize(stream[j]);
                    inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->set_data_ptr(std::move(dataptrs[j]));   
                    /**          
                    if(c10::Profile2::is_enabled()){
                        gettimeofday(&end_1, NULL);
                        timeuse = 1000000 * ( end_1.tv_sec - start_1.tv_sec ) + end_1.tv_usec - start_1.tv_usec;
                        c10::GetCompressProfile()->add_transfer_time(timeuse);
                    }**/
                }else{
                     at::GetTransferRun()->add_item(inputs[j].nbytes(),inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl());
                }
                inputs[j].grad_fn()->com_and_tran = 2;
                inputs[j].grad_fn()->storage_impl_ = inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();

            }
            x5.grad_fn()->multiple_input.input_tensors[j] = inputs[j].variable_data();
            x5.grad_fn()->multiple_input.pre_nodes[j] = inputs[j].grad_fn();
        }
        //if(cnt_>1) flaf = true;
        return x5;
    }
    BasicConv2d basic1;
    BasicConv2d basic2;
    BasicConv2d basic3;
    BasicConv2d basic4;
    BasicConv2d basic5;
    BasicConv2d basic6;
    BasicConv2d basic7;
    torch::nn::AvgPool2d avg_pool{nullptr};

};
TORCH_MODULE(InceptionA);

struct InceptionBImpl:torch::nn::Module{
    InceptionBImpl(){}
    InceptionBImpl(int in){
        basic1 = BasicConv2d(in,384,3,0,2);
        basic2 = BasicConv2d(in,64,1);
        basic3 = BasicConv2d(64,96,3,1);
        basic4 = BasicConv2d(96,96,3,0,2);
        register_module("basicConv2d1",basic1);
        register_module("basicConv2d2",basic2);
        register_module("basicConv2d3",basic3);
        register_module("basicConv2d4",basic4);
        maxpool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2));
        
    }
    torch::Tensor forward(torch::Tensor& input){
        torch::Tensor x1 = basic1->forward(input);
        torch::Tensor x2 = basic2->forward(input);
        x2 = basic3->forward(x2);
        x2 = basic4->forward(x2);
       
        index_++;
        //std::cout<<input.tensor_id<<"-";
        if((input.tensor_id>=0)&&(policy[input.tensor_id]==2)&&(index_==begin_stage[input.tensor_id])){
            if(c10::Passive::is_enabled()){
                if(c10::Profile2::is_enabled()){
                    gettimeofday(&start_1, NULL);
                }
                sizes = input.nbytes();
                //dataptr_ = at::getCPUAllocator()->allocate(sizes);
                dataptr_ = at::detail::getCUDAHooks().getPinnedMemoryAllocator()->allocate(sizes);
                cudaStreamCreateWithFlags(&stream1,cudaStreamNonBlocking);
                //cudaDeviceSynchronize();
                cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
                if(c10::BreakdownProfile::is_enabled()){
                    gettimeofday(&start_1, NULL);
                }
                cudaMemcpyAsync(dataptr_.get(),input.data_ptr(),sizes,cudaMemcpyDeviceToHost,stream1);
                //cudaStreamAddCallback(stream1, c10::MyCallback, (void*)index, 0);
            }              
        }
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&start_, NULL);
        }
        if(c10::BreakdownProfile::is_enabled()){
            gettimeofday(&start_, NULL);
        }
        torch::Tensor x3 = maxpool->forward(input);
        x3.tensor_id = index_;
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&end_, NULL);
            timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
            c10::GetPolicyMaker()->add_comptime(timeuse);
            x3.grad_fn()->id_ = index_;
            x3.grad_fn()->ids.push_back(input.tensor_id);
            c10::GetPolicyMaker()->add_tensor(index_,x3.nbytes());
            c10::GetPolicyMaker()->add_stepdep({input.tensor_id,index_});
        }else{
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            if(c10::Profile3::is_enabled()){
                const  c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
                c10::GetCompressProfile()->add_memory_load((stats.allocated_bytes)[0].current);
                    //std::cout<<(stats.allocated_bytes)[0].current<<"-";
            }
            if(c10::BreakdownProfile::is_enabled()){
                gettimeofday(&end_, NULL);
                timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
                c10::GetBreakdownProfile()->add_compute_time(timeuse);
            }
        } 
        if((input.tensor_id>=0)&&((policy[input.tensor_id]==1)||is_super)&&(index_==begin_stage[input.tensor_id])){
            input.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->data_ptr().clear();
            input.grad_fn()->storage_impl_ = input.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
            input.grad_fn()->com_and_tran = 1;
            //x3.grad_fn()->is_recomp = true;     
        }else if((input.tensor_id>=0)&&(policy[input.tensor_id]==2)&&(index_==begin_stage[input.tensor_id])){
            if(c10::Passive::is_enabled()){
                if(c10::BreakdownProfile::is_enabled()){
                    gettimeofday(&start_, NULL);
                }
                cudaStreamSynchronize(stream1);
                if(c10::BreakdownProfile::is_enabled()){
                    gettimeofday(&end_1, NULL);
                    timeuse = 1000000 * ( end_1.tv_sec - start_1.tv_sec ) + end_1.tv_usec - start_1.tv_usec;                      
                    c10::GetBreakdownProfile()->add_transfer_time(timeuse);
                    timeuse = 1000000 * ( end_1.tv_sec - start_.tv_sec ) + end_1.tv_usec - start_.tv_usec;                      
                    c10::GetBreakdownProfile()->add_sync_time(timeuse);
                }
                input.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->set_data_ptr(std::move(dataptr_));
                input.grad_fn()->storage_impl_ = input.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
                input.grad_fn()->com_and_tran = 2;      
                if(c10::Profile2::is_enabled()){
                    gettimeofday(&end_1, NULL);
                    timeuse = 1000000 * ( end_1.tv_sec - start_1.tv_sec ) + end_1.tv_usec - start_1.tv_usec;
                    c10::GetCompressProfile()->add_transfer_time(timeuse);
                }
            }else{                
                input.grad_fn()->com_and_tran = 2;
                input.grad_fn()->storage_impl_ = input.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
                at::GetTransferRun()->add_item(input.nbytes(),input.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl());
            }
        }
        x3.grad_fn()->pre_node = input.grad_fn();    
        x3.grad_fn()->func_ = std::bind(&torch::nn::MaxPool2dImpl::forward,maxpool,std::placeholders::_1);
        x3.grad_fn()->input_tensor = input.variable_data();


        index_++;
        //std::cout<<x1.tensor_id<<";"<<x2.tensor_id<<";"<<x3.tensor_id<<"-";
        std::vector<torch::Tensor> inputs{x1,x2,x3};
        std::vector<c10::DataPtr> dataptrs(3);
        cudaStream_t stream[3];

        for(int j=0;j<3;j++){
            if((inputs[j].tensor_id>=0)&&(policy[inputs[j].tensor_id]==2)&&(index_==begin_stage[inputs[j].tensor_id])){
                if(c10::Passive::is_enabled()){
                    //if(c10::Profile2::is_enabled()){
                    //    gettimeofday(&start_1, NULL);
                    //}
                    sizes = inputs[j].nbytes();
                    //dataptr_ = at::getCPUAllocator()->allocate(sizes);
                    dataptrs[j] = at::detail::getCUDAHooks().getPinnedMemoryAllocator()->allocate(sizes);
                    cudaStreamCreateWithFlags(&stream[j],cudaStreamNonBlocking);
                    //cudaDeviceSynchronize();
                    //cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
                    cudaMemcpyAsync(dataptrs[j].get(),inputs[j].data_ptr(),sizes,cudaMemcpyDeviceToHost,stream[j]);
                    //cudaStreamAddCallback(stream1, c10::MyCallback, (void*)index, 0);
                }              
            }
        }
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&start_, NULL);
        }
        torch::Tensor x4 = torch::cat({x1,x2,x3},1);
        x4.tensor_id = index_;
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&end_, NULL);
            timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
            c10::GetPolicyMaker()->add_comptime(timeuse);
            x4.grad_fn()->id_ = index_;
            x4.grad_fn()->ids.push_back(x1.tensor_id);
            x4.grad_fn()->ids.push_back(x2.tensor_id);
            x4.grad_fn()->ids.push_back(x3.tensor_id);
            c10::GetPolicyMaker()->add_tensor(index_,x4.nbytes());
            c10::GetPolicyMaker()->add_stepdep({x1.tensor_id,x2.tensor_id,x3.tensor_id,index_});
        }else{
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            if(c10::Profile3::is_enabled()){
                const  c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
                c10::GetCompressProfile()->add_memory_load((stats.allocated_bytes)[0].current);
                    //std::cout<<(stats.allocated_bytes)[0].current<<"-";
            }
        } 
        x4.grad_fn()->is_cat = true;
        //x4.grad_fn()->multiple_input.com_and_trans.resize(3,-1);
        x4.grad_fn()->multiple_input.input_tensors.resize(3);
        x4.grad_fn()->multiple_input.pre_nodes.resize(3,nullptr);
        //x4.grad_fn()->multiple_input.storage_impl_s.resize(3,nullptr);
        //x4.grad_fn()->multiple_input.dataptrs.resize(3);
        //int cnt_ = 0;
        for(int j=0;j<3;j++){
            if((inputs[j].tensor_id>=0)&&(policy[inputs[j].tensor_id]==1)&&(index_==begin_stage[inputs[j].tensor_id])){
                inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->data_ptr().clear();
                inputs[j].grad_fn()->storage_impl_ = inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
                inputs[j].grad_fn()->com_and_tran = 1;
                

            }else if((inputs[j].tensor_id>=0)&&(policy[inputs[j].tensor_id]==2)&&(index_==begin_stage[inputs[j].tensor_id])){
                //cnt_++;
                if(c10::Passive::is_enabled()){
                    cudaStreamSynchronize(stream[j]);
                    inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->set_data_ptr(std::move(dataptrs[j]));   
                    /**          
                    if(c10::Profile2::is_enabled()){
                        gettimeofday(&end_1, NULL);
                        timeuse = 1000000 * ( end_1.tv_sec - start_1.tv_sec ) + end_1.tv_usec - start_1.tv_usec;
                        c10::GetCompressProfile()->add_transfer_time(timeuse);
                    }**/
                }else{
                     at::GetTransferRun()->add_item(inputs[j].nbytes(),inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl());
                }
                inputs[j].grad_fn()->com_and_tran = 2;
                inputs[j].grad_fn()->storage_impl_ = inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();

            }
            x4.grad_fn()->multiple_input.input_tensors[j] = inputs[j].variable_data();
            x4.grad_fn()->multiple_input.pre_nodes[j] = inputs[j].grad_fn();
        }
        //if(cnt_>1) flaf = true;
        return x4;
    }

    BasicConv2d basic1;
    BasicConv2d basic2;
    BasicConv2d basic3;
    BasicConv2d basic4;
    torch::nn::MaxPool2d maxpool{nullptr};

};
TORCH_MODULE(InceptionB);
struct InceptionCImpl:torch::nn::Module{
    InceptionCImpl(){}
    InceptionCImpl(int in,int out){
        basic1 = BasicConv2d(in,192,1);
        basic2 = BasicConv2d(in,out,1);
        basic3 = BasicConv2d(out,out,torch::ExpandingArray<2>({7,1}),torch::ExpandingArray<2>({3,0}));
        basic4 = BasicConv2d(out,192,torch::ExpandingArray<2>({1,7}),torch::ExpandingArray<2>({0,3}));
        basic5 = BasicConv2d(in,out,1);
        basic6 = BasicConv2d(out,out,torch::ExpandingArray<2>({7,1}),torch::ExpandingArray<2>({3,0}));
        basic7 = BasicConv2d(out,out,torch::ExpandingArray<2>({1,7}),torch::ExpandingArray<2>({0,3}));
        basic8 = BasicConv2d(out,out,torch::ExpandingArray<2>({7,1}),torch::ExpandingArray<2>({3,0}));
        basic9 = BasicConv2d(out,192,torch::ExpandingArray<2>({1,7}),torch::ExpandingArray<2>({0,3}));
        basic10 = BasicConv2d(in,192,1);
        register_module("basicConv2d1",basic1);
        register_module("basicConv2d2",basic2);
        register_module("basicConv2d3",basic3);
        register_module("basicConv2d4",basic4);
        register_module("basicConv2d5",basic5);
        register_module("basicConv2d6",basic6);
        register_module("basicConv2d7",basic7);
        register_module("basicConv2d8",basic8);
        register_module("basicConv2d9",basic9);
        register_module("basicConv2d10",basic10);
        avg_pool = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(3).stride(1).padding(1));
    }

    torch::Tensor forward(torch::Tensor& input){
        torch::Tensor x1 = basic1->forward(input);
        torch::Tensor x2 = basic2->forward(input);
        x2 = basic3->forward(x2);
        x2 = basic4->forward(x2);
        torch::Tensor x3 = basic5->forward(input);
        x3 = basic6->forward(x3);
        x3 = basic7->forward(x3);
        x3 = basic8->forward(x3);
        x3 = basic9->forward(x3);

        index_++;
        //std::cout<<input.tensor_id<<"-";
        if((input.tensor_id>=0)&&(policy[input.tensor_id]==2)&&(index_==begin_stage[input.tensor_id])){
            if(c10::Passive::is_enabled()){
                if(c10::Profile2::is_enabled()){
                    gettimeofday(&start_1, NULL);
                }
                sizes = input.nbytes();
                //dataptr_ = at::getCPUAllocator()->allocate(sizes);
                dataptr_ = at::detail::getCUDAHooks().getPinnedMemoryAllocator()->allocate(sizes);
                cudaStreamCreateWithFlags(&stream1,cudaStreamNonBlocking);
                //cudaDeviceSynchronize();
                cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
                if(c10::BreakdownProfile::is_enabled()){
                    gettimeofday(&start_1, NULL);
                }
                cudaMemcpyAsync(dataptr_.get(),input.data_ptr(),sizes,cudaMemcpyDeviceToHost,stream1);
                //cudaStreamAddCallback(stream1, c10::MyCallback, (void*)index, 0);
            }              
        }
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&start_, NULL);
        }
        if(c10::BreakdownProfile::is_enabled()){
            gettimeofday(&start_, NULL);
        }
        torch::Tensor x4 = avg_pool->forward(input);
        x4.tensor_id = index_;

        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&end_, NULL);
            timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
            c10::GetPolicyMaker()->add_comptime(timeuse);
            x4.grad_fn()->id_ = index_;
            x4.grad_fn()->ids.push_back(input.tensor_id);
            c10::GetPolicyMaker()->add_tensor(index_,x4.nbytes());
            c10::GetPolicyMaker()->add_stepdep({input.tensor_id,index_});
        }else{
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            if(c10::Profile3::is_enabled()){
                const  c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
                c10::GetCompressProfile()->add_memory_load((stats.allocated_bytes)[0].current);
                    //std::cout<<(stats.allocated_bytes)[0].current<<"-";
            }
            if(c10::BreakdownProfile::is_enabled()){
                gettimeofday(&end_, NULL);
                timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
                c10::GetBreakdownProfile()->add_compute_time(timeuse);
            }
        } 
        if((input.tensor_id>=0)&&((policy[input.tensor_id]==1)||is_super)&&(index_==begin_stage[input.tensor_id])){
            input.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->data_ptr().clear();
            input.grad_fn()->storage_impl_ = input.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
            input.grad_fn()->com_and_tran = 1;
            //x4.grad_fn()->is_recomp = true;     
        }else if((input.tensor_id>=0)&&(policy[input.tensor_id]==2)&&(index_==begin_stage[input.tensor_id])){
            if(c10::Passive::is_enabled()){
                if(c10::BreakdownProfile::is_enabled()){
                    gettimeofday(&start_, NULL);
                }
                cudaStreamSynchronize(stream1);
                if(c10::BreakdownProfile::is_enabled()){
                    gettimeofday(&end_1, NULL);
                    timeuse = 1000000 * ( end_1.tv_sec - start_1.tv_sec ) + end_1.tv_usec - start_1.tv_usec;                      
                    c10::GetBreakdownProfile()->add_transfer_time(timeuse);
                    timeuse = 1000000 * ( end_1.tv_sec - start_.tv_sec ) + end_1.tv_usec - start_.tv_usec;                      
                    c10::GetBreakdownProfile()->add_sync_time(timeuse);
                }
                input.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->set_data_ptr(std::move(dataptr_));
                input.grad_fn()->storage_impl_ = input.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
                input.grad_fn()->com_and_tran = 2;      
                if(c10::Profile2::is_enabled()){
                    gettimeofday(&end_1, NULL);
                    timeuse = 1000000 * ( end_1.tv_sec - start_1.tv_sec ) + end_1.tv_usec - start_1.tv_usec;
                    c10::GetCompressProfile()->add_transfer_time(timeuse);
                }
            }else{                
                input.grad_fn()->com_and_tran = 2;
                input.grad_fn()->storage_impl_ = input.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
                at::GetTransferRun()->add_item(input.nbytes(),input.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl());
            }
        }
        x4.grad_fn()->pre_node = input.grad_fn();    
        x4.grad_fn()->func_ = std::bind(&torch::nn::AvgPool2dImpl::forward,avg_pool,std::placeholders::_1);
        x4.grad_fn()->input_tensor = input.variable_data();


        x4 = basic10->forward(x4);

        index_++;
        //std::cout<<x1.tensor_id<<";"<<x2.tensor_id<<";"<<x3.tensor_id<<";"<<x4.tensor_id<<"-";
        std::vector<torch::Tensor> inputs{x1,x2,x3,x4};
        std::vector<c10::DataPtr> dataptrs(4);
        cudaStream_t stream[4];

        for(int j=0;j<4;j++){
            if((inputs[j].tensor_id>=0)&&(policy[inputs[j].tensor_id]==2)&&(index_==begin_stage[inputs[j].tensor_id])){
                if(c10::Passive::is_enabled()){
                    //if(c10::Profile2::is_enabled()){
                    //    gettimeofday(&start_1, NULL);
                    //}
                    sizes = inputs[j].nbytes();
                    //dataptr_ = at::getCPUAllocator()->allocate(sizes);
                    dataptrs[j] = at::detail::getCUDAHooks().getPinnedMemoryAllocator()->allocate(sizes);
                    cudaStreamCreateWithFlags(&stream[j],cudaStreamNonBlocking);
                    //cudaDeviceSynchronize();
                    //cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
                    cudaMemcpyAsync(dataptrs[j].get(),inputs[j].data_ptr(),sizes,cudaMemcpyDeviceToHost,stream[j]);
                    //cudaStreamAddCallback(stream1, c10::MyCallback, (void*)index, 0);
                }              
            }
        }

        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&start_, NULL);
        }
        torch::Tensor x5 = torch::cat({x1,x2,x3,x4},1);
        x5.tensor_id = index_;
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&end_, NULL);
            timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
            c10::GetPolicyMaker()->add_comptime(timeuse);
            x5.grad_fn()->id_ = index_;
            x5.grad_fn()->ids.push_back(x1.tensor_id);
            x5.grad_fn()->ids.push_back(x2.tensor_id);
            x5.grad_fn()->ids.push_back(x3.tensor_id);
            x5.grad_fn()->ids.push_back(x4.tensor_id);
            c10::GetPolicyMaker()->add_tensor(index_,x5.nbytes());
            c10::GetPolicyMaker()->add_stepdep({x1.tensor_id,x2.tensor_id,x3.tensor_id,x4.tensor_id,index_});
        }else{
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            if(c10::Profile3::is_enabled()){
                const  c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
                c10::GetCompressProfile()->add_memory_load((stats.allocated_bytes)[0].current);
                    //std::cout<<(stats.allocated_bytes)[0].current<<"-";
            }
        } 
        x5.grad_fn()->is_cat = true;
        //x5.grad_fn()->multiple_input.com_and_trans.resize(4,-1);
        x5.grad_fn()->multiple_input.input_tensors.resize(4);
        x5.grad_fn()->multiple_input.pre_nodes.resize(4,nullptr);
        //x5.grad_fn()->multiple_input.storage_impl_s.resize(4,nullptr);
        //x5.grad_fn()->multiple_input.dataptrs.resize(4);
        //int cnt_ = 0;
        for(int j=0;j<4;j++){
            if((inputs[j].tensor_id>=0)&&(policy[inputs[j].tensor_id]==1)&&(index_==begin_stage[inputs[j].tensor_id])){
                inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->data_ptr().clear();
                inputs[j].grad_fn()->storage_impl_ = inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
                inputs[j].grad_fn()->com_and_tran = 1;
                

            }else if((inputs[j].tensor_id>=0)&&(policy[inputs[j].tensor_id]==2)&&(index_==begin_stage[inputs[j].tensor_id])){
                //cnt_++;
                if(c10::Passive::is_enabled()){
                    cudaStreamSynchronize(stream[j]);
                    inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->set_data_ptr(std::move(dataptrs[j]));   
                    /**          
                    if(c10::Profile2::is_enabled()){
                        gettimeofday(&end_1, NULL);
                        timeuse = 1000000 * ( end_1.tv_sec - start_1.tv_sec ) + end_1.tv_usec - start_1.tv_usec;
                        c10::GetCompressProfile()->add_transfer_time(timeuse);
                    }**/
                }else{
                    at::GetTransferRun()->add_item(inputs[j].nbytes(),inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl());
                }
                inputs[j].grad_fn()->com_and_tran = 2;
                inputs[j].grad_fn()->storage_impl_ = inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();

            }
            x5.grad_fn()->multiple_input.input_tensors[j] = inputs[j].variable_data();
            x5.grad_fn()->multiple_input.pre_nodes[j] = inputs[j].grad_fn();
        }
        //if(cnt_>1) flaf = true;
        return x5;
    }


    BasicConv2d basic1;
    BasicConv2d basic2;
    BasicConv2d basic3;
    BasicConv2d basic4;
    BasicConv2d basic5;
    BasicConv2d basic6;
    BasicConv2d basic7;
    BasicConv2d basic8;
    BasicConv2d basic9;
    BasicConv2d basic10;
    torch::nn::AvgPool2d avg_pool{nullptr};
    
};
TORCH_MODULE(InceptionC);

struct InceptionDImpl:torch::nn::Module{
    InceptionDImpl(){}
    InceptionDImpl(int in){
        basic1 = BasicConv2d(in,192,1);
        basic2 = BasicConv2d(192,320,3,0,2);
        basic3 = BasicConv2d(in,192,1);
        basic4 = BasicConv2d(192,192,torch::ExpandingArray<2>({1,7}),torch::ExpandingArray<2>({0,3}));
        basic5 = BasicConv2d(192,192,torch::ExpandingArray<2>({7,1}),torch::ExpandingArray<2>({3,0}));
        basic6 = BasicConv2d(192,192,3,0,2);
        register_module("basicConv2d1",basic1);
        register_module("basicConv2d2",basic2);
        register_module("basicConv2d3",basic3);
        register_module("basicConv2d4",basic4);
        register_module("basicConv2d5",basic5);
        register_module("basicConv2d6",basic6);
        maxpool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2));
    }

    torch::Tensor forward(torch::Tensor& input){
        torch::Tensor x1 = basic1->forward(input);
        x1 = basic2->forward(x1);
        torch::Tensor x2 = basic3->forward(input);
        x2 = basic4->forward(x2);
        x2 = basic5->forward(x2);
        x2 = basic6->forward(x2);
        
        index_++;
        //std::cout<<input.tensor_id<<"-";
        if((input.tensor_id>=0)&&(policy[input.tensor_id]==2)&&(index_==begin_stage[input.tensor_id])){
            if(c10::Passive::is_enabled()){
                if(c10::Profile2::is_enabled()){
                    gettimeofday(&start_1, NULL);
                }
                sizes = input.nbytes();
                //dataptr_ = at::getCPUAllocator()->allocate(sizes);
                dataptr_ = at::detail::getCUDAHooks().getPinnedMemoryAllocator()->allocate(sizes);
                cudaStreamCreateWithFlags(&stream1,cudaStreamNonBlocking);
                //cudaDeviceSynchronize();
                cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
                if(c10::BreakdownProfile::is_enabled()){
                    gettimeofday(&start_1, NULL);
                }
                cudaMemcpyAsync(dataptr_.get(),input.data_ptr(),sizes,cudaMemcpyDeviceToHost,stream1);
                //cudaStreamAddCallback(stream1, c10::MyCallback, (void*)index, 0);
            }              
        }
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&start_, NULL);
        }
        if(c10::BreakdownProfile::is_enabled()){
            gettimeofday(&start_, NULL);
        }
        torch::Tensor x3 = maxpool->forward(input);
        x3.tensor_id = index_;
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&end_, NULL);
            timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
            c10::GetPolicyMaker()->add_comptime(timeuse);
            x3.grad_fn()->id_ = index_;
            x3.grad_fn()->ids.push_back(input.tensor_id);
            c10::GetPolicyMaker()->add_tensor(index_,x3.nbytes());
            c10::GetPolicyMaker()->add_stepdep({input.tensor_id,index_});
        }else{
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            if(c10::Profile3::is_enabled()){
                const  c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
                c10::GetCompressProfile()->add_memory_load((stats.allocated_bytes)[0].current);
                    //std::cout<<(stats.allocated_bytes)[0].current<<"-";
            }
            if(c10::BreakdownProfile::is_enabled()){
                gettimeofday(&end_, NULL);
                timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
                c10::GetBreakdownProfile()->add_compute_time(timeuse);
            }
        } 
        if((input.tensor_id>=0)&&((policy[input.tensor_id]==1)||is_super)&&(index_==begin_stage[input.tensor_id])){
            input.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->data_ptr().clear();
            input.grad_fn()->storage_impl_ = input.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
            input.grad_fn()->com_and_tran = 1;
            //x3.grad_fn()->is_recomp = true;     
        }else if((input.tensor_id>=0)&&(policy[input.tensor_id]==2)&&(index_==begin_stage[input.tensor_id])){
            if(c10::Passive::is_enabled()){
                if(c10::BreakdownProfile::is_enabled()){
                    gettimeofday(&start_, NULL);
                }
                cudaStreamSynchronize(stream1);
                if(c10::BreakdownProfile::is_enabled()){
                    gettimeofday(&end_1, NULL);
                    timeuse = 1000000 * ( end_1.tv_sec - start_1.tv_sec ) + end_1.tv_usec - start_1.tv_usec;                      
                    c10::GetBreakdownProfile()->add_transfer_time(timeuse);
                    timeuse = 1000000 * ( end_1.tv_sec - start_.tv_sec ) + end_1.tv_usec - start_.tv_usec;                      
                    c10::GetBreakdownProfile()->add_sync_time(timeuse);
                }
                input.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->set_data_ptr(std::move(dataptr_));
                input.grad_fn()->storage_impl_ = input.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
                input.grad_fn()->com_and_tran = 2;      
                if(c10::Profile2::is_enabled()){
                    gettimeofday(&end_1, NULL);
                    timeuse = 1000000 * ( end_1.tv_sec - start_1.tv_sec ) + end_1.tv_usec - start_1.tv_usec;
                    c10::GetCompressProfile()->add_transfer_time(timeuse);
                }
            }else{                
                input.grad_fn()->com_and_tran = 2;
                input.grad_fn()->storage_impl_ = input.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
                at::GetTransferRun()->add_item(input.nbytes(),input.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl());
            }
        }
        x3.grad_fn()->pre_node = input.grad_fn();    
        x3.grad_fn()->func_ = std::bind(&torch::nn::MaxPool2dImpl::forward,maxpool,std::placeholders::_1);
        x3.grad_fn()->input_tensor = input.variable_data();
        index_++;
        //std::cout<<x1.tensor_id<<";"<<x2.tensor_id<<";"<<x3.tensor_id<<"-";
        std::vector<torch::Tensor> inputs{x1,x2,x3};
        std::vector<c10::DataPtr> dataptrs(3);
        cudaStream_t stream[3];

        for(int j=0;j<3;j++){
            if((inputs[j].tensor_id>=0)&&(policy[inputs[j].tensor_id]==2)&&(index_==begin_stage[inputs[j].tensor_id])){
                if(c10::Passive::is_enabled()){
                    //if(c10::Profile2::is_enabled()){
                    //    gettimeofday(&start_1, NULL);
                    //}
                    sizes = inputs[j].nbytes();
                    //dataptr_ = at::getCPUAllocator()->allocate(sizes);
                    dataptrs[j] = at::detail::getCUDAHooks().getPinnedMemoryAllocator()->allocate(sizes);
                    cudaStreamCreateWithFlags(&stream[j],cudaStreamNonBlocking);
                    //cudaDeviceSynchronize();
                    //cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
                    cudaMemcpyAsync(dataptrs[j].get(),inputs[j].data_ptr(),sizes,cudaMemcpyDeviceToHost,stream[j]);
                    //cudaStreamAddCallback(stream1, c10::MyCallback, (void*)index, 0);
                }              
            }
        }
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&start_, NULL);
        }
        torch::Tensor x4 = torch::cat({x1,x2,x3},1);
        x4.tensor_id = index_;
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&end_, NULL);
            timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
            c10::GetPolicyMaker()->add_comptime(timeuse);
            x4.grad_fn()->id_ = index_;
            x4.grad_fn()->ids.push_back(x1.tensor_id);
            x4.grad_fn()->ids.push_back(x2.tensor_id);
            x4.grad_fn()->ids.push_back(x3.tensor_id);
            c10::GetPolicyMaker()->add_tensor(index_,x4.nbytes());
            c10::GetPolicyMaker()->add_stepdep({x1.tensor_id,x2.tensor_id,x3.tensor_id,index_});
        }else{
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            if(c10::Profile3::is_enabled()){
                const  c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
                c10::GetCompressProfile()->add_memory_load((stats.allocated_bytes)[0].current);
                    //std::cout<<(stats.allocated_bytes)[0].current<<"-";
            }
        } 
        x4.grad_fn()->is_cat = true;
        //x4.grad_fn()->multiple_input.com_and_trans.resize(3,-1);
        x4.grad_fn()->multiple_input.input_tensors.resize(3);
        x4.grad_fn()->multiple_input.pre_nodes.resize(3,nullptr);
        //x4.grad_fn()->multiple_input.storage_impl_s.resize(3,nullptr);
        //x4.grad_fn()->multiple_input.dataptrs.resize(3);
        //int cnt_ = 0;
        for(int j=0;j<3;j++){
            if((inputs[j].tensor_id>=0)&&(policy[inputs[j].tensor_id]==1)&&(index_==begin_stage[inputs[j].tensor_id])){
                inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->data_ptr().clear();
                inputs[j].grad_fn()->storage_impl_ = inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
                inputs[j].grad_fn()->com_and_tran = 1;
                

            }else if((inputs[j].tensor_id>=0)&&(policy[inputs[j].tensor_id]==2)&&(index_==begin_stage[inputs[j].tensor_id])){
                //cnt_++;
                if(c10::Passive::is_enabled()){
                    cudaStreamSynchronize(stream[j]);
                    inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->set_data_ptr(std::move(dataptrs[j]));   
                    /**          
                    if(c10::Profile2::is_enabled()){
                        gettimeofday(&end_1, NULL);
                        timeuse = 1000000 * ( end_1.tv_sec - start_1.tv_sec ) + end_1.tv_usec - start_1.tv_usec;
                        c10::GetCompressProfile()->add_transfer_time(timeuse);
                    }**/
                }else{
                     at::GetTransferRun()->add_item(inputs[j].nbytes(),inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl());
                }
                inputs[j].grad_fn()->com_and_tran = 2;
                inputs[j].grad_fn()->storage_impl_ = inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();

            }
            x4.grad_fn()->multiple_input.input_tensors[j] = inputs[j].variable_data();
            x4.grad_fn()->multiple_input.pre_nodes[j] = inputs[j].grad_fn();
        }
        //if(cnt_>1) flaf = true;
        return x4;
    }

    BasicConv2d basic1;
    BasicConv2d basic2;
    BasicConv2d basic3;
    BasicConv2d basic4;
    BasicConv2d basic5;
    BasicConv2d basic6;
    torch::nn::MaxPool2d maxpool{nullptr};


};
TORCH_MODULE(InceptionD);


struct InceptionEImpl:torch::nn::Module{
    InceptionEImpl(){}
    InceptionEImpl(int in){
        basic1 = BasicConv2d(in,320,1);
        basic2 = BasicConv2d(in,384,1);
        basic3 = BasicConv2d(384,384,torch::ExpandingArray<2>({1,3}),torch::ExpandingArray<2>({0,1}));
        basic4 = BasicConv2d(384,384,torch::ExpandingArray<2>({3,1}),torch::ExpandingArray<2>({1,0}));
        basic5 = BasicConv2d(in,448,1);
        basic6 = BasicConv2d(448,384,3,1);
        basic7 = BasicConv2d(384,384,torch::ExpandingArray<2>({1,3}),torch::ExpandingArray<2>({0,1}));
        basic8 = BasicConv2d(384,384,torch::ExpandingArray<2>({3,1}),torch::ExpandingArray<2>({1,0}));
        basic9 = BasicConv2d(in,192,1);
        register_module("basicConv2d1",basic1);
        register_module("basicConv2d2",basic2);
        register_module("basicConv2d3",basic3);
        register_module("basicConv2d4",basic4);
        register_module("basicConv2d5",basic5);
        register_module("basicConv2d6",basic6);
        register_module("basicConv2d7",basic7);
        register_module("basicConv2d8",basic8);
        register_module("basicConv2d9",basic9);
        avg_pool = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(3).stride(1).padding(1));
    }
    torch::Tensor forward(torch::Tensor& input){
        torch::Tensor x1 = basic1->forward(input);
        torch::Tensor x2 = basic2->forward(input);

        torch::Tensor x2_1 = basic3->forward(x2);
        torch::Tensor x2_2 = basic4->forward(x2);

        index_++;
        //std::cout<<x2_1.tensor_id<<";"<<x2_2.tensor_id<<"-";
        std::vector<torch::Tensor> inputs{x2_1,x2_2};
        std::vector<c10::DataPtr> dataptrs(4);
        cudaStream_t stream[4];

        for(int j=0;j<2;j++){
            if((inputs[j].tensor_id>=0)&&(policy[inputs[j].tensor_id]==2)&&(index_==begin_stage[inputs[j].tensor_id])){
                if(c10::Passive::is_enabled()){
                    //if(c10::Profile2::is_enabled()){
                    //    gettimeofday(&start_1, NULL);
                    //}
                    sizes = inputs[j].nbytes();
                    //dataptr_ = at::getCPUAllocator()->allocate(sizes);
                    dataptrs[j] = at::detail::getCUDAHooks().getPinnedMemoryAllocator()->allocate(sizes);
                    cudaStreamCreateWithFlags(&stream[j],cudaStreamNonBlocking);
                    //cudaDeviceSynchronize();
                    //cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
                    cudaMemcpyAsync(dataptrs[j].get(),inputs[j].data_ptr(),sizes,cudaMemcpyDeviceToHost,stream[j]);
                    //cudaStreamAddCallback(stream1, c10::MyCallback, (void*)index, 0);
                }              
            }
        }
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&start_, NULL);
        }
        x2 = torch::cat({x2_1,x2_2},1);
        x2.tensor_id = index_;
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&end_, NULL);
            timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
            c10::GetPolicyMaker()->add_comptime(timeuse);
            x2.grad_fn()->id_ = index_;
            x2.grad_fn()->ids.push_back(x2_1.tensor_id);
            x2.grad_fn()->ids.push_back(x2_2.tensor_id);
            c10::GetPolicyMaker()->add_tensor(index_,x2.nbytes());
            c10::GetPolicyMaker()->add_stepdep({x2_1.tensor_id,x2_2.tensor_id,index_});
        }else{
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            if(c10::Profile3::is_enabled()){
                const  c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
                c10::GetCompressProfile()->add_memory_load((stats.allocated_bytes)[0].current);
                    //std::cout<<(stats.allocated_bytes)[0].current<<"-";
            }
        } 
        x2.grad_fn()->is_cat = true;
        //x2.grad_fn()->multiple_input.com_and_trans.resize(2,-1);
        x2.grad_fn()->multiple_input.input_tensors.resize(2);
        x2.grad_fn()->multiple_input.pre_nodes.resize(2,nullptr);
        //x2.grad_fn()->multiple_input.storage_impl_s.resize(2,nullptr);
        //x2.grad_fn()->multiple_input.dataptrs.resize(2);
        //int cnt_ = 0;
        for(int j=0;j<2;j++){
            if((inputs[j].tensor_id>=0)&&(policy[inputs[j].tensor_id]==1)&&(index_==begin_stage[inputs[j].tensor_id])){
                inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->data_ptr().clear();
                inputs[j].grad_fn()->storage_impl_ = inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
                inputs[j].grad_fn()->com_and_tran = 1;
                

            }else if((inputs[j].tensor_id>=0)&&(policy[inputs[j].tensor_id]==2)&&(index_==begin_stage[inputs[j].tensor_id])){
                //cnt_++;
                if(c10::Passive::is_enabled()){
                    cudaStreamSynchronize(stream[j]);
                    inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->set_data_ptr(std::move(dataptrs[j]));   
                    /**          
                    if(c10::Profile2::is_enabled()){
                        gettimeofday(&end_1, NULL);
                        timeuse = 1000000 * ( end_1.tv_sec - start_1.tv_sec ) + end_1.tv_usec - start_1.tv_usec;
                        c10::GetCompressProfile()->add_transfer_time(timeuse);
                    }**/
                }else{
                     at::GetTransferRun()->add_item(inputs[j].nbytes(),inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl());
                }
                inputs[j].grad_fn()->com_and_tran = 2;
                inputs[j].grad_fn()->storage_impl_ = inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();

            }
            x2.grad_fn()->multiple_input.input_tensors[j] = inputs[j].variable_data();
            x2.grad_fn()->multiple_input.pre_nodes[j] = inputs[j].grad_fn();
        }
        //if(cnt_>1) flaf = true;


        torch::Tensor x3 = basic5->forward(input);
        x3 = basic6->forward(x3);

        torch::Tensor x3_1 = basic7->forward(x3);
        torch::Tensor x3_2 = basic8->forward(x3);

        index_++;
        //std::cout<<x3_1.tensor_id<<";"<<x3_2.tensor_id<<"-";
        inputs[0] = x3_1;
        inputs[1] = x3_2;
        for(int j=0;j<2;j++){
            if((inputs[j].tensor_id>=0)&&(policy[inputs[j].tensor_id]==2)&&(index_==begin_stage[inputs[j].tensor_id])){
                if(c10::Passive::is_enabled()){
                    //if(c10::Profile2::is_enabled()){
                    //    gettimeofday(&start_1, NULL);
                    //}
                    sizes = inputs[j].nbytes();
                    //dataptr_ = at::getCPUAllocator()->allocate(sizes);
                    dataptrs[j] = at::detail::getCUDAHooks().getPinnedMemoryAllocator()->allocate(sizes);
                    cudaStreamCreateWithFlags(&stream[j],cudaStreamNonBlocking);
                    //cudaDeviceSynchronize();
                    //cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
                    cudaMemcpyAsync(dataptrs[j].get(),inputs[j].data_ptr(),sizes,cudaMemcpyDeviceToHost,stream[j]);
                    //cudaStreamAddCallback(stream1, c10::MyCallback, (void*)index, 0);
                }              
            }
        }
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&start_, NULL);
        }
        x3 = torch::cat({x3_1,x3_2},1);
        x3.tensor_id = index_;
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&end_, NULL);
            timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
            c10::GetPolicyMaker()->add_comptime(timeuse);
            x3.grad_fn()->id_ = index_;
            x3.grad_fn()->ids.push_back(x3_1.tensor_id);
            x3.grad_fn()->ids.push_back(x3_2.tensor_id); 
            c10::GetPolicyMaker()->add_tensor(index_,x3.nbytes());
            c10::GetPolicyMaker()->add_stepdep({x3_1.tensor_id,x3_2.tensor_id,index_});
        }else{
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            if(c10::Profile3::is_enabled()){
                const  c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
                c10::GetCompressProfile()->add_memory_load((stats.allocated_bytes)[0].current);
                    //std::cout<<(stats.allocated_bytes)[0].current<<"-";
            }
        } 
        x3.grad_fn()->is_cat = true;
        //x3.grad_fn()->multiple_input.com_and_trans.resize(2,-1);
        x3.grad_fn()->multiple_input.input_tensors.resize(2);
        x3.grad_fn()->multiple_input.pre_nodes.resize(2,nullptr);
        //x3.grad_fn()->multiple_input.storage_impl_s.resize(2,nullptr);
        //x3.grad_fn()->multiple_input.dataptrs.resize(2);
        //cnt_ = 0;
        for(int j=0;j<2;j++){
            if((inputs[j].tensor_id>=0)&&(policy[inputs[j].tensor_id]==1)&&(index_==begin_stage[inputs[j].tensor_id])){
                inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->data_ptr().clear();
                inputs[j].grad_fn()->storage_impl_ = inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
                inputs[j].grad_fn()->com_and_tran = 1;
                

            }else if((inputs[j].tensor_id>=0)&&(policy[inputs[j].tensor_id]==2)&&(index_==begin_stage[inputs[j].tensor_id])){
                //cnt_++;
                if(c10::Passive::is_enabled()){
                    cudaStreamSynchronize(stream[j]);
                    inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->set_data_ptr(std::move(dataptrs[j]));   
                    /**          
                    if(c10::Profile2::is_enabled()){
                        gettimeofday(&end_1, NULL);
                        timeuse = 1000000 * ( end_1.tv_sec - start_1.tv_sec ) + end_1.tv_usec - start_1.tv_usec;
                        c10::GetCompressProfile()->add_transfer_time(timeuse);
                    }**/
                }else{
                     at::GetTransferRun()->add_item(inputs[j].nbytes(),inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl());
                }
                inputs[j].grad_fn()->com_and_tran = 2;
                inputs[j].grad_fn()->storage_impl_ = inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();

            }
            x3.grad_fn()->multiple_input.input_tensors[j] = inputs[j].variable_data();
            x3.grad_fn()->multiple_input.pre_nodes[j] = inputs[j].grad_fn();
        }
        //if(cnt_>1) flaf = true;

        index_++;
        //std::cout<<input.tensor_id<<"-";
        if((input.tensor_id>=0)&&(policy[input.tensor_id]==2)&&(index_==begin_stage[input.tensor_id])){
            if(c10::Passive::is_enabled()){
                if(c10::Profile2::is_enabled()){
                    gettimeofday(&start_1, NULL);
                }
                sizes = input.nbytes();
                //dataptr_ = at::getCPUAllocator()->allocate(sizes);
                dataptr_ = at::detail::getCUDAHooks().getPinnedMemoryAllocator()->allocate(sizes);
                cudaStreamCreateWithFlags(&stream1,cudaStreamNonBlocking);
                //cudaDeviceSynchronize();
                cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
                if(c10::BreakdownProfile::is_enabled()){
                    gettimeofday(&start_1, NULL);
                }
                cudaMemcpyAsync(dataptr_.get(),input.data_ptr(),sizes,cudaMemcpyDeviceToHost,stream1);
                //cudaStreamAddCallback(stream1, c10::MyCallback, (void*)index, 0);
            }              
        }
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&start_, NULL);
        }
        if(c10::BreakdownProfile::is_enabled()){
            gettimeofday(&start_, NULL);
        }
        torch::Tensor x4 = avg_pool->forward(input);
        x4.tensor_id = index_;
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&end_, NULL);
            timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
            c10::GetPolicyMaker()->add_comptime(timeuse);
            x4.grad_fn()->id_ = index_;
            x4.grad_fn()->ids.push_back(input.tensor_id);
            c10::GetPolicyMaker()->add_tensor(index_,x4.nbytes());
            c10::GetPolicyMaker()->add_stepdep({input.tensor_id,index_});
        }else{
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            if(c10::Profile3::is_enabled()){
                const  c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
                c10::GetCompressProfile()->add_memory_load((stats.allocated_bytes)[0].current);
                    //std::cout<<(stats.allocated_bytes)[0].current<<"-";
            }
            if(c10::BreakdownProfile::is_enabled()){
                gettimeofday(&end_, NULL);
                timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
                c10::GetBreakdownProfile()->add_compute_time(timeuse);
            }
        } 
        if((input.tensor_id>=0)&&((policy[input.tensor_id]==1)||is_super)&&(index_==begin_stage[input.tensor_id])){
            input.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->data_ptr().clear();
            input.grad_fn()->storage_impl_ = input.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
            input.grad_fn()->com_and_tran = 1;
            //x4.grad_fn()->is_recomp = true;     
        }else if((input.tensor_id>=0)&&(policy[input.tensor_id]==2)&&(index_==begin_stage[input.tensor_id])){
            if(c10::Passive::is_enabled()){
                if(c10::BreakdownProfile::is_enabled()){
                    gettimeofday(&start_, NULL);
                }
                cudaStreamSynchronize(stream1);
                if(c10::BreakdownProfile::is_enabled()){
                    gettimeofday(&end_1, NULL);
                    timeuse = 1000000 * ( end_1.tv_sec - start_1.tv_sec ) + end_1.tv_usec - start_1.tv_usec;                      
                    c10::GetBreakdownProfile()->add_transfer_time(timeuse);
                    timeuse = 1000000 * ( end_1.tv_sec - start_.tv_sec ) + end_1.tv_usec - start_.tv_usec;                      
                    c10::GetBreakdownProfile()->add_sync_time(timeuse);
                }
                input.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->set_data_ptr(std::move(dataptr_));
                input.grad_fn()->storage_impl_ = input.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
                input.grad_fn()->com_and_tran = 2;      
                if(c10::Profile2::is_enabled()){
                    gettimeofday(&end_1, NULL);
                    timeuse = 1000000 * ( end_1.tv_sec - start_1.tv_sec ) + end_1.tv_usec - start_1.tv_usec;
                    c10::GetCompressProfile()->add_transfer_time(timeuse);
                }
            }else{                
                input.grad_fn()->com_and_tran = 2;
                input.grad_fn()->storage_impl_ = input.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
                at::GetTransferRun()->add_item(input.nbytes(),input.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl());
            }
        }
        x4.grad_fn()->pre_node = input.grad_fn();    
        x4.grad_fn()->func_ = std::bind(&torch::nn::AvgPool2dImpl::forward,avg_pool,std::placeholders::_1);
        x4.grad_fn()->input_tensor = input.variable_data();


        x4 = basic9->forward(x4);

        index_++;
        //std::cout<<x1.tensor_id<<";"<<x2.tensor_id<<";"<<x3.tensor_id<<";"<<x4.tensor_id<<"-";
        inputs.clear();
        inputs.emplace_back(x1);
        inputs.emplace_back(x2);
        inputs.emplace_back(x3);
        inputs.emplace_back(x4);
       
        for(int j=0;j<4;j++){
            if((inputs[j].tensor_id>=0)&&(policy[inputs[j].tensor_id]==2)&&(index_==begin_stage[inputs[j].tensor_id])){
                if(c10::Passive::is_enabled()){
                    //if(c10::Profile2::is_enabled()){
                    //    gettimeofday(&start_1, NULL);
                    //}
                    sizes = inputs[j].nbytes();
                    //dataptr_ = at::getCPUAllocator()->allocate(sizes);
                    dataptrs[j] = at::detail::getCUDAHooks().getPinnedMemoryAllocator()->allocate(sizes);
                    cudaStreamCreateWithFlags(&stream[j],cudaStreamNonBlocking);
                    //cudaDeviceSynchronize();
                    //cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
                    cudaMemcpyAsync(dataptrs[j].get(),inputs[j].data_ptr(),sizes,cudaMemcpyDeviceToHost,stream[j]);
                    //cudaStreamAddCallback(stream1, c10::MyCallback, (void*)index, 0);
                }              
            }
        }
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&start_, NULL);
        }
        torch::Tensor x5 = torch::cat({x1,x2,x3,x4},1);
        x5.tensor_id = index_;
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&end_, NULL);
            timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
            c10::GetPolicyMaker()->add_comptime(timeuse);
            x5.grad_fn()->id_ = index_;
            x5.grad_fn()->ids.push_back(x1.tensor_id);
            x5.grad_fn()->ids.push_back(x2.tensor_id);
            x5.grad_fn()->ids.push_back(x3.tensor_id);
            x5.grad_fn()->ids.push_back(x4.tensor_id);
            c10::GetPolicyMaker()->add_tensor(index_,x5.nbytes());
            c10::GetPolicyMaker()->add_stepdep({x1.tensor_id,x2.tensor_id,x3.tensor_id,x4.tensor_id,index_});
        }else{
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            if(c10::Profile3::is_enabled()){
                const  c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
                c10::GetCompressProfile()->add_memory_load((stats.allocated_bytes)[0].current);
                    //std::cout<<(stats.allocated_bytes)[0].current<<"-";
            }
        } 
        x5.grad_fn()->is_cat = true;
        //x5.grad_fn()->multiple_input.com_and_trans.resize(4,-1);
        x5.grad_fn()->multiple_input.input_tensors.resize(4);
        x5.grad_fn()->multiple_input.pre_nodes.resize(4,nullptr);
        //x5.grad_fn()->multiple_input.storage_impl_s.resize(4,nullptr);
        //x5.grad_fn()->multiple_input.dataptrs.resize(4);
        //cnt_ = 0;
        for(int j=0;j<4;j++){
            if((inputs[j].tensor_id>=0)&&(policy[inputs[j].tensor_id]==1)&&(index_==begin_stage[inputs[j].tensor_id])){
                inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->data_ptr().clear();
                inputs[j].grad_fn()->storage_impl_ = inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
                inputs[j].grad_fn()->com_and_tran = 1;
                
            }else if((inputs[j].tensor_id>=0)&&(policy[inputs[j].tensor_id]==2)&&(index_==begin_stage[inputs[j].tensor_id])){
                //cnt_++;
                if(c10::Passive::is_enabled()){
                    cudaStreamSynchronize(stream[j]);
                    inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->set_data_ptr(std::move(dataptrs[j]));   
                    /**          
                    if(c10::Profile2::is_enabled()){
                        gettimeofday(&end_1, NULL);
                        timeuse = 1000000 * ( end_1.tv_sec - start_1.tv_sec ) + end_1.tv_usec - start_1.tv_usec;
                        c10::GetCompressProfile()->add_transfer_time(timeuse);
                    }**/
                }else{
                     at::GetTransferRun()->add_item(inputs[j].nbytes(),inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl());
                }
                inputs[j].grad_fn()->com_and_tran = 2;
                inputs[j].grad_fn()->storage_impl_ = inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();

            }
            x5.grad_fn()->multiple_input.input_tensors[j] = inputs[j].variable_data();
            x5.grad_fn()->multiple_input.pre_nodes[j] = inputs[j].grad_fn();
        }
        //if(cnt_>1) flaf = true;

        return x5;
    }
    BasicConv2d basic1;
    BasicConv2d basic2;
    BasicConv2d basic3;
    BasicConv2d basic4;
    BasicConv2d basic5;
    BasicConv2d basic6;
    BasicConv2d basic7;
    BasicConv2d basic8;
    BasicConv2d basic9;
    torch::nn::AvgPool2d avg_pool{nullptr};
};
TORCH_MODULE(InceptionE);
struct InceptionAuxImpl:torch::nn::Module{
    InceptionAuxImpl(){}
    InceptionAuxImpl(int in,int num_classes){
        basic1 = BasicConv2d(in,128,1);
        basic2 = BasicConv2d(128,768,5);
        register_module("basicConv2d1",basic1);
        register_module("basicConv2d2",basic2);
        linear = torch::nn::Linear(768,num_classes);
        register_module("linear",linear);
        avg_pool = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(5).stride(3));
        adaptivepool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1,1}));
    }

    torch::Tensor forward(torch::Tensor& input){
        torch::Tensor x = avg_pool->forward(input);
        x = basic1->forward(x);
        x = basic2->forward(x);
        x = adaptivepool->forward(x);
        x = torch::flatten(x,1);
        x = linear->forward(x);
        return x;
    }

    BasicConv2d basic1;
    BasicConv2d basic2;
    torch::nn::Linear linear{nullptr};
    torch::nn::AdaptiveAvgPool2d adaptivepool{nullptr};
    torch::nn::AvgPool2d avg_pool{nullptr};
};
TORCH_MODULE(InceptionAux);

struct Inception3:torch::nn::Module{

    Inception3(int num_classes=100){
        basic1 = BasicConv2d(3,32,3,1);
        basic2 = BasicConv2d(32,32,3,1);
        basic3 = BasicConv2d(32,64,3,1);
        basic4 = BasicConv2d(64,80,1);
        basic5 = BasicConv2d(80,192,3);
        register_module("basicConv2d1",basic1);
        register_module("basicConv2d2",basic2);
        register_module("basicConv2d3",basic3);
        register_module("basicConv2d4",basic4);
        register_module("basicConv2d5",basic5);
        inceptionA1 = InceptionA(192,32);
        inceptionA2 = InceptionA(256,64);
        inceptionA3 = InceptionA(288,64);
        inceptionB = InceptionB(288);
        inceptionC1 = InceptionC(768,128);
        inceptionC2 = InceptionC(768,160);
        inceptionC3 = InceptionC(768,160);
        inceptionC4 = InceptionC(768,192);
        //inceptionAux = InceptionAux(768,num_classes);
        inceptionD = InceptionD(768);
        inceptionE1 = InceptionE(1280);
        inceptionE2 = InceptionE(2048);
        linear = torch::nn::Linear(2048,num_classes);
        register_module("inceptionA1",inceptionA1);
        register_module("inceptionA2",inceptionA2);
        register_module("inceptionA3",inceptionA3);
        register_module("inceptionB",inceptionB);
        register_module("inceptionC1",inceptionC1);
        register_module("inceptionC2",inceptionC2);
        register_module("inceptionC3",inceptionC3);
        register_module("inceptionC4",inceptionC4);
       // register_module("inceptionAux",inceptionAux);
        register_module("inceptionD",inceptionD);
        register_module("inceptionE1",inceptionE1);
        register_module("inceptionE2",inceptionE2);
        register_module("linear",linear);
        //maxpool1 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2));
        //maxpool2 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2));
        apaptivepool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1,1}));
        loss = torch::nn::CrossEntropyLoss();
    }

    torch::Tensor forward(torch::Tensor& input,torch::Tensor& target){

        at::GetTransferRun()->init();
        input.tensor_id = -1;
        torch::Tensor x = basic1->forward(input);
        
        x = basic2->forward(x);
       
        x = basic3->forward(x);
       
        x = basic4->forward(x);
      
        x = basic5->forward(x);
      
        x = inceptionA1->forward(x);
     
        x = inceptionA2->forward(x);
        x = inceptionA3->forward(x);
        x = inceptionB->forward(x);
        x = inceptionC1->forward(x);
        x = inceptionC2->forward(x);
        x = inceptionC3->forward(x);
        x = inceptionC4->forward(x);
        x = inceptionD->forward(x);
        x = inceptionE1->forward(x);
        x = inceptionE2->forward(x);
       
        pre = x;
        index_++;
        //std::cout<<pre.tensor_id<<"-";
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&start_, NULL);
        }
        x = apaptivepool->forward(pre);
        x.tensor_id = index_;
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&end_, NULL);
            timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
            c10::GetPolicyMaker()->add_comptime(timeuse);
            x.grad_fn()->id_ = index_;
            x.grad_fn()->ids.push_back(index_-1);
            //x.grad_fn()->ids.push_back(index);
            c10::GetPolicyMaker()->add_tensor(index_,x.nbytes());
            c10::GetPolicyMaker()->add_stepdep({index_-1,index_});

            
        }else{
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            if(c10::Profile3::is_enabled()){
                const  c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
                c10::GetCompressProfile()->add_memory_load((stats.allocated_bytes)[0].current);
                //std::cout<<(stats.allocated_bytes)[0].current<<"-";
            }
        }

        index_++;
        pre = x;
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&start_, NULL);
        }
        x = torch::nn::Dropout2d()->forward(pre);
        x.tensor_id = index_;
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&end_, NULL);
            timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
            c10::GetPolicyMaker()->add_comptime(timeuse);
            x.grad_fn()->id_ = index_;
            x.grad_fn()->ids.push_back(index_-1);
            //x.grad_fn()->ids.push_back(index);
            c10::GetPolicyMaker()->add_tensor(index_,x.nbytes());
            c10::GetPolicyMaker()->add_stepdep({index_-1,index_});

            
        }else{
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            if(c10::Profile3::is_enabled()){
                const  c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
                c10::GetCompressProfile()->add_memory_load((stats.allocated_bytes)[0].current);
                //std::cout<<(stats.allocated_bytes)[0].current<<"-";
            }
        }


        index_++;
        pre = x;
        //std::cout<<pre.tensor_id<<"-";
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&start_, NULL);
        }
        x = pre.view({pre.size(0),-1});
        x.tensor_id = index_;
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&end_, NULL);
            timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
            c10::GetPolicyMaker()->add_comptime(timeuse);
            x.grad_fn()->id_ = index_;
            x.grad_fn()->ids.push_back(index_-1);
            //x.grad_fn()->ids.push_back(index);
            c10::GetPolicyMaker()->add_tensor(index_,x.nbytes());
            c10::GetPolicyMaker()->add_stepdep({index_-1,index_});

            
        }else{
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            if(c10::Profile3::is_enabled()){
                const  c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
                c10::GetCompressProfile()->add_memory_load((stats.allocated_bytes)[0].current);
                //std::cout<<(stats.allocated_bytes)[0].current<<"-";
            }
        }
        index_++;
        pre = x;
        //std::cout<<pre.tensor_id<<"-";
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&start_, NULL);
        }
        x = linear->forward(pre);
        x.tensor_id = index_;
         if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&end_, NULL);
            timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
            c10::GetPolicyMaker()->add_comptime(timeuse);
            x.grad_fn()->id_ = index_;
            x.grad_fn()->ids.push_back(index_-1);
            //x.grad_fn()->ids.push_back(index);
            c10::GetPolicyMaker()->add_tensor(index_,x.nbytes());
            c10::GetPolicyMaker()->add_stepdep({index_-1,index_});

            
        }else{
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            if(c10::Profile3::is_enabled()){
                const  c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
                c10::GetCompressProfile()->add_memory_load((stats.allocated_bytes)[0].current);
                //std::cout<<(stats.allocated_bytes)[0].current<<"-";
            }
        }
        index_++;
        pre = x;
        //std::cout<<pre.tensor_id<<"-";
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&start_, NULL);
        }
        x = loss->forward(pre,target);
        x.tensor_id = index_;
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&end_, NULL);
            timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
            c10::GetPolicyMaker()->add_comptime(timeuse);
            x.grad_fn()->id_ = index_;
            x.grad_fn()->ids.push_back(index_-1);
            //x.grad_fn()->ids.push_back(index_);
            c10::GetPolicyMaker()->add_tensor(index_,x.nbytes());
            c10::GetPolicyMaker()->add_stepdep({index_-1,index_});

            
        }else{
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            if(c10::Profile3::is_enabled()){
                const  c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
                c10::GetCompressProfile()->add_memory_load((stats.allocated_bytes)[0].current);
                //std::cout<<(stats.allocated_bytes)[0].current<<"-";
            }
        }
        ++index_;
        if(c10::Profile::is_enabled()){
            c10::GetPolicyMaker()->set_num(index_);
        }
        return x;
    }
    BasicConv2d basic1;
    BasicConv2d basic2;
    BasicConv2d basic3;
    BasicConv2d basic4;
    BasicConv2d basic5;
    InceptionA inceptionA1;
    InceptionA inceptionA2;
    InceptionA inceptionA3;
    InceptionB inceptionB;
    InceptionC inceptionC1;
    InceptionC inceptionC2;
    InceptionC inceptionC3;
    InceptionC inceptionC4;
    //InceptionAux inceptionAux;
    InceptionD inceptionD;
    InceptionE inceptionE1;
    InceptionE inceptionE2;
    torch::nn::Linear linear{nullptr};
    //torch::nn::MaxPool2d maxpool1{nullptr};
    //torch::nn::MaxPool2d maxpool2{nullptr};
    torch::nn::AdaptiveAvgPool2d apaptivepool{nullptr};
    torch::nn::CrossEntropyLoss loss{nullptr};
};





int main(int argc,char* argv[]){
    //struct timeval start_, end_;
    
    int batch_size = 128;
    int zjlab = 0;
    int policy_ = 1;
    double ratio = 0.6;
    if(argc>=2){
        batch_size = atoi(argv[1]);
    }
    if(argc>=3){
        zjlab = atoi(argv[2]);
    }

    if(argc>=4){
        policy_ = atoi(argv[3]);
    }
    if(argc>=5){
        ratio = atof(argv[4])/100.0;
    }



    //c10::Passive::set_enabled(true);
    //c10::Profile3::set_enabled(true);
    CIFAR100 c;
    if(zjlab){
        c.init("/nfs/home/schen/dcgan/cifar-100-binary/train.bin");
    }else{
        c.init("/home/jsxnh/dcgan/cifar-100-binary/train.bin");
    }
    //CIFAR100 c("/home/jsxnh/dcgan/cifar-100-binary/train.bin");
    auto train_dataset = c.map(Normalize(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());
    auto data_loader = torch::data::make_data_loader(
        train_dataset, batch_size);
    

    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Training on GPU" << std::endl;
        device_type = torch::kCUDA;
    } else {
        std::cout << "Training on CPU" << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    Inception3 net;
    net.to(device);

    torch::optim::Adam admm(net.parameters(),torch::optim::AdamOptions(0.0001));

    int iter = 0;
    if(zjlab){
        c10::GetPolicyMaker()->read_lastvisit("/nfs/home/schen/dcgan/build/inceptionv3-cifar100/lastvisit.txt",begin_stage);
    }else{
        c10::GetPolicyMaker()->read_lastvisit("/home/jsxnh/dcgan/build/inceptionv3-cifar100/lastvisit.txt",begin_stage);
    }
    
    c10::Passive::set_enabled(true);
    for(int i=0;i<200;i++){
        policy[i] = 0;
    }


    std::vector<int> policy1,policy2;
        
    
    for (auto& batch : *data_loader){
        //std::cout<<batch.data.sizes()<<std::endl;
        //std::cout<<batch.target.sizes()<<std::endl;
        //std::cout<<batch.target<<std::endl;
        flaf = false;
        //srand((unsigned)time(NULL));
        //for(int j=0;j<300;j++){
        //    policy[j] = rand()%3;
        //}
        //for(int j=0;j<320;j++){
        //    std::cout<<policy[j]<<","<<begin_stage[j]<<";";
        //}
        //std::cout<<std::endl;
       
        

        index_ = -1;
        if(iter==2){
            c10::Profile::set_enabled(true);
        }
        auto data = batch.data.to(device);
        auto target = batch.target.to(device);
        
        cudaDeviceSynchronize();
        gettimeofday(&start_, NULL );
        auto loss = net.forward(data,target);
        //std::cout<<loss<<std::endl;
        //const  c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
        //std::cout<<(stats.allocated_bytes)[0].current<<std::endl;
        //if(flaf) std::cout<<"may be error"<<std::endl;
        loss.backward();
        cudaDeviceSynchronize();
        gettimeofday(&end_, NULL );
        int timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
        std::cout<<timeuse<<std::endl;
        //fprintf(f,"%d\n",timeuse);
        admm.step();
        
        if(c10::Profile3::is_enabled()){
            std::cout<<"max_mem:"<<c10::GetCompressProfile()->get_max_mem()<<std::endl;
            c10::GetCompressProfile()->to_txt2("/home/pytorch/memory_mamagement/build/inceptionv3-cifar100/memory_load.txt");
            c10::GetCompressProfile()->finish_iteration2();
        }
        if(c10::BreakdownProfile::is_enabled()){
            c10::GetBreakdownProfile()->finished();
        }


        if(iter==2){
            int time_;
            c10::Profile::set_enabled(false);
            c10::Passive::set_enabled(false);
            c10::GetPolicyMaker()->init(ratio);
            //c10::GetPolicyMaker()->print();
                //c10::GetPolicyMaker()->write_lastvisit("/nfs/home/schen/dcgan/build/inceptionv3-cifar100/lastvisit.txt");
                
                //c10::GetPolicyMaker()->print_time();
                //auto policy = c10::GetPolicyMaker()->capuchin(time_);
            
            if(policy_==1){
                policy1 = c10::GetPolicyMaker()->make_policy(time_);
                std::cout<<"iteration_time:"<<time_<<std::endl;
                for(int po:policy1){
                    std::cout<<po<<",";
                }
                if(policy1.size()==0) return 0;
                policy = policy1;
            }
            if(policy_==2){
                policy2 = c10::GetPolicyMaker()->capuchin(time_);
                std::cout<<"capuchin_time:"<<time_<<std::endl;
                for(int po:policy2){
                    std::cout<<po<<",";
                }
                if(policy2.size()==0) return 0;
                policy = policy2;
            }
                
            if(policy_==3){
                c10::Passive::set_enabled(true);
                std::cout<<"+,"<<std::endl;
                is_vdnn = true;
                for(int i=0;i<200;i++){
                    policy[i] = 0;
                }
            }
            if(policy_==4){
                std::cout<<"+,"<<std::endl;
                is_super = true;
                for(int i=0;i<200;i++){
                    policy[i] = 0;
                }
            }
            //c10::BreakdownProfile::set_enabled(true);
        }
        //iter++;
        if(iter==11) break;
    }
    

    return 0;
}
