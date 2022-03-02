#include "tinyimagenet.h"
#include<iostream>
#include<fstream>
#include <torch/torch.h>
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
#include<torch/csrc/autograd/generated/Functions.h>
#include<c10/core/PolicyMaker.h>
#include<ATen/core/TransferRun.h>
#include <ATen/core/grad_mode.h>
#include "Execution.h"



Execution exe;
struct Net : torch::nn::Module {
    Net() {
        conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, { 11,11 }).padding(2).stride(4));
        conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 192, { 5,5 }).padding(2));
        conv3 = torch::nn::Conv2d(torch::nn::Conv2dOptions(192, 384, { 3,3 }).padding(1));
        conv4 = torch::nn::Conv2d(torch::nn::Conv2dOptions(384, 256, { 3,3 }).padding(1));
        conv5 = torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, { 3,3 }).padding(1));
        

        maxpool1 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({3,3}).stride(2));
        maxpool2 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({3,3}).stride(2));
        maxpool3 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({3,3}).stride(2));
        avg_pool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({6,6}));
        fc1 = torch::nn::Linear(256*6*6,4096);
        fc2 = torch::nn::Linear(4096,4096);
        fc3 = torch::nn::Linear(4096,200);
        loss = torch::nn::CrossEntropyLoss();


        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv3", conv3);
        register_module("conv4", conv4);
        register_module("conv5", conv5);
        
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);
        register_module("crossloss", loss);

        std::function<torch::Tensor(const torch::Tensor&)> functions[22] = {
        std::bind(&torch::nn::Conv2dImpl::forward,conv1,std::placeholders::_1),
        torch::relu,
        std::bind(&torch::nn::MaxPool2dImpl::forward,maxpool1,std::placeholders::_1),
        std::bind(&torch::nn::Conv2dImpl::forward,conv2,std::placeholders::_1),
        torch::relu,
        std::bind(&torch::nn::MaxPool2dImpl::forward,maxpool2,std::placeholders::_1),
        std::bind(&torch::nn::Conv2dImpl::forward,conv3,std::placeholders::_1),
        torch::relu,
        std::bind(&torch::nn::Conv2dImpl::forward,conv4,std::placeholders::_1),
        torch::relu,
        std::bind(&torch::nn::Conv2dImpl::forward,conv5,std::placeholders::_1),
        torch::relu,
        std::bind(&torch::nn::MaxPool2dImpl::forward,maxpool3,std::placeholders::_1),
        std::bind(&torch::nn::AdaptiveAvgPool2dImpl::forward,avg_pool,std::placeholders::_1),
        torch::relu,
        //std::bind(torch::flatten,std::placeholders::_1,(int64_t)1,(int64_t)-1),
        //torch::flatten()
        //torch::dropout()
        std::bind(torch::dropout,std::placeholders::_1,0.2,true),
        std::bind(&torch::nn::LinearImpl::forward,fc1,std::placeholders::_1),
        torch::relu,
        std::bind(torch::dropout,std::placeholders::_1,0.2,true),
        std::bind(&torch::nn::LinearImpl::forward,fc2,std::placeholders::_1),
        torch::relu,
        std::bind(&torch::nn::LinearImpl::forward,fc3,std::placeholders::_1),
        };

        v = std::vector<std::function<torch::Tensor(const torch::Tensor&)>>(functions,functions+22);
        policy.resize(22,0);
    }
 
    // Implement the Net's algorithm.
    torch::Tensor forward(torch::Tensor x,torch::Tensor target) {
        exe.preforward();
        torch::Tensor pre = x;
        for(int index=0;index<v.size();index++){
            exe.pre(pre,ACT);
            if(index==15){
                x = torch::flatten(pre,1);
            }else{
                x = v[index](pre);
            }    
            exe.post(pre,x,ACT);
            //x.grad_fn()->func_ = v[index];
            pre = x;   

        }
        exe.pre({pre,target},ACT);
        x = loss->forward(pre,target);
        exe.post({pre,target},x,ACT,true);
        exe.postforward();
        return x;
    }
    
    //torch::nn::Conv2dImpl::forward;
    // Use one of many "standard library" modules.
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::Conv2d conv3{nullptr};
    torch::nn::Conv2d conv4{nullptr};
    torch::nn::Conv2d conv5{nullptr};
    torch::nn::AdaptiveAvgPool2d avg_pool{nullptr};

    torch::nn::MaxPool2d maxpool1{nullptr};
    torch::nn::MaxPool2d maxpool2{nullptr};
    torch::nn::MaxPool2d maxpool3{nullptr};

    torch::nn::Linear fc1{ nullptr };
    torch::nn::Linear fc2{ nullptr };
    torch::nn::Linear fc3{ nullptr };
    torch::nn::CrossEntropyLoss loss{nullptr};

    std::vector<std::function<torch::Tensor(const torch::Tensor&)>> v;
    std::vector<int> policy;
};


int main(int argc,char* argv[]){
    
    TinyImagenet c("/home/jsxnh/dcgan/tinyimagenet/train_sqlit_data_0.bin","/home/jsxnh/dcgan/tinyimagenet/train_sqlit_label_0.bin");
    
    int batch_size = 256;
    bool is_compress = false;
    if(argc>=2){
        batch_size = atoi(argv[1]);
    }
    if (argc>=3){
         is_compress = atoi(argv[2]);
    }
    auto train_dataset = c.map(Normalize(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());
    auto data_loader = torch::data::make_data_loader(
        train_dataset, batch_size);
    c10::Passive::set_enabled(true);

    
    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Training on GPU" << std::endl;
        device_type = torch::kCUDA;
    } else {
        std::cout << "Training on CPU" << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);
    
    Net net;
    net.to(device);
    int iter = 0;
    
    
    exe.init_(30);
    for(int i=0;i<30;i++){
        exe.begin_stage[i] = i+1;
    }
    exe.policy[2] = 6,exe.policy[5] = 6,exe.policy[8] = 6,exe.policy[11] = 6;

    torch::optim::Adam admm(net.parameters(),torch::optim::AdamOptions(0.003));
    // torch::optim::SGD SGD(net.parameters(), torch::optim::SGDOptions(1e-5).momentum(0.9));
    for(int i=0;i<2;i++){
        std::cout<<i<<std::endl;
        iter = 0;
        
        for (auto& batch : *data_loader){
           //std::cout<<batch.data.sizes()<<std::endl;
            //std::cout<<batch.target.sizes()<<std::endl;
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);
            target = target - 1;
             struct timeval start_, end_;
            cudaDeviceSynchronize();
            gettimeofday(&start_, NULL );
            auto loss = net.forward(data,target);
            
            //const  c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
            //std::cout<<(stats.allocated_bytes)[0].current<<std::endl;
            loss.backward();
            cudaDeviceSynchronize();
            gettimeofday(&end_, NULL );
            int timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
            std::cout<<timeuse<< " loss: " << loss.item<float>() << std::endl;
            if(c10::ProfilerSwitch::is_enabled()){
                c10::GetProfilerImpl()->add_total_time(timeuse);
                c10::GetProfilerImpl()->finish_iteration();
            }
            iter++;
            
            admm.step();
            
            
        }
    }
    if(c10::ProfilerSwitch::is_enabled()){
        c10::GetProfilerImpl()->to_txt("result/alexnet_imagenet_128.txt");
    }
    return 0;

}
