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
#include <cuda_runtime.h>
#include<c10/core/PolicyMaker.h>
//#include<c10/core/TransferRun.h>
#include<c10/core/CPUAllocator.h>
#include<ATen/core/TransferRun.h>
#include<thread>
#include <ATen/core/grad_mode.h>
#include "Execution.h"
using namespace std;


Execution exe;
struct Net : torch::nn::Module {
    Net() {
        convs[0] = torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, { 3,3 }).stride(1).padding(1));
        convs[1] = torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, { 3,3 }).stride(1).padding(1));
        convs[2] = torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, { 3,3 }).stride(1).padding(1));
        convs[3] = torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, { 3,3 }).stride(1).padding(1));
        convs[4] = torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, { 3,3 }).stride(1).padding(1));
        convs[5] = torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, { 3,3 }).stride(2).padding(1));
        convs[6] = torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, { 3,3 }).stride(1).padding(1));
        convs[7] = torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, { 3,3 }).stride(2).padding(1));
        convs[8] = torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, { 3,3 }).stride(1).padding(1));
        convs[9] = torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, { 3,3 }).stride(1).padding(1));
        convs[10] = torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, { 3,3 }).stride(2).padding(1));
        convs[11] = torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, { 3,3 }).stride(1).padding(1));
        convs[12] = torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, { 3,3 }).stride(2).padding(1));
        convs[13] = torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, { 3,3 }).stride(1).padding(1));
        convs[14] = torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, { 3,3 }).stride(1).padding(1));
        convs[15] = torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, { 3,3 }).stride(2).padding(1));
        convs[16] = torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, { 3,3 }).stride(1).padding(1));
        convs[17] = torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, { 3,3 }).stride(2).padding(1));
        convs[18] = torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, { 3,3 }).stride(1).padding(1));
        convs[19] = torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, { 3,3 }).stride(1).padding(1));

        batchnorm[0] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64));
        batchnorm[1] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64));
        batchnorm[2] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64));
        batchnorm[3] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64));
        batchnorm[4] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64));
        batchnorm[5] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(128));
        batchnorm[6] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(128));
        batchnorm[7] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(128));
        batchnorm[8] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(128));
        batchnorm[9] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(128));
        batchnorm[10] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(256));
        batchnorm[11] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(256));
        batchnorm[12] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(256));
        batchnorm[13] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(256));
        batchnorm[14] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(256));
        batchnorm[15] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512));
        batchnorm[16] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512));
        batchnorm[17] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512));
        batchnorm[18] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512));
        batchnorm[19] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512));

       
        avg_pool = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(4));
        //hpy
        fc1 = torch::nn::Linear(25088,200); 
        
        loss = torch::nn::CrossEntropyLoss();
        for(int i=0;i<20;i++){
            register_module("conv"+std::to_string(i), convs[i]);
            register_module("batch"+std::to_string(i),batchnorm[i]);
        }
        
        register_module("fc1", fc1);
       
        register_module("crossloss", loss);

        v.push_back(std::bind(&torch::nn::Conv2dImpl::forward,convs[0],std::placeholders::_1));
        v.push_back(std::bind(&torch::nn::BatchNorm2dImpl::forward,batchnorm[0],std::placeholders::_1));
        v.push_back(torch::relu);
        int index = 1;
        for(int i=0;i<8;i++){
            if((i==0)||(i==1)||(i==3)||(i==5)||(i==7)){
                v.push_back(std::bind(&torch::nn::Conv2dImpl::forward,convs[index],std::placeholders::_1));
                v.push_back(std::bind(&torch::nn::BatchNorm2dImpl::forward,batchnorm[index++],std::placeholders::_1));
                v.push_back(torch::relu);
                v.push_back(std::bind(&torch::nn::Conv2dImpl::forward,convs[index],std::placeholders::_1));
                v.push_back(std::bind(&torch::nn::BatchNorm2dImpl::forward,batchnorm[index++],std::placeholders::_1));
                v.push_back(torch::relu);
                v.push_back(torch::relu);
            }else{
                v.push_back(std::bind(&torch::nn::Conv2dImpl::forward,convs[index],std::placeholders::_1));
                v.push_back(std::bind(&torch::nn::BatchNorm2dImpl::forward,batchnorm[index++],std::placeholders::_1));
                v.push_back(torch::relu);
                v.push_back(std::bind(&torch::nn::Conv2dImpl::forward,convs[index],std::placeholders::_1));
                v.push_back(std::bind(&torch::nn::BatchNorm2dImpl::forward,batchnorm[index++],std::placeholders::_1));
                v.push_back(std::bind(&torch::nn::Conv2dImpl::forward,convs[index],std::placeholders::_1));
                v.push_back(std::bind(&torch::nn::BatchNorm2dImpl::forward,batchnorm[index++],std::placeholders::_1));
                v.push_back(torch::relu);
                v.push_back(torch::relu);
            }
        }
        v.push_back(std::bind(&torch::nn::AvgPool2dImpl::forward,avg_pool,std::placeholders::_1));
        v.push_back(torch::relu);
        v.push_back(std::bind(&torch::nn::LinearImpl::forward,fc1,std::placeholders::_1));
        policy.resize(69,0);
        // v.size() 68
    }
   
    // Implement the Net's algorithm.
    torch::Tensor forward(torch::Tensor x,torch::Tensor target) {
         vector<torch::Tensor> tensors;
        torch::Tensor pre = x;
        exe.preforward();
        for(int index =0;index<v.size();index++){
            /**
            if(index&&types[index]==CONV){
                if(types[index-1]==ACT){
                    std::cout<<index<<"*";
                }else{
                    std::cout<<index<<"-";
                }
            }**/
            if((index==8)||(index==15)||(index==31)||(index==47)||(index==63)){
                exe.pre({tensors[index-6],pre},ACT);
                x = torch::add(tensors[index-6],pre);
                exe.post({tensors[index-6],pre},x,ACT);
                x.grad_fn()->is_cat = true;
                x.grad_fn()->operation = 2;
            }else if((index==22)||(index==38)||(index==54)){
                pre = tensors[index-6];
                exe.pre(pre,ACT);
                x = v[index](pre);
                exe.post(pre,x,ACT);
                x.grad_fn()->func_ = v[index];  
            }else if((index==24)||(index==40)||(index==56)){
                exe.pre({tensors[index-3],pre},ACT);
                x = torch::add(tensors[index-3],pre);
                exe.post({tensors[index-3],pre},x,ACT);
                x.grad_fn()->is_cat = true;
                x.grad_fn()->operation = 2;
                
            }else if(index==66){
                exe.pre(pre,ACT);
                x = pre.view({ pre.size(0), -1 });
                exe.post(pre,x,ACT,true);
                
            }else{
                exe.pre(pre,ACT);
                x = v[index](pre);
                exe.post(pre,x,ACT);
                x.grad_fn()->func_ = v[index];    
            }
            
            tensors.push_back(x);
            pre = x;
            
        }
        exe.pre({pre,target},ACT);
        x = loss->forward(pre,target);
        exe.post({pre,target},x,ACT,true);
        exe.postforward();
        return x;
    }
   
  
    torch::nn::Linear fc1{ nullptr };
    //torch::nn::Linear fc2{ nullptr };
    //torch::nn::Linear fc3{ nullptr };
    torch::nn::CrossEntropyLoss loss{nullptr};
    torch::nn::AvgPool2d avg_pool{nullptr};
    //torch::nn::AvgPool2d avg_pool{nullptr};
    //torch::nn::Linear fc2{ nullptr };
    //torch::nn::Linear fc3{ nullptr };

    torch::nn::Conv2d convs[20]{nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
    nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr};
    torch::nn::BatchNorm2d batchnorm[20]{nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
    nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr};



    std::vector<std::function<torch::Tensor(const torch::Tensor&)>> v;
    std::vector<int> policy;
};


int main(int argc,char* argv[]){
    //hpy batch_size
    int batch_size = 16;
    bool is_compress = false;

    if(argc>=2){
        batch_size = atoi(argv[1]);
    }
    if (argc>=3){
        is_compress = atoi(argv[2]);
    }

    std::cout << "batch_size = " << batch_size << std::endl;

    
    //hpy: 加载数据集
    // CIFAR10 c("/nfs/home/schen/dcgan/cifar10/data_batch_2.bin");
    TinyImagenet c("/home/jsxnh/dcgan/tinyimagenet/train_sqlit_data_0.bin","/home/jsxnh/dcgan/tinyimagenet/train_sqlit_label_0.bin");

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

    Net net;
    net.to(device);

    torch::optim::Adam admm(net.parameters(),torch::optim::AdamOptions(0.002));
    //hpy: 不知道是什么路径
    exe.init_(100);
    c10::GetPolicyMaker()->read_lastvisit("/home/jsxnh/dcgan/build/resnet-cifar10/lastvisit.txt",exe.begin_stage);



    //net.policy[2] = 3,net.policy[5] = 3,net.policy[12] = 3,net.policy[28] = 3,net.policy[44] = 3,
    //net.policy[60] = 3,net.policy[9] = 3,net.policy[16] = 3,net.policy[32] = 3,net.policy[48] = 3,
    //net.policy[62] = 3,net.policy[19] = 3;
    //net.policy[35] = 2,net.policy[51] = 2,net.policy[25] = 2,net.policy[41] = 2;

    exe.policy[2] = 6,exe.policy[5] = 6,exe.policy[9] = 6,exe.policy[12] = 6;
    exe.policy[16] = 6,exe.policy[19] = 6,exe.policy[25] = 6,exe.policy[28] = 6;
    exe.policy[32] = 6,exe.policy[35] = 6,exe.policy[41] = 6,exe.policy[44] = 6;
    exe.policy[48] = 6,exe.policy[51] = 6,exe.policy[57] = 6,exe.policy[60] = 6;
    exe.policy[21] = 6,exe.policy[37] = 6,exe.policy[53] = 6;//conv前不是relu

    int iter = 0;
    for(int i=0;i<1;i++){
        std::cout<<i<<std::endl;
        
        for (auto& batch : *data_loader){
            //std::cout<<batch.data.sizes()<<std::endl;
            //std::cout<<batch.target.sizes()<<std::endl;

            auto data = batch.data.to(device);
            auto target = batch.target.to(device);
            target = target - 1; ///////////////////1-200 --> 0-199
            struct timeval start_, end_;
            cudaDeviceSynchronize();
            gettimeofday(&start_, NULL );
            auto loss = net.forward(data,target);
            //std::cout<<loss<<std::endl;
            //const  c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
            //std::cout<<(stats.allocated_bytes)[0].current<<std::endl;
            loss.backward();
            cudaDeviceSynchronize();
            gettimeofday(&end_, NULL );
            int timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
            std::cout<<"time use: "<<timeuse<<"  loss: "<<loss.item<float>()<<std::endl;
            if(c10::Profile2::is_enabled()||c10::Profile4::is_enabled()){
                c10::GetCompressProfile()->finish_iteration();
                if(is_compress){
                    c10::GetCompressProfile()->add_total_time2(timeuse);
                }else{
                    c10::GetCompressProfile()->add_total_time1(timeuse);
                }
            }
            //fprintf(f,"%d\n",timeuse);
            admm.step();
            if(c10::ProfilerSwitch::is_enabled()){
                c10::GetProfilerImpl()->add_total_time(timeuse);
                c10::GetProfilerImpl()->finish_iteration();
            }
            //c10::GetCompressProfile()->to_txt4("/home/jsxnh/dcgan/build/result/imagenet/resnet-tinyimagenet-cswap.txt-iteration");
    
        }
    }

    //hpy: 保存路径 
     if(c10::ProfilerSwitch::is_enabled()){
        c10::GetProfilerImpl()->to_txt("result/resnet_imagenet_128.txt");
    }
    return 0;
}
