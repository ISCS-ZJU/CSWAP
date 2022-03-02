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


Execution exe;
struct Net : torch::nn::Module {
    Net() {
        convs[0] = torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 32, { 3,3 }).padding(1).stride(1).bias(false));
        convs[1] = torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, { 1,1 }).padding(0).stride(1).bias(false));
        convs[2] = torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, { 3,3 }).padding(1).stride(2).bias(false));
        convs[3] = torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, { 1,1 }).padding(0).stride(1).bias(false));
        convs[4] = torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, { 3,3 }).padding(1).stride(1).bias(false));
        convs[5] = torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, { 1,1 }).padding(0).stride(1).bias(false));
        convs[6] = torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, { 3,3 }).padding(1).stride(2).bias(false));
        convs[7] = torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, { 1,1 }).padding(0).stride(1).bias(false));
        convs[8] = torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, { 3,3 }).padding(1).stride(1).bias(false));
        convs[9] = torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, { 1,1 }).padding(0).stride(1).bias(false));
        convs[10] = torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, { 3,3 }).padding(1).stride(2).bias(false));
        convs[11] = torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, { 1,1 }).padding(0).stride(1).bias(false));
        convs[12] = torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, { 3,3 }).padding(1).stride(1).bias(false));
        convs[13] = torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, { 1,1 }).padding(0).stride(1).bias(false));
        convs[14] = torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, { 3,3 }).padding(1).stride(1).bias(false));
        convs[15] = torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, { 1,1 }).padding(0).stride(1).bias(false));
        convs[16] = torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, { 3,3 }).padding(1).stride(1).bias(false));
        convs[17] = torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, { 1,1 }).padding(0).stride(1).bias(false));
        convs[18] = torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, { 3,3 }).padding(1).stride(1).bias(false));
        convs[19] = torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, { 1,1 }).padding(0).stride(1).bias(false));
        convs[20] = torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, { 3,3 }).padding(1).stride(1).bias(false));
        convs[21] = torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, { 1,1 }).padding(0).stride(1).bias(false));
        convs[22] = torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, { 3,3 }).padding(1).stride(2).bias(false));
        convs[23] = torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 1024, { 1,1 }).padding(0).stride(1).bias(false));
        convs[24] = torch::nn::Conv2d(torch::nn::Conv2dOptions(1024, 1024, { 3,3 }).padding(1).stride(1).bias(false));
        convs[25] = torch::nn::Conv2d(torch::nn::Conv2dOptions(1024, 1024, { 1,1 }).padding(0).stride(1).bias(false));


        batchnorm[0] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32));
        batchnorm[1] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64));
        batchnorm[2] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64));
        batchnorm[3] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(128));
        batchnorm[4] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(128));
        batchnorm[5] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(128));
        batchnorm[6] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(128));
        batchnorm[7] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(256));
        batchnorm[8] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(256));
        batchnorm[9] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(256));
        batchnorm[10] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(256));
        batchnorm[11] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512));
        batchnorm[12] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512));
        batchnorm[13] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512));
        batchnorm[14] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512));
        batchnorm[15] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512));
        batchnorm[16] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512));
        batchnorm[17] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512));
        batchnorm[18] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512));
        batchnorm[19] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512));
        batchnorm[20] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512));
        batchnorm[21] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512));
        batchnorm[22] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512));
        batchnorm[23] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(1024));
        batchnorm[24] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(1024));
        batchnorm[25] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(1024));
        
        conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(3,32,{3,3}).stride(1).padding(1).bias(false));
        bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32));
        avg_pool = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(7));
        //hpy
        //fc1 = torch::nn::Linear(50176,200);
        fc1 = torch::nn::Linear(4096,200);
        loss = torch::nn::CrossEntropyLoss();


        for(int i=0;i<26;i++){
            register_module("batchnorm_"+std::to_string(i),batchnorm[i]);
            register_module("conv_"+std::to_string(i),convs[i]);
        }

        register_module("conv", conv);
        register_module("bn", bn);
        register_module("fc1", fc1);
        register_module("crossloss", loss);


        v.push_back(std::bind(&torch::nn::Conv2dImpl::forward,conv,std::placeholders::_1));
        v.push_back(std::bind(&torch::nn::BatchNorm2dImpl::forward,bn,std::placeholders::_1));
        v.push_back(torch::relu);
        for(int i=0;i<26;i++){        
            v.push_back(std::bind(&torch::nn::Conv2dImpl::forward,convs[i],std::placeholders::_1));
            v.push_back(std::bind(&torch::nn::BatchNorm2dImpl::forward,batchnorm[i],std::placeholders::_1));
            v.push_back(torch::relu);
        }
        v.push_back(std::bind(&torch::nn::AvgPool2dImpl::forward,avg_pool,std::placeholders::_1));
        policy.resize(82,0);

       

        //policy[1] = 1;
        //policy[3] = 1,policy[7] = 1,policy[10] = 1,policy[14] = 1,policy[17] = 1,policy[20] = 1
        //,policy[24] = 1,policy[27] = 1,policy[30] = 1,policy[34] = 1,policy[37] = 1,policy[40] = 1;

        //policy[4] = 1;policy[8] = 1;policy[11]=1;policy[15]=1;policy[18]=1;policy[21]=1;
        //policy[25]=1;policy[28]=1;policy[31]=1;policy[35]=1;policy[38]=1;policy[41]=1;

    }
    

    void set_policy(int index,int po){
        policy[index] = po;
    }
    void clear_policy(){
        policy.clear();
        policy.resize(45);
    }
    // Implement the Net's algorithm.
    torch::Tensor forward(torch::Tensor x,torch::Tensor target) {
        torch::Tensor pre=x;
        exe.preforward();

        for(int index =0;index<v.size();index++){
            exe.pre(pre,ACT);
            x = v[index](pre);
            exe.post(pre,x,ACT);
            x.grad_fn()->func_ = v[index];
            pre = x;   
        }
        //gettimeofday(&start_, NULL);

        exe.pre(pre,ACT);
        x = pre.view({ pre.size(0), -1 });
        exe.post(pre,x,ACT,true);
        pre = x;
        
        exe.pre(pre,ACT);
        x = fc1->forward(pre);
        exe.post(pre,x,ACT,true);
        pre = x;
        exe.pre({pre,target},ACT);
        x = loss->forward(pre,target);
        exe.post({pre,target},x,ACT,true);
        exe.postforward();
        
        return x;
    }
    
    //torch::nn::Conv2dImpl::forward;
    // Use one of many "standard library" modules.

    torch::nn::Conv2d convs[26]{nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
    nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
    nullptr,nullptr,nullptr,nullptr,nullptr};

    torch::nn::BatchNorm2d batchnorm[26]{nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
    nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
    nullptr,nullptr,nullptr,nullptr,nullptr};

    torch::nn::Conv2d conv{nullptr};
    torch::nn::BatchNorm2d bn{nullptr};
    torch::nn::Linear fc1{ nullptr };
    torch::nn::CrossEntropyLoss loss{nullptr};
    torch::nn::AvgPool2d avg_pool{nullptr};
    //torch::nn::Linear fc2{ nullptr };
    //torch::nn::Linear fc3{ nullptr };

    std::vector<std::function<torch::Tensor(const torch::Tensor&)>> v;
    std::vector<int> policy;//操作的输入决策
};


int main(int argc,char* argv[]){
    
    int batch_size = 32;
    bool is_compress = false;

    if(argc>=2){
        batch_size = atoi(argv[1]);
    }
    if (argc>=3){
        is_compress = atoi(argv[2]);
    }

    //hpy
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

    torch::optim::Adam admm(net.parameters(),torch::optim::AdamOptions(0.003));

    std::vector<int> policy1,policy2;
    exe.init_(100);
    for(int i=0;i<100;i++){
        exe.begin_stage[i] = i+1;
    }
    for(int j=0;j<25;j++){
        exe.policy[3*j+2] = 6;
    }

    int iter = 0;
    for(int i=0;i<2;i++){
        std::cout<<i<<std::endl;
        iter = 0;
        
        for (auto& batch : *data_loader){
            /**
            if(iter==3){
                c10::Profile::set_enabled(true);
            }**/
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
            //iter++;
            //if(iter==3)
            //    break;
            admm.step();
            
            if(c10::Profile2::is_enabled()||c10::Profile4::is_enabled()){
                c10::GetCompressProfile()->finish_iteration();
                if(is_compress){
                    c10::GetCompressProfile()->add_total_time2(timeuse);
                }else{
                    c10::GetCompressProfile()->add_total_time1(timeuse);
                }
            }
            iter++;
            if(c10::ProfilerSwitch::is_enabled()){
                c10::GetProfilerImpl()->add_total_time(timeuse);
                c10::GetProfilerImpl()->finish_iteration();
            }
        }
       
    }

    if(c10::ProfilerSwitch::is_enabled()){
        c10::GetProfilerImpl()->to_txt("result/mobilenet_imagenet_128.txt");
    }


    return 0;
}
