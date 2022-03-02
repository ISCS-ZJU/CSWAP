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
        conv1_1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, { 3,3 }).padding(1));
        conv1_2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, { 3,3 }).padding(1));
        conv2_1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, { 3,3 }).padding(1));
        conv2_2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, { 3,3 }).padding(1));
        conv3_1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, { 3,3 }).padding(1));
        conv3_2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, { 3,3 }).padding(1));
        conv3_3 = torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, { 3,3 }).padding(1));
        conv4_1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, { 3,3 }).padding(1));
        conv4_2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, { 3,3 }).padding(1));
        conv4_3 = torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, { 3,3 }).padding(1));
        conv5_1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, { 3,3 }).padding(1));
        conv5_2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, { 3,3 }).padding(1));
        conv5_3 = torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, { 3,3 }).padding(1));

        maxpool1 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2,2}));
        maxpool2 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2,2}));
        maxpool3 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2,2}));
        maxpool4 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2,2}));
        maxpool5 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2,2}));

        //avg_pool = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(1));
        //hpy
        fc1 = torch::nn::Linear(25088,4096);
        fc2 = torch::nn::Linear(4096,4096);
        fc3 = torch::nn::Linear(4096,200);
        loss = torch::nn::CrossEntropyLoss();


        batchnorm[0] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64));
        batchnorm[1] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64));
        batchnorm[2] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(128));
        batchnorm[3] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(128));
        batchnorm[4] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(256));
        batchnorm[5] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(256));
        batchnorm[6] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(256));
        batchnorm[7] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512));
        batchnorm[8] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512));
        batchnorm[9] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512));
        batchnorm[10] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512));
        batchnorm[11] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512));
        batchnorm[12] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512));
        for(int i=0;i<13;i++){
            register_module("batchnorm_"+std::to_string(i),batchnorm[i]);
        }

        register_module("conv1_1", conv1_1);
        register_module("conv1_2", conv1_2);
        register_module("conv2_1", conv2_1);
        register_module("conv2_2", conv2_2);
        register_module("conv3_1", conv3_1);
        register_module("conv3_2", conv3_2);
        register_module("conv3_3", conv3_3);
        register_module("conv4_1", conv4_1);
        register_module("conv4_2", conv4_2);
        register_module("conv4_3", conv4_3);
        register_module("conv5_1", conv5_1);
        register_module("conv5_2", conv5_2);
        register_module("conv5_3", conv5_3);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);
        register_module("crossloss", loss);

        std::function<torch::Tensor(const torch::Tensor&)> functions[44] = {
        std::bind(&torch::nn::Conv2dImpl::forward,conv1_1,std::placeholders::_1),
        std::bind(&torch::nn::BatchNorm2dImpl::forward,batchnorm[0],std::placeholders::_1),
        torch::relu,
        std::bind(&torch::nn::Conv2dImpl::forward,conv1_2,std::placeholders::_1),
        std::bind(&torch::nn::BatchNorm2dImpl::forward,batchnorm[1],std::placeholders::_1),
        torch::relu,
        std::bind(&torch::nn::MaxPool2dImpl::forward,maxpool1,std::placeholders::_1),
        std::bind(&torch::nn::Conv2dImpl::forward,conv2_1,std::placeholders::_1),
        std::bind(&torch::nn::BatchNorm2dImpl::forward,batchnorm[2],std::placeholders::_1),
        torch::relu,std::bind(&torch::nn::Conv2dImpl::forward,conv2_2,std::placeholders::_1),
        std::bind(&torch::nn::BatchNorm2dImpl::forward,batchnorm[3],std::placeholders::_1),
        torch::relu,
        std::bind(&torch::nn::MaxPool2dImpl::forward,maxpool2,std::placeholders::_1),
        std::bind(&torch::nn::Conv2dImpl::forward,conv3_1,std::placeholders::_1),
        std::bind(&torch::nn::BatchNorm2dImpl::forward,batchnorm[4],std::placeholders::_1),
        torch::relu,std::bind(&torch::nn::Conv2dImpl::forward,conv3_2,std::placeholders::_1),
        std::bind(&torch::nn::BatchNorm2dImpl::forward,batchnorm[5],std::placeholders::_1),
        torch::relu, 
        std::bind(&torch::nn::Conv2dImpl::forward,conv3_3,std::placeholders::_1),
        std::bind(&torch::nn::BatchNorm2dImpl::forward,batchnorm[6],std::placeholders::_1),
        torch::relu,
        std::bind(&torch::nn::MaxPool2dImpl::forward,maxpool3,std::placeholders::_1),
        std::bind(&torch::nn::Conv2dImpl::forward,conv4_1,std::placeholders::_1),
        std::bind(&torch::nn::BatchNorm2dImpl::forward,batchnorm[7],std::placeholders::_1),
        torch::relu,std::bind(&torch::nn::Conv2dImpl::forward,conv4_2,std::placeholders::_1),
        std::bind(&torch::nn::BatchNorm2dImpl::forward,batchnorm[8],std::placeholders::_1),
        torch::relu,
        std::bind(&torch::nn::Conv2dImpl::forward,conv4_3,std::placeholders::_1),
        std::bind(&torch::nn::BatchNorm2dImpl::forward,batchnorm[9],std::placeholders::_1),
        torch::relu,
        std::bind(&torch::nn::MaxPool2dImpl::forward,maxpool4,std::placeholders::_1),
        std::bind(&torch::nn::Conv2dImpl::forward,conv5_1,std::placeholders::_1),
        std::bind(&torch::nn::BatchNorm2dImpl::forward,batchnorm[10],std::placeholders::_1),
        torch::relu,std::bind(&torch::nn::Conv2dImpl::forward,conv5_2,std::placeholders::_1),
        std::bind(&torch::nn::BatchNorm2dImpl::forward,batchnorm[11],std::placeholders::_1),torch::relu,
        std::bind(&torch::nn::Conv2dImpl::forward,conv5_3,std::placeholders::_1),
        std::bind(&torch::nn::BatchNorm2dImpl::forward,batchnorm[12],std::placeholders::_1),torch::relu,
        std::bind(&torch::nn::MaxPool2dImpl::forward,maxpool5,std::placeholders::_1)};
        //std::bind(&torch::nn::AvgPool2dImpl::forward,avg_pool,std::placeholders::_1)};

        v = std::vector<std::function<torch::Tensor(const torch::Tensor&)>>(functions,functions+44);
        policy.resize(44,0);
        /**
        policy[0] = 1;
        policy[1] = 1;
        //policy[2] = 1;
        policy[3] = 1;
        policy[4] = 1;
        policy[7] = 1;
        policy[9] = 1;
        policy[11] = 1;
        **/
        
        //for(int i=0;i<2;i++)
        //    policy[i*2]=0;

        //policy[1] = 1;
        //policy[3] = 2,policy[7] = 2,policy[10] = 2,policy[14] = 2,policy[17] = 2,policy[20] = 2
        //,policy[24] = 2,policy[27] = 2,policy[30] = 2,policy[34] = 2,policy[37] = 2,policy[40] = 2;

        //policy[2] = 2,policy[6] = 2,policy[9] = 2,policy[13] = 2,policy[16] = 2,policy[19] = 2
        //,policy[23] = 2,policy[26] = 2,policy[29] = 2,policy[33] = 2,policy[36] = 2,policy[39] = 2;

        //policy[2] = 3,policy[6] = 3,policy[9] = 3,policy[13] = 3,policy[16] = 3,policy[19] = 3
        //,policy[23] = 3,policy[26] = 3,policy[29] = 3,policy[33] = 3,policy[36] = 3,policy[39] = 3;

        

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
            //f_size +=x.nbytes();
        }
      
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
    torch::nn::Conv2d conv1_1{ nullptr };
    torch::nn::Conv2d conv1_2{ nullptr };
    torch::nn::Conv2d conv2_1{ nullptr };
    torch::nn::Conv2d conv2_2{ nullptr };
    torch::nn::Conv2d conv3_1{ nullptr };
    torch::nn::Conv2d conv3_2{ nullptr };
    torch::nn::Conv2d conv3_3{ nullptr };
    torch::nn::Conv2d conv4_1{ nullptr };
    torch::nn::Conv2d conv4_2{ nullptr };
    torch::nn::Conv2d conv4_3{ nullptr };
    torch::nn::Conv2d conv5_1{ nullptr };
    torch::nn::Conv2d conv5_2{ nullptr };
    torch::nn::Conv2d conv5_3{ nullptr };
    torch::nn::BatchNorm2d batchnorm[13]{nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
    nullptr,nullptr,nullptr,nullptr,nullptr};

    torch::nn::MaxPool2d maxpool1{nullptr};
    torch::nn::MaxPool2d maxpool2{nullptr};
    torch::nn::MaxPool2d maxpool3{nullptr};
    torch::nn::MaxPool2d maxpool4{nullptr};
    torch::nn::MaxPool2d maxpool5{nullptr};

    torch::nn::Linear fc1{ nullptr };
    torch::nn::Linear fc2{ nullptr };
    torch::nn::Linear fc3{ nullptr };
    torch::nn::CrossEntropyLoss loss{nullptr};
    //torch::nn::AvgPool2d avg_pool{nullptr};

    std::vector<std::function<torch::Tensor(const torch::Tensor&)>> v;
    std::vector<int> policy;//操作的输入决策
};


int main(int argc,char* argv[]){


    
    bool is_compress = false;
    int batch_size = 32;

    if(argc>=2){
        batch_size = atoi(argv[1]);
    }
    if (argc>=3){
        is_compress = atoi(argv[2]);
    }
    std::cout<<"batch_size = "<<batch_size<<std::endl;

    //hpy
    //CIFAR10 c("/nfs/home/schen/dcgan/cifar10/data_batch_2.bin");
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


    // torch::optim::Adam admm(net.parameters(),torch::optim::AdamOptions(0.001));
    torch::optim::SGD SGD(net.parameters(), torch::optim::SGDOptions(0.001).momentum(0.001));
    std::vector<int> policy1,policy2;


    int iter = 0;
    
    exe.init_(100);

    for(int i=0;i<50;i++){
        exe.begin_stage[i] = i+1;
    }
    exe.policy[2] = 6,exe.policy[6] = 6,exe.policy[9] = 6,exe.policy[13] = 6,exe.policy[16] = 6,exe.policy[19] = 6,
    exe.policy[23] = 6,exe.policy[26] = 6,exe.policy[29] = 6,exe.policy[33] = 6,exe.policy[36] = 6,exe.policy[39] = 6;

    for(int i=0;i<2;i++){
        std::cout<<i<<std::endl;
        iter = 0;
       
        for (auto& batch : *data_loader){
            //std::cout<<"enter"<<std::endl;
            /**
            if(iter==4){
                c10::Profile::set_enabled(true);
            }**/
            //std::cout<<batch.data.sizes()<<std::endl;
            //std::cout<<batch.target.sizes()<<std::endl;
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);
            target = target - 1;
            struct timeval start_, end_;
            cudaDeviceSynchronize();
            gettimeofday(&start_, NULL );
            auto loss = net.forward(data,target);
            //std::cout<<loss<<std::endl;
            //const  c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
            //std::cout<<(stats.allocated_bytes)[0].current<<std::endl;
            //std::cout<<"s"<<std::endl;
            loss.backward();
            cudaDeviceSynchronize();
            gettimeofday(&end_, NULL );
            //const  c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
            //std::cout<<(stats.allocated_bytes)[0].current<<"-"<<std::endl;
            int timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
            std::cout<<timeuse<<std::endl;
            if(c10::Profile2::is_enabled()||c10::Profile4::is_enabled()){
                c10::GetCompressProfile()->finish_iteration();
                if(is_compress){
                    c10::GetCompressProfile()->add_total_time2(timeuse);
                }else{
                    c10::GetCompressProfile()->add_total_time1(timeuse);
                }
            }

            if(c10::Profile3::is_enabled()){
                std::cout<<"max_mem:"<<c10::GetCompressProfile()->get_max_mem()<<std::endl;
                c10::GetCompressProfile()->to_txt2("/home/jsxnh/dcgan/build/vgg16-cifar10/memory_load1.txt");
                c10::GetCompressProfile()->finish_iteration2();
            }

            // admm.step();
            SGD.step();

            iter++;
            std::cout<<"time use: "<<timeuse<<"  loss: "<<loss.item<float>()<<std::endl;
            if(c10::ProfilerSwitch::is_enabled()){
                c10::GetProfilerImpl()->add_total_time(timeuse);
                c10::GetProfilerImpl()->finish_iteration();
            }


        }
        
    }

    //c10::GetCompressProfile()->to_txt("/nfs/home/schen/dcgan/build/compress2/2560/vgg16-cifar10_1.txt");
    //hpy
     if(c10::ProfilerSwitch::is_enabled()){
        c10::GetProfilerImpl()->to_txt("result/vgg16_imagenet_128.txt");
    }

    return 0;
}
