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


Execution ex;

struct Fire: torch::nn::Module{
    Fire(int inc = 96, int outc = 128, int squeezec = 16) {
        conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(inc, squeezec, {1,1}));
        batchnorm1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(squeezec));
        
        conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(squeezec, (outc/2), {1,1}));
        batchnorm2 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(outc/2));

        conv3 = torch::nn::Conv2d(torch::nn::Conv2dOptions(squeezec, (outc/2), {3,3}).padding(1));
        batchnorm3 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(outc/2));

        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv3", conv3);
        register_module("batchnorm1", batchnorm1);
        register_module("batchnorm2", batchnorm2);
        register_module("batchnorm3", batchnorm3);
    }
    
    torch::Tensor forward(torch::Tensor x) {
        torch::Tensor pre = x;
        ex.pre(pre);
        x = conv1->forward(x);
        ex.post(pre,x);
        x.grad_fn()->func_ = std::bind(&torch::nn::Conv2dImpl::forward,conv1,std::placeholders::_1);
        pre = x;
        ex.pre(pre);
        x = batchnorm1->forward(x);
        ex.post(pre,x);
        x.grad_fn()->func_ = std::bind(&torch::nn::BatchNorm2dImpl::forward,batchnorm1,std::placeholders::_1);
        pre = x;
        ex.pre(pre);
        x = torch::relu(x);
        ex.post(pre,x);
        //std::cout<<x.tensor_id<<";";
        x.grad_fn()->func_ = torch::relu;
        ex.pre(x);
        torch::Tensor x1 = conv2->forward(x);
        ex.post(x,x1);
        x1.grad_fn()->func_ = std::bind(&torch::nn::Conv2dImpl::forward,conv2,std::placeholders::_1);
        pre = x1;
        ex.pre(pre);
        x1 = batchnorm2->forward(x1);
        ex.post(pre,x1);
        x1.grad_fn()->func_ = std::bind(&torch::nn::BatchNorm2dImpl::forward,batchnorm2,std::placeholders::_1);
        pre = x1;
        ex.pre(pre);
        x1 = torch::relu(x1);
        x1 = torch::relu(pre);
        ex.post(pre,x1);
        //std::cout<<x1.tensor_id<<";";
        x1.grad_fn()->func_ = torch::relu;
        ex.pre(x);
        torch::Tensor x2 = conv3->forward(x);
        ex.post(x,x2);
        x2.grad_fn()->func_ = std::bind(&torch::nn::Conv2dImpl::forward,conv3,std::placeholders::_1);
        pre = x2;
        ex.pre(pre);
        x2 = batchnorm3->forward(x2);
        ex.post(pre,x2);
        x2.grad_fn()->func_ = std::bind(&torch::nn::BatchNorm2dImpl::forward,batchnorm3,std::placeholders::_1);
        pre = x2;
        ex.pre(pre);
        x2 = torch::relu(x2);
        
        //x1 shape: [2560, 64, 16, 16]    batch_size,channel, 
        //std::cout<<"x1 shape: "<<x1.sizes()<<std::endl;
        //std::cout<<"x2 shape: "<<x2.sizes()<<std::endl;
        ex.post(pre,x2);
        //std::cout<<x2.tensor_id<<";";
        x2.grad_fn()->func_ = torch::relu;
        vector<torch::Tensor> inputs{x1,x2};
        ex.pre(inputs);
        x = torch::cat({x1, x2}, 1);   //cat
        ex.post(inputs,x);
        return x;
    }

    torch::nn::Conv2d conv1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::Conv2d conv3{nullptr};
    torch::nn::BatchNorm2d batchnorm1{nullptr};
    torch::nn::BatchNorm2d batchnorm2{nullptr};
    torch::nn::BatchNorm2d batchnorm3{nullptr};
};

torch::Tensor shortcut(torch::Tensor x, Fire fire){
    torch::Tensor x1 = fire.forward(x);
    vector<torch::Tensor> inputs{x,x1};
    ex.pre(inputs);
    torch::Tensor y = torch::add(x,x1);
    ex.post(inputs,y);
    y.grad_fn()->operation = 2;
    return y;
}

struct Net : torch::nn::Module {
    Net() {
        conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 96, {7,7}).padding(1).stride(2));
        batchnorm1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(96));
        maxpool1 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({3,3}).stride(2));
        fire2 = Fire(96, 128, 16);
        fire3 = Fire(128, 128, 16);
        fire4 = Fire(128, 256, 32);
        maxpool4 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({3,3}).stride(2));
        fire5 = Fire(256, 256, 32);
        fire6 = Fire(256, 384, 48);
        fire7 = Fire(384, 384, 48);
        fire8 = Fire(384, 512, 64);
        maxpool8 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({3,3}).stride(2));
        fire9 = Fire(512, 512, 64);
        dropout9 = torch::nn::Dropout2d(torch::nn::Dropout2dOptions().p(0.5));
        //class_num: 200
        conv10 = torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 200, {1,1}));
        //avg_pool10 = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(1));
        avg_pool10 = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(1));
        loss = torch::nn::CrossEntropyLoss();


        register_module("conv1", conv1);
        register_module("batchnorm1", batchnorm1);
        //register_module("maxpool1", maxpool1);
        register_module("fire2", std::make_shared<Fire>(fire2));
        register_module("fire3", std::make_shared<Fire>(fire3));
        register_module("fire4", std::make_shared<Fire>(fire4));
        //register_module("maxpool4", maxpool4);
        register_module("fire5", std::make_shared<Fire>(fire5));
        register_module("fire6", std::make_shared<Fire>(fire6));
        register_module("fire7", std::make_shared<Fire>(fire7));
        register_module("fire8", std::make_shared<Fire>(fire8));
        //register_module("maxpool8", maxpool8);
        register_module("fire9", std::make_shared<Fire>(fire9));
        register_module("dropout9", dropout9);
        register_module("conv10", conv10);
        //register_module("avg_pool10", avg_pool10);
        register_module("crossloss", loss);

        const int layer_num = 17;
        std::function<torch::Tensor(const torch::Tensor&)> functions[layer_num] = {
            std::bind(&torch::nn::Conv2dImpl::forward, conv1, std::placeholders::_1),
            std::bind(&torch::nn::BatchNorm2dImpl::forward, batchnorm1, std::placeholders::_1),
            torch::relu,
            std::bind(&torch::nn::MaxPool2dImpl::forward, maxpool1, std::placeholders::_1),

            std::bind(&Fire::forward, fire2, std::placeholders::_1),
            std::bind(&shortcut, std::placeholders::_1, fire3),
            std::bind(&Fire::forward, fire4, std::placeholders::_1),
            std::bind(&torch::nn::MaxPool2dImpl::forward, maxpool4, std::placeholders::_1),

            std::bind(&shortcut, std::placeholders::_1, fire5),
            std::bind(&Fire::forward, fire6, std::placeholders::_1),
            std::bind(&shortcut, std::placeholders::_1, fire7),
            std::bind(&Fire::forward, fire8, std::placeholders::_1),
            std::bind(&torch::nn::MaxPool2dImpl::forward, maxpool8, std::placeholders::_1),

            std::bind(&Fire::forward, fire9, std::placeholders::_1),
            std::bind(&torch::nn::Dropout2dImpl::forward, dropout9, std::placeholders::_1),

            std::bind(&torch::nn::Conv2dImpl::forward, conv10, std::placeholders::_1),

            std::bind(&torch::nn::AdaptiveAvgPool2dImpl::forward, avg_pool10, std::placeholders::_1),
        };

        v = std::vector<std::function<torch::Tensor(const torch::Tensor&)>>(functions, functions+layer_num);

    } //end of Net()
    
    torch::Tensor forward(torch::Tensor x,torch::Tensor target) {
        
        //std::cout<<"x input shape: "<<x.sizes()<<std::endl;
        ex.preforward();
        torch::Tensor pre = x;


        for (int index = 0; index < v.size(); index++){
            if(index==0||index==1||index==2||index==3||index==7
            ||index==12||index==14||index==15||index==16){
                ex.pre(pre);
            }
            x = v[index](pre); 
            if(index==0||index==1||index==2||index==3||index==7
            ||index==12||index==14||index==15||index==16){
                ex.post(pre,x);
                x.grad_fn()->func_ = v[index];
            }
            pre = x;
            //std::cout<<"output size: "<<x.sizes()<<std::endl;
        }

        ex.pre(pre,ACT);
        x = torch::flatten(pre, 1);
        ex.post(pre,x,ACT,true);
        pre = x;
        ex.pre({pre,target},ACT);
        x = loss->forward(pre, target);
        ex.post({pre,target},x,ACT,true);
        ex.postforward();
        return x;
    }
    
    torch::nn::Conv2d conv1{ nullptr };
    torch::nn::BatchNorm2d batchnorm1{nullptr};
    torch::nn::MaxPool2d maxpool1{nullptr};
    torch::nn::MaxPool2d maxpool4{nullptr};
    torch::nn::MaxPool2d maxpool8{nullptr};
    torch::nn::Dropout2d dropout9{nullptr};   
    torch::nn::Conv2d conv10{ nullptr };
    torch::nn::AdaptiveAvgPool2d avg_pool10{nullptr};
    torch::nn::CrossEntropyLoss loss{nullptr};
    Fire fire2, fire3, fire4, fire5, fire6, fire7, fire8, fire9;

    std::vector<std::function<torch::Tensor(const torch::Tensor&)>> v; 
};


int main(int argc, char * argv[]){
    
    int batch_size = 128;
    bool is_compress = false;

    if(argc>=2){
        batch_size = atoi(argv[1]);
    }
    if (argc>=3){
        is_compress = atoi(argv[2]);
    }
    printf("batch_size = %d\n", batch_size);

    TinyImagenet c("/home/jsxnh/dcgan/tinyimagenet/train_sqlit_data_0.bin","/home/jsxnh/dcgan/tinyimagenet/train_sqlit_label_0.bin");

    auto train_dataset = c.map(Normalize(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());
    auto data_loader = torch::data::make_data_loader(
        train_dataset, batch_size);  //batchsize
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
    ex.init_(200);
    c10::GetPolicyMaker()->read_lastvisit("/home/jsxnh/dcgan/build/squeezenet-imagenet/lastvisit.txt",ex.begin_stage);
    //bool is_compress = false;
    torch::optim::Adam admm(net.parameters(),torch::optim::AdamOptions(1e-5));

    ex.policy[3] = 6,ex.policy[6] = 6,ex.policy[9] = 6,
    ex.policy[13] = 6,ex.policy[16] = 6,ex.policy[19] = 6,
    ex.policy[24] = 6,ex.policy[27] = 6,ex.policy[30] = 6,
    ex.policy[35] = 6,ex.policy[38] = 6,ex.policy[41] = 6;
    ex.policy[46] = 6,ex.policy[49] = 6,ex.policy[52] = 6,
    ex.policy[56] = 6,ex.policy[59] = 6,ex.policy[62] = 6,
    ex.policy[67] = 6,ex.policy[70] = 6,ex.policy[73] = 6,
    ex.policy[78] = 6,ex.policy[81] = 6,ex.policy[84] = 6;
    ex.policy[88] = 6;

    int iter;
    for(int epoch = 0; epoch < 3; epoch++){      //epoch
        std::cout << "epoch: " << epoch << std::endl;
        iter = 0;
        
        for (auto& batch : *data_loader){
            /**
            if(iter==4){
                c10::Profile::set_enabled(true);
            }**/
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);
            target = target - 1; ///////////////////1-200 --> 0-199
            struct timeval start_, end_;
            cudaDeviceSynchronize();
            gettimeofday(&start_, NULL);

            auto loss = net.forward(data,target);
            loss.backward();
            admm.step();
            
            cudaDeviceSynchronize();
            gettimeofday(&end_, NULL);
            int timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
            std::cout<<"time use:" << timeuse << " loss: " << loss.item<float>() << std::endl;
            if(c10::Profile4::is_enabled()){
                c10::GetCompressProfile()->finish_iteration();
                if(is_compress){
                    c10::GetCompressProfile()->add_total_time2(timeuse);
                }else{
                    c10::GetCompressProfile()->add_total_time1(timeuse);
                }
            }
            if(c10::ProfilerSwitch::is_enabled()){
                c10::GetProfilerImpl()->add_total_time(timeuse);
                c10::GetProfilerImpl()->finish_iteration();
            }
            iter++;
        }
    }

    if(c10::ProfilerSwitch::is_enabled()){
        c10::GetProfilerImpl()->to_txt("result/squeezenet_imagenet_128.txt");
    }


    return 0;
}
