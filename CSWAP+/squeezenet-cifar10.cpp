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

/**
extern "C"{
void PreGPUcompression(int arraySize, int kernelSize, float* arrayGPU, int* valueIndex,int dimsize,int blocksize,cudaStream_t stream);
void PreAcc(int* valueIndex,int KernelSize,int* num,int dimsize,int blocksize,cudaStream_t stream);
void GPUcompression(int arraySize, int kernelSize, float* arrayGPU, float* compressedList, int* compressedValueIndex, uint32_t* compressedBinIndex,int dimsize,int blocksize,cudaStream_t stream);
}**/

class CIFAR10 :public torch::data::datasets::Dataset<CIFAR10>{
public:

    torch::data::Example<> get(size_t index) {
        return {images_[index], targets_[index]};
    }
    torch::optional<size_t> size() const {
        return images_.size(0);
    }
    CIFAR10(){}
    CIFAR10(std::string path){
        images_ = torch::empty({10000, 3,32, 32}, torch::kByte);
        targets_ =  torch::empty(10000, torch::kByte);
        std::ifstream reader(path,std::ios::binary);
        for(int i=0;i<10000;i++){
            reader.read(reinterpret_cast<char*>(targets_.data_ptr())+i,1);
            reader.read(reinterpret_cast<char*>(images_.data_ptr())+i*3072,3072);
        }
        targets_ = targets_.to(torch::kInt64);
        images_ = images_.to(torch::kFloat32).div_(255);
        //std::cout<<images_<<std::endl;
    }
    void init(std::string path){
        images_ = torch::empty({10000, 3,32, 32}, torch::kByte);
        targets_ =  torch::empty(10000, torch::kByte);
        std::ifstream reader(path,std::ios::binary);
        for(int i=0;i<10000;i++){
            reader.read(reinterpret_cast<char*>(targets_.data_ptr())+i,1);
            reader.read(reinterpret_cast<char*>(images_.data_ptr())+i*3072,3072);
        }
        targets_ = targets_.to(torch::kInt64);
        images_ = images_.to(torch::kFloat32).div_(255);
    }
private:
    torch::Tensor images_,targets_;
};



bool is_super = false;
bool is_vdnn = false;

Execution exe;
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
        exe.pre(pre,CONV);
        x = conv1->forward(pre);
        exe.post(pre,x,CONV);
        x.grad_fn()->func_ = std::bind(&torch::nn::Conv2dImpl::forward,conv1,std::placeholders::_1);
        pre = x;
        exe.pre(pre,BN);
        x = batchnorm1->forward(pre);
        exe.post(pre,x,BN);
        x.grad_fn()->func_ = std::bind(&torch::nn::BatchNorm2dImpl::forward,batchnorm1,std::placeholders::_1);
        pre = x;
        exe.pre(pre,ACT);
        x = torch::relu(pre);
        exe.post(pre,x,ACT);
        //std::cout<<x.tensor_id<<";";
        x.grad_fn()->func_ = torch::relu;
        exe.pre(x,CONV);
        torch::Tensor x1 = conv2->forward(x);
        exe.post(x,x1,CONV);
        x1.grad_fn()->func_ = std::bind(&torch::nn::Conv2dImpl::forward,conv2,std::placeholders::_1);
        pre = x1;
        exe.pre(pre,BN);
        x1 = batchnorm2->forward(pre);
        exe.post(pre,x1,BN);
        x1.grad_fn()->func_ = std::bind(&torch::nn::BatchNorm2dImpl::forward,batchnorm2,std::placeholders::_1);
        pre = x1;
        exe.pre(pre,ACT);
        x1 = torch::relu(pre);
        exe.post(pre,x1,ACT);
        //std::cout<<x1.tensor_id<<";";
        x1.grad_fn()->func_ = torch::relu;
        exe.pre(x,CONV);
        torch::Tensor x2 = conv3->forward(x);
        exe.post(x,x2,CONV);
        x2.grad_fn()->func_ = std::bind(&torch::nn::Conv2dImpl::forward,conv3,std::placeholders::_1);
        pre = x2;
        exe.pre(pre,BN);
        x2 = batchnorm3->forward(pre);
        exe.post(pre,x2,BN);
        x2.grad_fn()->func_ = std::bind(&torch::nn::BatchNorm2dImpl::forward,batchnorm3,std::placeholders::_1);
        pre = x2;
        exe.pre(pre,ACT);
        x2 = torch::relu(pre);
        exe.post(pre,x2,ACT);
        //std::cout<<x2.tensor_id<<";";
        x2.grad_fn()->func_ = torch::relu;
        vector<torch::Tensor> inputs{x1,x2};
        exe.pre(inputs);
        x = torch::cat({x1, x2}, 1);
        exe.post(inputs,x);
        //x.grad_fn()->operation = 1;
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
    exe.pre(inputs);
    torch::Tensor y = torch::add(x,x1);
    exe.post(inputs,y);
    y.grad_fn()->operation = 2;
    return y;
}

struct Net : torch::nn::Module {
    Net() {
        conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 96, {3,3}).padding(1));
        batchnorm1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(96));
        maxpool1 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2,2}));
        fire2 = Fire(96, 128, 16);
        fire3 = Fire(128, 128, 16);
        fire4 = Fire(128, 256, 32);
        maxpool4 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2,2}));
        fire5 = Fire(256, 256, 32);
        fire6 = Fire(256, 384, 48);
        fire7 = Fire(384, 384, 48);
        fire8 = Fire(384, 512, 64);
        maxpool8 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2,2}));
        fire9 = Fire(512, 512, 64);
        //class_num: 10
        conv10 = torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 10, {1,1}));
        avg_pool10 = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(1));
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
        register_module("conv10", conv10);
        //register_module("avg_pool10", avg_pool10);
        register_module("crossloss", loss);

        std::function<torch::Tensor(const torch::Tensor&)> functions[16] = {
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
            std::bind(&torch::nn::Conv2dImpl::forward, conv10, std::placeholders::_1),
            std::bind(&torch::nn::AvgPool2dImpl::forward, avg_pool10, std::placeholders::_1),
        };
        Layer_type types_[16] = {CONV,BN,ACT,POOL,ACT,ACT,ACT,POOL,ACT,ACT,ACT,ACT,POOL,ACT,CONV,POOL};
        v = std::vector<std::function<torch::Tensor(const torch::Tensor&)>>(functions, functions+16);
        types = std::vector<Layer_type>(types_,types_+16);

    } //end of Net()
    
    torch::Tensor forward(torch::Tensor x,torch::Tensor target) {
        exe.preforward();
        torch::Tensor pre = x;
        

        for (int index = 0; index < v.size(); index++){
            //std::cout<<index<<std::endl;
            if(index==0||index==1||index==2||index==3||index==7
            ||index==12||index==14||index==15){
                exe.pre(pre,types[index]);
            }
            x = v[index](pre); 
            if(index==0||index==1||index==2||index==3||index==7
            ||index==12||index==14||index==15){
                exe.post(pre,x,types[index]);
                x.grad_fn()->func_ = v[index];
            }
            pre = x;
        }
        exe.pre(pre,ACT);
        x = torch::flatten(pre, 1);
        exe.post(pre,x,ACT,true);
        pre = x;
        exe.pre({pre,target},ACT);
        x = loss->forward(pre, target);
        exe.post({pre,target},x,ACT,true);
        exe.postforward();
        return x;
    }
    
    torch::nn::Conv2d conv1{ nullptr };
    torch::nn::BatchNorm2d batchnorm1{nullptr};
    torch::nn::MaxPool2d maxpool1{nullptr};
    torch::nn::MaxPool2d maxpool4{nullptr};
    torch::nn::MaxPool2d maxpool8{nullptr};
    torch::nn::Conv2d conv10{ nullptr };
    torch::nn::AvgPool2d avg_pool10{nullptr};
    torch::nn::CrossEntropyLoss loss{nullptr};
    Fire fire2, fire3, fire4, fire5, fire6, fire7, fire8, fire9;

    std::vector<std::function<torch::Tensor(const torch::Tensor&)>> v; 
    std::vector<Layer_type> types;
};


struct Normalize : public torch::data::transforms::TensorTransform<> {
    Normalize(float mean, float stddev)
        : mean_(torch::tensor(mean)), stddev_(torch::tensor(stddev)) {}
    torch::Tensor operator()(torch::Tensor input) {
        return input.sub_(mean_).div_(stddev_);
    }
    torch::Tensor mean_, stddev_;
};


int main(int argc,char* argv[]){

    
    int batch_size = 1280;
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

    CIFAR10 c;
    if(zjlab){
        c.init("/nfs/home/schen/dcgan/cifar10/data_batch_1.bin");
    }else{
        c.init("/home/jsxnh/dcgan/cifar10/data_batch_1.bin");
    }
   
   
    auto train_dataset = c.map(Normalize(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());
    auto data_loader = torch::data::make_data_loader(
        train_dataset, batch_size);  //batchsize


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
    exe.init_(200);

    if(zjlab){
        c10::GetPolicyMaker()->read_lastvisit("/nfs/home/schen/dcgan/build/squeezenet-cifar10/lastvisit.txt",exe.begin_stage);
    }else{
        c10::GetPolicyMaker()->read_lastvisit("/home/jsxnh/dcgan/build/squeezenet-cifar10/lastvisit.txt",exe.begin_stage);
    }
    
    torch::optim::Adam admm(net.parameters(),torch::optim::AdamOptions(0.002));
    //bool is_compress = false;
    /**
    ex.policy[6]=2,ex.policy[9]=2,ex.policy[12]=2,ex.policy[16]=2,
    ex.policy[19]=2,ex.policy[22]=2,ex.policy[27]=2,ex.policy[30]=2,
    ex.policy[33]=2,ex.policy[38]=2,ex.policy[41]=2,ex.policy[44]=2,
    ex.policy[49]=2,ex.policy[52]=2,ex.policy[55]=2,ex.policy[59]=2,
    ex.policy[62]=2,ex.policy[65]=2,ex.policy[70]=2,ex.policy[73]=2,
    ex.policy[76]=2,ex.policy[81]=2,ex.policy[84]=2,ex.policy[87]=2;
    **/
    
    //ex.policy[6]=3,ex.policy[9]=3,ex.policy[12]=3,ex.policy[16]=3,
    //ex.policy[19]=3,ex.policy[22]=3,ex.policy[27]=3,ex.policy[30]=3,
    //ex.policy[33]=3,ex.policy[38]=3,ex.policy[41]=3,ex.policy[44]=3,
    //ex.policy[49]=3,ex.policy[52]=3,ex.policy[55]=3,ex.policy[59]=3,
    //ex.policy[62]=3,ex.policy[65]=3,ex.policy[70]=3,ex.policy[73]=3,
    //ex.policy[76]=3,ex.policy[81]=3,ex.policy[84]=3,ex.policy[87]=3;
    //is_compress = true;

    /**
    exe.policy[3] = 2,exe.policy[6] = 2,exe.policy[9] = 2,
    exe.policy[13] = 2,exe.policy[16] = 2,exe.policy[19] = 2,
    exe.policy[24] = 2,exe.policy[27] = 2,exe.policy[30] = 2,
    exe.policy[35] = 2,exe.policy[38] = 2,exe.policy[41] = 2;
    exe.policy[46] = 2,exe.policy[49] = 2,exe.policy[52] = 2,
    exe.policy[56] = 2,exe.policy[59] = 2,exe.policy[62] = 2,
    exe.policy[67] = 2,exe.policy[70] = 2,exe.policy[73] = 2,
    exe.policy[78] = 2,exe.policy[81] = 2,exe.policy[84] = 2;
    exe.policy[88] = 2;**/
    /**
    exe.policy[3] = 6,exe.policy[6] = 3,exe.policy[9] = 3,
    exe.policy[13] = 6,exe.policy[16] = 3,exe.policy[19] = 3,
    exe.policy[24] = 6,exe.policy[27] = 3,exe.policy[30] = 3,
    exe.policy[35] = 6,exe.policy[38] = 3,exe.policy[41] = 3;
    exe.policy[46] = 6,exe.policy[49] = 3,exe.policy[52] = 3,
    exe.policy[56] = 6,exe.policy[59] = 3,exe.policy[62] = 3,
    exe.policy[67] = 6,exe.policy[70] = 3,exe.policy[73] = 3,
    exe.policy[78] = 6,exe.policy[81] = 3,exe.policy[84] = 3;
    exe.policy[88] = 6;**/

    exe.policy[3] = 6,exe.policy[6] = 6,exe.policy[9] = 6,
    exe.policy[13] = 6,exe.policy[16] = 6,exe.policy[19] = 6,
    exe.policy[24] = 6,exe.policy[27] = 6,exe.policy[30] = 6,
    exe.policy[35] = 6,exe.policy[38] = 6,exe.policy[41] = 6;
    exe.policy[46] = 6,exe.policy[49] = 6,exe.policy[52] = 6,
    exe.policy[56] = 6,exe.policy[59] = 6,exe.policy[62] = 6,
    exe.policy[67] = 6,exe.policy[70] = 6,exe.policy[73] = 6,
    exe.policy[78] = 6,exe.policy[81] = 6,exe.policy[84] = 6;
    exe.policy[88] = 6;
    

    int iter;

    std::vector<int> policy1,policy2;
    iter = 0;
    for(int epoch = 0; epoch < 3; epoch++){      //epoch
        //std::cout << "epoch: " << epoch << std::endl;
        
        //c10::Profile4::set_enabled(true);
        for (auto& batch : *data_loader){
            /**
            if(iter==4){
                c10::Profile::set_enabled(true);
            }**/
            if(iter==2){
                c10::Profile::set_enabled(true);
            }
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);
            struct timeval start_, end_;
            cudaDeviceSynchronize();
            gettimeofday(&start_, NULL);

            auto loss = net.forward(data,target);
            loss.backward();
            admm.step();
            
            cudaDeviceSynchronize();
            gettimeofday(&end_, NULL);
            int timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
            //std::cout<<"time use:" << timeuse << "loss:" << loss.item<float>() << std::endl;
            std::cout<< timeuse<< std::endl;
            /**
            if(c10::Profile4::is_enabled()){
                c10::GetCompressProfile()->finish_iteration();
                if(is_compress){
                    c10::GetCompressProfile()->add_total_time2(timeuse);
                }else{
                    c10::GetCompressProfile()->add_total_time1(timeuse);
                }
            }**/
            if(c10::ProfilerSwitch::is_enabled()){
                c10::GetProfilerImpl()->add_total_time(timeuse);
                c10::GetProfilerImpl()->finish_iteration();
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
                    exe.policy = policy1;
                }
                //for(int po:policy1){
                //    std::cout<<po<<",";
                //}
                //std::cout<<std::endl;
                if(policy_==2){
                    policy2 = c10::GetPolicyMaker()->capuchin(time_);
                    std::cout<<"capuchin_time:"<<time_<<std::endl;
                    for(int po:policy2){
                        std::cout<<po<<",";
                    }
                    if(policy2.size()==0) return 0;
                    exe.policy = policy2;
                }
                if(policy_==3){
                    c10::Passive::set_enabled(true);
                    std::cout<<"+,"<<std::endl;
                    is_vdnn = true;
                    for(int i=0;i<80;i++){
                        exe.policy[i] = 0;
                    }
                }
                if(policy_==4){
                    std::cout<<"+,"<<std::endl;
                    is_super = true;
                    for(int i=0;i<80;i++){
                        exe.policy[i] = 0;
                    }
                }
            
            }
            
            //iter++;
        }
    }

    //c10::GetCompressProfile()->to_txt4("/nfs/home/schen/model-imagenet/build/result/squeezenet-cifar10.txt");
    //c10::GetCompressProfile()->to_txt4("/nfs/home/schen/dcgan/build/compress4/2560/squeezenet-cifar10_.txt");

    if(c10::ProfilerSwitch::is_enabled()){
        c10::GetProfilerImpl()->to_txt("result/squeezenet_cifar10_32.txt");
    }
    return 0;
}