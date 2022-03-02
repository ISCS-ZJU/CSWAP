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

/**
void log_transfer_time(torch::Tensor& x){
    torch::Device device1(torch::kCUDA);
    torch::Device device2(torch::kCPU);
    FILE *f = fopen("alexnet-cifar10/transfer_time.txt","a+");
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

    f = fopen("alexnet-cifar10/mem.txt","a+");
    fprintf(f,"%zu\n",x.nbytes());
    fclose(f);
}**/
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
cudaStream_t stream1;


Execution exe;

struct Net : torch::nn::Module {
    Net() {
        conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 6, { 3,3 }).stride(1).padding(1));
        conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(6, 16, { 3,3 }).stride(1).padding(1));
        conv3 = torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, { 3,3 }).stride(1).padding(1));
        conv4 = torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, { 3,3 }).stride(1).padding(1));
        conv5 = torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, { 3,3 }).stride(1).padding(1));

        maxpool1 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2,2}).stride(2));
        maxpool2 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2,2}).stride(2));
        maxpool3 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2,2}).stride(2));
        maxpool4 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2,2}).stride(2));
        maxpool5 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2,2}).stride(2));
      
        
        fc1 = torch::nn::Linear(128,120);
        fc2 = torch::nn::Linear(120,84);
        fc3 = torch::nn::Linear(84,10);
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

        std::function<torch::Tensor(const torch::Tensor&)> functions[19] = {
        std::bind(&torch::nn::Conv2dImpl::forward,conv1,std::placeholders::_1),torch::relu,
        std::bind(&torch::nn::MaxPool2dImpl::forward,maxpool1,std::placeholders::_1),
        std::bind(&torch::nn::Conv2dImpl::forward,conv2,std::placeholders::_1),torch::relu,
        std::bind(&torch::nn::MaxPool2dImpl::forward,maxpool2,std::placeholders::_1),
        std::bind(&torch::nn::Conv2dImpl::forward,conv3,std::placeholders::_1),torch::relu,
        std::bind(&torch::nn::MaxPool2dImpl::forward,maxpool3,std::placeholders::_1),
        std::bind(&torch::nn::Conv2dImpl::forward,conv4,std::placeholders::_1),torch::relu,
        std::bind(&torch::nn::MaxPool2dImpl::forward,maxpool4,std::placeholders::_1),
        std::bind(&torch::nn::Conv2dImpl::forward,conv5,std::placeholders::_1),torch::relu,
        std::bind(&torch::nn::MaxPool2dImpl::forward,maxpool5,std::placeholders::_1),
        torch::relu,
        std::bind(&torch::nn::LinearImpl::forward,fc1,std::placeholders::_1),
        std::bind(&torch::nn::LinearImpl::forward,fc2,std::placeholders::_1),
        std::bind(&torch::nn::LinearImpl::forward,fc3,std::placeholders::_1)
        };

        v = std::vector<std::function<torch::Tensor(const torch::Tensor&)>>(functions,functions+19);
        policy.resize(20,0);
    }
 
    // Implement the Net's algorithm.
    torch::Tensor forward(torch::Tensor x,torch::Tensor target) {
       
        exe.preforward();
        torch::Tensor pre = x;
        for(int index=0;index<v.size();index++){
            exe.pre(pre,ACT);
            if(index==15){
                x = pre.view({-1,128});
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
    torch::nn::Conv2d conv1{ nullptr };
    torch::nn::Conv2d conv2{ nullptr };
    torch::nn::Conv2d conv3{ nullptr };
    torch::nn::Conv2d conv4{ nullptr };
    torch::nn::Conv2d conv5{ nullptr };
   

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
    //torch::nn::Linear fc2{ nullptr };
    //torch::nn::Linear fc3{ nullptr };

    std::vector<std::function<torch::Tensor(const torch::Tensor&)>> v;
    std::vector<int> policy;//操作的输入决策
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

    int batch_size = 2560;
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
        ratio = atof(argv[4]);
    }

    //c10::Passive::set_enabled(true);
    //c10::Profile2::set_enabled(true);
    CIFAR10 c;
    if(zjlab){
        c.init("/nfs/home/schen/dcgan/cifar10/data_batch_1.bin");
    }else{
        c.init("/home/jsxnh/dcgan/cifar10/data_batch_1.bin");
    }
    auto train_dataset = c.map(Normalize(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());
    auto data_loader = torch::data::make_data_loader(
        train_dataset, batch_size);
    cudaStreamCreateWithFlags(&stream1,cudaStreamNonBlocking); 

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
    std::vector<int> policy1,policy2;

    bool is_compress = false;
    int iter = 0;
    //net.policy[1] = 2,net.policy[4] = 2,net.policy[7] = 2,net.policy[10] = 2,
    //net.policy[13] = 2;
    //net.policy[1] = 3,net.policy[4] = 3,net.policy[7] = 3,net.policy[10] = 3,
    //net.policy[13] = 3;
    //is_compress = true;

    exe.init_(30);
    for(int i=0;i<30;i++){
        exe.begin_stage[i] = i+1;
    }

    //exe.policy[1] = 3,exe.policy[4] = 3,exe.policy[7] = 3,exe.policy[10] = 3,
    //exe.policy[13] = 3;

    exe.policy[2] = 6,exe.policy[5] = 6,exe.policy[8] = 6,exe.policy[11] = 6;
    
    
    for(int i=0;i<5;i++){
        std::cout<<i<<std::endl;
        iter = 0;
        //c10::Profile4::set_enabled(true);
        for (auto& batch : *data_loader){
            //std::cout<<batch.data.sizes()<<std::endl;
            //std::cout<<batch.target.sizes()<<std::endl;

            if(iter==4){
                c10::Profile::set_enabled(true);
            }

            auto data = batch.data.to(device);
            auto target = batch.target.to(device);
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
            std::cout<<timeuse<<std::endl;
            /**
            if(c10::Profile2::is_enabled()||c10::Profile4::is_enabled()){
                c10::GetCompressProfile()->finish_iteration();
                if(is_compress){
                    c10::GetCompressProfile()->add_total_time2(timeuse);
                }else{
                    c10::GetCompressProfile()->add_total_time1(timeuse);
                }
            }**/
            /**
            if(iter==3){
                //c10::Profile2::set_enabled(false);
                c10::Profile4::set_enabled(false);
            }**/
            if(c10::ProfilerSwitch::is_enabled()){
                c10::GetProfilerImpl()->add_total_time(timeuse);
                c10::GetProfilerImpl()->finish_iteration();
            }
            
            if(iter==4){
                int time_;
                c10::Profile::set_enabled(false);
                c10::Passive::set_enabled(false);
                c10::GetPolicyMaker()->init(ratio);
                c10::GetPolicyMaker()->print();
                //c10::GetPolicyMaker()->write_lastvisit("/nfs/home/schen/dcgan/build/inceptionv3-cifar100/lastvisit.txt");
                
                //c10::GetPolicyMaker()->print_time();
                //auto policy = c10::GetPolicyMaker()->capuchin(time_);
                if(policy_==1)
                    policy1 = c10::GetPolicyMaker()->make_policy(time_);
                std::cout<<"iteration_time:"<<time_<<std::endl;
                //for(int po:policy1){
                //    std::cout<<po<<",";
                //}
                //std::cout<<std::endl;
                if(policy_==2)
                    policy2 = c10::GetPolicyMaker()->capuchin(time_);
                std::cout<<"capuchin_time:"<<time_<<std::endl;
                for(int po:policy2){
                    std::cout<<po<<",";
                }
                if(policy_==1&&policy1.size()){
                    net.policy = policy1;
                    net.policy[15] = 0;
                }
                if(policy2.size()&&policy_==2){
                    net.policy = policy2;
                    net.policy[15] = 0;
                }
                //std::cout<<std::endl;
                //if(policy1.size())
                //    net.policy = policy1;
                    //iter = 1;
            
            }

            //iter++;
            admm.step();
        }
        /**
        if(i==99){
            net.policy[1] = 3,net.policy[4] = 3,net.policy[7] = 3,net.policy[10] = 3,
            net.policy[13] = 3;
            is_compress = true;
        }**/
    }
    //c10::GetCompressProfile()->to_txt("/nfs/home/schen/dcgan/build/compress2/2560/alexnet-cifar10_.txt");
    //c10::GetCompressProfile()->to_txt4("/nfs/home/schen/dcgan/build/compress4/2560/alexnet-cifar10_.txt");
    if(c10::ProfilerSwitch::is_enabled()){
        c10::GetProfilerImpl()->to_txt("result/alexnet_cifar10_32.txt");
    }

    return 0;
}
