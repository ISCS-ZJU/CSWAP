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


vector<int> begin_stage(69);

bool is_super = false;
bool is_vdnn = false;


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
        fc1 = torch::nn::Linear(512,10);
        
        loss = torch::nn::CrossEntropyLoss();
        for(int i=0;i<20;i++){
            register_module("conv"+std::to_string(i), convs[i]);
            register_module("batch"+std::to_string(i),batchnorm[i]);
        }
        
        register_module("fc1", fc1);
       
        register_module("crossloss", loss);
        v.push_back(std::bind(&torch::nn::Conv2dImpl::forward,convs[0],std::placeholders::_1));
        types.push_back(CONV);
        v.push_back(std::bind(&torch::nn::BatchNorm2dImpl::forward,batchnorm[0],std::placeholders::_1));
        types.push_back(BN);
        v.push_back(torch::relu);
        types.push_back(ACT);
        int index = 1;
        for(int i=0;i<8;i++){
            if((i==0)||(i==1)||(i==3)||(i==5)||(i==7)){
                v.push_back(std::bind(&torch::nn::Conv2dImpl::forward,convs[index],std::placeholders::_1));
                types.push_back(CONV);
                v.push_back(std::bind(&torch::nn::BatchNorm2dImpl::forward,batchnorm[index++],std::placeholders::_1));
                types.push_back(BN);
                v.push_back(torch::relu);
                types.push_back(ACT);
                v.push_back(std::bind(&torch::nn::Conv2dImpl::forward,convs[index],std::placeholders::_1));
                types.push_back(CONV);
                v.push_back(std::bind(&torch::nn::BatchNorm2dImpl::forward,batchnorm[index++],std::placeholders::_1));
                types.push_back(BN);
                v.push_back(torch::relu);
                types.push_back(ACT);
                v.push_back(torch::relu);
                types.push_back(ACT);
            }else{
                v.push_back(std::bind(&torch::nn::Conv2dImpl::forward,convs[index],std::placeholders::_1));
                types.push_back(CONV);
                v.push_back(std::bind(&torch::nn::BatchNorm2dImpl::forward,batchnorm[index++],std::placeholders::_1));
                types.push_back(BN);
                v.push_back(torch::relu);
                types.push_back(ACT);
                v.push_back(std::bind(&torch::nn::Conv2dImpl::forward,convs[index],std::placeholders::_1));
                types.push_back(CONV);
                v.push_back(std::bind(&torch::nn::BatchNorm2dImpl::forward,batchnorm[index++],std::placeholders::_1));
                types.push_back(BN);
                v.push_back(std::bind(&torch::nn::Conv2dImpl::forward,convs[index],std::placeholders::_1));
                types.push_back(CONV);
                v.push_back(std::bind(&torch::nn::BatchNorm2dImpl::forward,batchnorm[index++],std::placeholders::_1));
                types.push_back(BN);
                v.push_back(torch::relu);
                types.push_back(ACT);
                v.push_back(torch::relu);
                types.push_back(ACT);
            }
        }
        v.push_back(std::bind(&torch::nn::AvgPool2dImpl::forward,avg_pool,std::placeholders::_1));
        types.push_back(POOL);
        v.push_back(torch::relu);
        types.push_back(ACT);
        v.push_back(std::bind(&torch::nn::LinearImpl::forward,fc1,std::placeholders::_1));
        types.push_back(FC);
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
    int policy_ = 1;// 1 CSR 2 capuchin 3 vdnn 4 superneurons
    double ratio = 0.8;
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
    //c10::Profile2::set_enabled(true);
    //c10::Profile::set_enabled(true);
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
    

    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Training on GPU" << std::endl;
        device_type = torch::kCUDA;
    } else {
        std::cout << "Training on CPU" << std::endl;
        device_type = torch::kCPU;
    }
    //device_type = torch::kCPU;
    torch::Device device(device_type);

    Net net;
    net.to(device);

    torch::optim::Adam admm(net.parameters(),torch::optim::AdamOptions(0.002));
    exe.init_(100);

    if(zjlab){
        c10::GetPolicyMaker()->read_lastvisit("/nfs/home/schen/dcgan/build/resnet-cifar10/lastvisit.txt",exe.begin_stage);
    }else{
        c10::GetPolicyMaker()->read_lastvisit("/home/jsxnh/dcgan/build/resnet-cifar10/lastvisit.txt",exe.begin_stage);
    }
    
    //net.policy[2] = 2,net.policy[5] = 2,net.policy[12] = 2,net.policy[28] = 2,net.policy[44] = 2,
    //net.policy[60] = 2,net.policy[9] = 2,net.policy[16] = 2,net.policy[32] = 2,net.policy[48] = 2,
    //net.policy[62] = 2,net.policy[19] = 2,net.policy[35] = 2,net.policy[51] = 2,net.policy[25] = 2,
    //net.policy[41] = 2;

    std::vector<int> policy1,policy2;
    //net.policy[1] = 3;
    //net.policy[0] = 3;
    //net.policy[10] = 3;
    //net.policy[15] = 3;
    //net.policy[18] = 3;
    //net.policy[25] = 3;
    /**
    net.policy[2] = 3;
    net.policy[5] = 3,net.policy[12] = 3,net.policy[28] = 3,net.policy[44] = 3,
    net.policy[60] = 3,net.policy[9] = 3,net.policy[16] = 3,net.policy[32] = 3,net.policy[48] = 3,
    net.policy[62] = 3,net.policy[19] = 3,net.policy[35] = 3,net.policy[51] = 3,net.policy[25] = 3,
    net.policy[41] = 3;
    **/

   /**
    exe.policy[2] = 3,exe.policy[5] = 3,exe.policy[9] = 3,exe.policy[12] = 3;
    exe.policy[16] = 3,exe.policy[19] = 3,exe.policy[25] = 3,exe.policy[28] = 3;
    exe.policy[32] = 3,exe.policy[35] = 3,exe.policy[41] = 3,exe.policy[44] = 3;
    exe.policy[48] = 3,exe.policy[51] = 3,exe.policy[57] = 3,exe.policy[60] = 3;
    exe.policy[21] = 3,exe.policy[37] = 3,exe.policy[53] = 3;//conv前不是relu
    **/
    exe.policy[2] = 6,exe.policy[5] = 6,exe.policy[9] = 6,exe.policy[12] = 6;
    exe.policy[16] = 6,exe.policy[19] = 6,exe.policy[25] = 6,exe.policy[28] = 6;
    exe.policy[32] = 6,exe.policy[35] = 6,exe.policy[41] = 6,exe.policy[44] = 6;
    exe.policy[48] = 6,exe.policy[51] = 6,exe.policy[57] = 6,exe.policy[60] = 6;
    exe.policy[21] = 6,exe.policy[37] = 6,exe.policy[53] = 6;//conv前不是relu
    
    int iter = 0;
    for(int i=0;i<3;i++){
        
        for (auto& batch : *data_loader){
           
            
            if(iter==2){
                c10::Profile::set_enabled(true);
            }
            //for(int i=0;i<60;i++){
            //    net.policy[i] = rand()%3;
            //}

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
                c10::GetCompressProfile()->add_total_time1(timeuse);
            }**/
            //fprintf(f,"%d\n",timeuse);
            admm.step();
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
                    net.policy = policy1;
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
                    net.policy = policy2;
                }
                
                if(policy_==3){
                    c10::Passive::set_enabled(true);
                    std::cout<<"+,"<<std::endl;
                    is_vdnn = true;
                    for(int i=0;i<65;i++){
                        net.policy[i] = 0;
                    }
                }
                if(policy_==4){
                    std::cout<<"+,"<<std::endl;
                    is_super = true;
                    for(int i=0;i<65;i++){
                        net.policy[i] = 0;
                    }
                }
              
            
            }
            //iter++;
            //if(iter==3){
            //    c10::Profile::set_enabled(false);
            //}
            //if(iter==3)
            //    break;
            
        }
    }

    //c10::GetCompressProfile()->to_txt4("/nfs/home/schen/dcgan/build/compress4/2560/resnet-cifar10.txt");
    if(c10::ProfilerSwitch::is_enabled()){
        c10::GetProfilerImpl()->to_txt("result/resnet_cifar10_32.txt");
    }
    return 0;
}
