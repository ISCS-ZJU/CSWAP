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
#include <ATen/core/zfp.h>
#include "Execution.h"




cudaStream_t stream1;


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

        avg_pool = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(1));
        fc1 = torch::nn::Linear(512,10);
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
        register_module("crossloss", loss);

        std::function<torch::Tensor(const torch::Tensor&)> functions[45] = {
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
        std::bind(&torch::nn::MaxPool2dImpl::forward,maxpool5,std::placeholders::_1),
        std::bind(&torch::nn::AvgPool2dImpl::forward,avg_pool,std::placeholders::_1)};

        v = std::vector<std::function<torch::Tensor(const torch::Tensor&)>>(functions,functions+45);
        Layer_type types_[45]={CONV,BN,ACT,CONV,BN,ACT,POOL,CONV,BN,ACT,CONV,BN,ACT,POOL,CONV,BN,ACT,CONV,BN,ACT
        ,CONV,BN,ACT,POOL,CONV,BN,ACT,CONV,BN,ACT,CONV,BN,ACT,POOL,CONV,BN,ACT,CONV,BN,ACT,CONV,BN,ACT,POOL,POOL};
        types = std::vector<Layer_type>(types_,types_+45);
        policy.resize(45,0);
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
    torch::nn::CrossEntropyLoss loss{nullptr};
    torch::nn::AvgPool2d avg_pool{nullptr};
    //torch::nn::Linear fc2{ nullptr };
    //torch::nn::Linear fc3{ nullptr };

    std::vector<std::function<torch::Tensor(const torch::Tensor&)>> v;
    std::vector<int> policy;//操作的输入决策
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
    //c10::Passive::set_enabled(true);
    //c10::Profile2::set_enabled(true);
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
    //device_type = torch::kCPU;
    torch::Device device(device_type);
    //std::cout<<"***"<<std::endl;
    Net net;
    //std::cout<<"***"<<std::endl;
    net.to(device);
    //std::cout<<"***"<<std::endl;

    //size_t total_size = 0;
    //for(torch::Tensor x:net.parameters()){
    //    total_size+=x.nbytes();
    //}
    //std::cout<<"weight size:"<<total_size<<std::endl;

    torch::optim::Adam admm(net.parameters(),torch::optim::AdamOptions(0.001));
    //bool is_compress = false;

    int iter = 0;
    //net.policy[2] = 2,net.policy[6] = 2,net.policy[9] = 2,
    //net.policy[13] = 2,net.policy[16] = 2,net.policy[19] = 2
    //,net.policy[23] = 2,net.policy[26] = 2,net.policy[29] = 2,
    //net.policy[33] = 2,net.policy[36] = 2,net.policy[39] = 2;
    
    //net.policy[2] = 3,net.policy[6] = 3,net.policy[9] = 3,net.policy[13] = 3,net.policy[16] = 3,net.policy[19] = 3
    //,net.policy[23] = 3,net.policy[26] = 3,net.policy[29] = 3,net.policy[33] = 3,net.policy[36] = 3,net.policy[39] = 3;
    
    //net.policy[2] = 3,net.policy[6] = 3,net.policy[9] = 3,net.policy[13] = 3,net.policy[16] = 3,net.policy[19] = 3
    //,net.policy[23] = 3,net.policy[26] = 3,net.policy[29] = 2,net.policy[33] = 2,net.policy[36] = 2,net.policy[39] = 2;
    //is_compress = true;

    //std::cout<<"***"<<std::endl;
    
    //net.policy[2] = 6;

    
    
   
    /**
    for(int i=0;i<43;i++){
        net.policy[i] = 0;
    }**/
    exe.init_(100);

    for(int i=0;i<50;i++){
        exe.begin_stage[i] = i+1;
        exe.policy[i] = 2;
    }

    //exe.policy[2] = 2,exe.policy[6] = 2,exe.policy[9] = 2,exe.policy[13] = 2,exe.policy[16] = 2,exe.policy[19] = 2,
    //exe.policy[23] = 2,exe.policy[26] = 2,exe.policy[29] = 2,exe.policy[33] = 2,exe.policy[36] = 2,exe.policy[39] = 2;

    //exe.policy[2] = 3,exe.policy[6] = 3,exe.policy[9] = 3,exe.policy[13] = 3,exe.policy[16] = 3,exe.policy[19] = 3,
    //exe.policy[23] = 3,exe.policy[26] = 3,exe.policy[29] = 3,exe.policy[33] = 3,exe.policy[36] = 3,exe.policy[39] = 3;


    //exe.policy[2] = 3,exe.policy[6] = 6,exe.policy[9] = 3,exe.policy[13] = 6,exe.policy[16] = 3,exe.policy[19] = 3,
    //exe.policy[23] = 6,exe.policy[26] = 3,exe.policy[29] = 3,exe.policy[33] = 6,exe.policy[36] = 3,exe.policy[39] = 3;

    exe.policy[2] = 6,exe.policy[6] = 6,exe.policy[9] = 6,exe.policy[13] = 6,exe.policy[16] = 6,exe.policy[19] = 6,
    exe.policy[23] = 6,exe.policy[26] = 6,exe.policy[29] = 6,exe.policy[33] = 6,exe.policy[36] = 6,exe.policy[39] = 6;
    

    std::vector<int> policy1,policy2;
    iter = 0;
    for(int i=0;i<50;i++){
        //std::cout<<i<<std::endl;
        
        //c10::Profile2::set_enabled(true);
        //c10::Profile4::set_enabled(true);
        for (auto& batch : *data_loader){
            //std::cout<<"enter"<<std::endl;
            /**
            if(iter==4){
                c10::Profile::set_enabled(true);
            }**/
            //std::cout<<batch.data.sizes()<<std::endl;
            //std::cout<<batch.target.sizes()<<std::endl;

            //if(iter==2){
            //    c10::Profile::set_enabled(true);
            //}
            //if(iter==5){
            //    c10::LifeProfile::set_enabled(true);
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
            //std::cout<<"s"<<std::endl;
            loss.backward();
            cudaDeviceSynchronize();
            gettimeofday(&end_, NULL );
            //const  c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
            //std::cout<<(stats.allocated_bytes)[0].current<<"-"<<std::endl;
            int timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
            std::cout<<timeuse<<std::endl;

            if(c10::BreakdownProfile::is_enabled()){
                c10::GetBreakdownProfile()->finished();
            }
            if(c10::ProfilerSwitch::is_enabled()){
                c10::GetProfilerImpl()->add_total_time(timeuse);
                c10::GetProfilerImpl()->finish_iteration();
            }
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
            if(c10::Profile3::is_enabled()){
                std::cout<<"max_mem:"<<c10::GetCompressProfile()->get_max_mem()<<std::endl;
                c10::GetCompressProfile()->to_txt2("/nfs/home/schen/dcgan/build/vgg16-cifar10/memory_load1.txt");
                c10::GetCompressProfile()->finish_iteration2();
            }**/
            //const  c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
            //std::cout<<(stats.allocated_bytes)[0].current<<std::endl;
            //iter++;
            //if(iter==3)
            //    break;
            admm.step();
            /**
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
                    for(int i=0;i<43;i++){
                        net.policy[i] = 0;
                    }
                }
                if(policy_==4){
                    std::cout<<"+,"<<std::endl;
                    is_super = true;
                    for(int i=0;i<43;i++){
                        net.policy[i] = 0;
                    }
                }
                //c10::BreakdownProfile::set_enabled(true);
            }**/
            
            //iter++;
           
        }
        
    }
    
    if(c10::ProfilerSwitch::is_enabled()){
        c10::GetProfilerImpl()->to_txt("result/vgg16_cifar10_32.txt");
    }

    return 0;
}
