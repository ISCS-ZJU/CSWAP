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
extern "C"{
void PreGPUcompression(int arraySize, int kernelSize, float* arrayGPU, int* valueIndex,int dimsize,int blocksize,cudaStream_t stream);
void PreAcc(int* valueIndex,int KernelSize,int* num,int dimsize,int blocksize,cudaStream_t stream);
void GPUcompression(int arraySize, int kernelSize, float* arrayGPU, float* compressedList, int* compressedValueIndex, uint32_t* compressedBinIndex,int dimsize,int blocksize,cudaStream_t stream);
}**/

/**
void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, void *data) {
    printf("Inside callback %d\n", (size_t)data);
}**/
//using namespace torch::autograd;
//using namespace c10;
/**
void run();
void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, void *data){
    c10::StorageImpl* storage = torch::autograd::GetTransferRun()->storages.front();
    std::shared_ptr<torch::autograd::Node> node = torch::autograd::GetTransferRun()->nodes.front();
    storage->set_data_ptr(std::move(torch::autograd::GetTransferRun()->dataptr_));
    node->storage_impl_ = storage;
    node->com_and_tran = 2;
    torch::autograd::GetTransferRun()->sizes.pop();
    torch::autograd::GetTransferRun()->storages.pop();
    torch::autograd::GetTransferRun()->nodes.pop();
    torch::autograd::GetTransferRun()->flag = false;
    if(!torch::autograd::GetTransferRun()->sizes.empty()){
        std::thread t(run);
    }
}
void run(){
    c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
    size_t size= torch::autograd::GetTransferRun()->sizes.front();
    torch::autograd::GetTransferRun()->dataptr_ = at::detail::getCUDAHooks().getPinnedMemoryAllocator()->allocate(size);
    cudaMemcpyAsync(torch::autograd::GetTransferRun()->dataptr_.get(),torch::autograd::GetTransferRun()->storages.front()->data(),size,cudaMemcpyDeviceToHost,stream);
    cudaStreamAddCallback(stream, MyCallback, nullptr, 0);
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



bool is_super = false;
bool is_vdnn = false;
Execution exe;
struct Net : torch::nn::Module {
    Net() {
        convs[0] = torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 16, { 3,3 }).padding(1));
        convs[1] = torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 16, { 3,3 }).padding(1));
        convs[2] = torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 16, { 3,3 }).padding(1));
        convs[3] = torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 16, { 3,3 }).padding(1));
        convs[4]= torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 16, { 3,3 }).padding(1));
        convs[5] = torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 16, { 3,3 }).padding(1));
        convs[6] = torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 16, { 3,3 }).padding(1));
        convs[7]= torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, { 3,3 }).padding(1));
        convs[8] = torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 32, { 3,3 }).padding(1));
        convs[9] = torch::nn::Conv2d(torch::nn::Conv2dOptions(32,32, { 3,3 }).padding(1));
        convs[10] = torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 32, { 3,3 }).padding(1));
        convs[11] = torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 32, { 3,3 }).padding(1));
        convs[12] = torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 32, { 3,3 }).padding(1));
        convs[13]= torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, { 3,3 }).padding(1));
        convs[14] = torch::nn::Conv2d(torch::nn::Conv2dOptions(64,64, { 3,3 }).padding(1));
        convs[15] = torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, { 3,3 }).padding(1));
        convs[16] = torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, { 3,3 }).padding(1));
        convs[17]= torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, { 3,3 }).padding(1));
        convs[18]= torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, { 3,3 }).padding(1));

        avg_pool = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(8).stride(1));
        fc1 = torch::nn::Linear(40000,10);
        loss = torch::nn::CrossEntropyLoss();


        batchnorm[0] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(16));
        batchnorm[1] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(16));
        batchnorm[2] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(16));
        batchnorm[3] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(16));
        batchnorm[4] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(16));
        batchnorm[5] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(16));
        batchnorm[6] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(16));
        batchnorm[7] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32));
        batchnorm[8] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32));
        batchnorm[9] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32));
        batchnorm[10] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32));
        batchnorm[11] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32));
        batchnorm[12] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32));
        batchnorm[13] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64));
        batchnorm[14] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64));
        batchnorm[15] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64));
        batchnorm[16] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64));
        batchnorm[17] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64));
        batchnorm[18] = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64));
        for(int i=0;i<19;i++){
            register_module("batchnorm_"+std::to_string(i),batchnorm[i]);
            register_module("conv_"+std::to_string(i),convs[i]);
        }
        register_module("fc1", fc1);
        register_module("crossloss", loss);

        for(int i=0;i<19;i++){
            v.push_back(std::bind(&torch::nn::Conv2dImpl::forward,convs[i],std::placeholders::_1));
            types.push_back(CONV);
            v.push_back(std::bind(&torch::nn::BatchNorm2dImpl::forward,batchnorm[i],std::placeholders::_1));
            types.push_back(BN);
            v.push_back(torch::relu);
            types.push_back(ACT);
        }
        v.push_back(std::bind(&torch::nn::AvgPool2dImpl::forward,avg_pool,std::placeholders::_1));
        types.push_back(POOL);
        policy.resize(58,0);

        
        

    }
    

    void set_policy(int index,int po){
        policy[index] = po;
    }
   
    // Implement the Net's algorithm.
    torch::Tensor forward(torch::Tensor x,torch::Tensor target) {
        //FILE *f = fopen("vgg16-cifar10-compression/compression_time.txt","a+");
        //FILE *f1 = fopen("vgg16-cifar10-compression/transfer_time_.txt","a+");
        
        exe.preforward();
        torch::Tensor pre=x;
        for(int index =0;index<v.size();index++){
            exe.pre(pre,ACT);
            x = v[index](pre);
            exe.post(pre,x,ACT);
            x.grad_fn()->func_ = v[index];
            pre = x;   
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
    torch::nn::Conv2d convs[19]={nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
    nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr};
    torch::nn::BatchNorm2d batchnorm[19]{nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
    nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr};

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
    //void* ptr=nullptr;
    //cudaMallocHost(&ptr,4096);

    //c10::Allocator* allo = c10::GetDefaultCPUAllocator();
    //c10::Profile::set_enabled(true);
    //c10::PolicyMaker* pm= c10::GetPolicyMaker();

    //c10::PolicyMaker pm;
    //torch::pickle_save()
    //torch::autograd::saved_variable_list()
    //at::GradMode::set_enabled(true);
    
    //at::TransferRun* rn = at::GetTransferRun();

    //c10::Passive::set_enabled(true);
    //c10::Profile2::set_enabled(true);
    int batch_size = 1024;
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

    if(torch::jit::tracer::isTracing()){
        std::cout<<"trace"<<std::endl;
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
    torch::Device device(device_type);

    Net net;
    net.to(device);

    torch::optim::Adam admm(net.parameters(),torch::optim::AdamOptions(0.001));

    std::vector<int> policy1,policy2;

    //bool is_compress = false;

    int iter = 0;
    exe.init_(100);
    for(int i=0;i<100;i++){
        exe.begin_stage[i] = i+1;
    }
    //for(int i=0;i<18;i++){
    //    net.policy[i*3+2] = 2;
    //}
    //for(int j=0;j<18;j++){
    //    net.policy[j*3+2] = 3;
    //}
    //for(int j=0;j<14;j++){
    //    net.policy[j*3+2] = 3;
    //}
    //for(int j=14;j<18;j++){
    //    net.policy[j*3+2] = 2;
    //}
    //is_compress = true;

    for(int j=0;j<18;j++){
        exe.policy[j*3+2] = 6;
    }

    iter = 0;
    for(int i=0;i<10;i++){
        //std::cout<<i<<std::endl;
        
        //c10::Profile4::set_enabled(true);
        for (auto& batch : *data_loader){
            if(iter==2){
                c10::Profile::set_enabled(true);
            }
            //std::cout<<"enter"<<std::endl;
            /**
            if(iter==3){
                c10::Profile::set_enabled(true);
            }**/
            //std::cout<<batch.data.sizes()<<std::endl;
            //std::cout<<batch.target.sizes()<<std::endl;
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);
            struct timeval start_, end_;
            //std::cout<<"enter"<<std::endl;
            cudaDeviceSynchronize();
            //std::cout<<"enter"<<std::endl;
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
            if(c10::BreakdownProfile::is_enabled()){
                c10::GetBreakdownProfile()->finished();
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
            //iter++;
            //if(iter==3)
            //    break;
            if(c10::ProfilerSwitch::is_enabled()){
                c10::GetProfilerImpl()->add_total_time(timeuse);
                c10::GetProfilerImpl()->finish_iteration();
            }
            admm.step();
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
                    for(int i=0;i<56;i++){
                        net.policy[i] = 0;
                    }
                }
                if(policy_==4){
                    std::cout<<"+,"<<std::endl;
                    is_super = true;
                    for(int i=0;i<56;i++){
                        net.policy[i] = 0;
                    }
                }
                //c10::BreakdownProfile::set_enabled(true);
            
            }
            //iter++;
            //if(iter==20){
                //net.policy[2] = 3,net.policy[6] = 3,net.policy[9] = 3,net.policy[13] = 3,net.policy[16] = 3,net.policy[19] = 3
                //,net.policy[23] = 3,net.policy[26] = 3,net.policy[29] = 3,net.policy[33] = 3,net.policy[36] = 3,net.policy[39] = 3;
            //}
            //iter++;
            /**
            if(iter==3){
                //c10::Profile2::set_enabled(false);
                c10::Profile4::set_enabled(false);
            }**/
        }
        /**
        if(i==99){
            for(int j=0;j<18;j++){
                net.policy[j*3+2] = 3;
            }
            is_compress = true;
        }**/
    }
    //c10::GetCompressProfile()->to_txt("/nfs/home/schen/dcgan/build/compress2/2560/plain20-cifar10_1.txt");
    //c10::GetCompressProfile()->to_txt4("/nfs/home/schen/dcgan/build/compress4/2560/plain20-cifar10_.txt");

    if(c10::ProfilerSwitch::is_enabled()){
        c10::GetProfilerImpl()->to_txt("result/plain20_cifar10_32.txt");
    }

    return 0;
}
