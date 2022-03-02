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


// Global define  =====================================================
int *cpuValueIndex=nullptr;
int *beginIndex=nullptr;
// 6096
int dimSize = 256;
int blocksize = 32;
int process = dimSize*blocksize;

void log_transfer_time(torch::Tensor& x){
    /**
    torch::Device device1(torch::kCUDA);
    torch::Device device2(torch::kCPU);
    FILE *f = fopen("transfer_time.txt","a+");
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
    **/
    FILE* f = fopen("vgg16-cifar10-2/mem.txt","a+");
    fprintf(f,"%zu\n",x.nbytes());
    fclose(f);
}
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
        policy.resize(45,0);
    }
    

    void set_policy(int index,int po){
        policy[index] = po;
    }
    
    // Implement the Net's algorithm.
    torch::Tensor forward(torch::Tensor x,torch::Tensor target) 
    {
        struct timeval start_, end_;
        struct timeval start_1, end_1;
        int timeuse;
        int index;
        torch::Tensor pre=x;
        size_t sizes;
        
        at::DataPtr dataptr_;
        void* index_begin = at::cuda::getCUDADeviceAllocator()->raw_allocate(process*sizeof(int));
       
        void* num = at::cuda::getCUDADeviceAllocator()->raw_allocate(sizeof(int));
        // serial needed parameter
        void* num_ptr=nullptr;
        cudaMallocHost(&num_ptr,sizeof(int));
        
        
        void* compress_value=nullptr;
        void* index_ = nullptr; 
        at::GetTransferRun()->init();
        
        //std::cout<<"enter"<<std::endl;
        for(index =0;index<v.size();index++){
            if(index>0&&policy[index-1]==2){
                if(c10::Passive::is_enabled()){
                    if(c10::Profile2::is_enabled()||c10::Profile4::is_enabled()){
                        gettimeofday(&start_1, NULL);
                    }
                    sizes = pre.nbytes();
                    dataptr_ = at::detail::getCUDAHooks().getPinnedMemoryAllocator()->allocate(sizes);
                    cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
                    if(c10::Profile4::is_enabled()){
                        c10::GetCompressProfile()->add_tensor_size(sizes);
                    }
                    cudaMemcpyAsync(dataptr_.get(),pre.data_ptr(),sizes,cudaMemcpyDeviceToHost,stream1);
                    if(c10::Profile4::is_enabled()){
                        gettimeofday(&start_, NULL);
                    }
                }              
            }
            if(c10::Profile::is_enabled()){
                cudaDeviceSynchronize();
                gettimeofday(&start_, NULL);
            }
            // 执行
            x = v[index](pre);

            const c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
            // std::cout<<"Layer "<<index<<", Tensor size is: "<<long(x.nbytes())/(1024*1024)<<" MB, Memory usage is: "<<long((stats.allocated_bytes)[0].current)/(1024*1024)<<" MB, Size without transfer "<<std::endl;
            
            if(c10::Profile::is_enabled()){
                cudaDeviceSynchronize();
                gettimeofday(&end_, NULL);
                timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
                c10::GetPolicyMaker()->add_comptime(timeuse);
                x.grad_fn()->id_ = index;
                x.grad_fn()->ids.push_back(index-1);
                c10::GetPolicyMaker()->add_tensor(index,x.nbytes());
                c10::GetPolicyMaker()->add_stepdep({index-1,index});

            }else{
                cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
                
                if(c10::Profile3::is_enabled()){
                    const  c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
                    c10::GetCompressProfile()->add_memory_load((stats.allocated_bytes)[0].current);
                }
            }
            
            if(index>0&&policy[index-1]==1){
                pre.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->data_ptr().clear();
                pre.grad_fn()->storage_impl_ = pre.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
                pre.grad_fn()->com_and_tran = 1;
            }else if(index>0&&policy[index-1]==2){
                if(c10::Passive::is_enabled()){
                    if(c10::Profile4::is_enabled()){
                        gettimeofday(&end_, NULL);
                        timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
                        c10::GetCompressProfile()->add_compute_time(timeuse);
                    }
                    cudaStreamSynchronize(stream1);
                    pre.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->set_data_ptr(std::move(dataptr_));
                    pre.grad_fn()->storage_impl_ = pre.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
                    pre.grad_fn()->com_and_tran = 2;
                    
                    if(c10::Profile2::is_enabled()||c10::Profile4::is_enabled()){
                        gettimeofday(&end_1, NULL);
                        timeuse = 1000000 * ( end_1.tv_sec - start_1.tv_sec ) + end_1.tv_usec - start_1.tv_usec;
                        c10::GetCompressProfile()->add_transfer_time(timeuse);
                    }
                }else{                
                    pre.grad_fn()->com_and_tran = 2;
                    pre.grad_fn()->storage_impl_ = pre.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
                    at::GetTransferRun()->add_item(pre.nbytes(),pre.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl());
                }
            }
            x.grad_fn()->pre_node = pre.grad_fn();    
            x.grad_fn()->func_ = v[index];
            x.grad_fn()->input_tensor = pre.variable_data();
            pre = x;   
        }
      
        if(c10::Profile::is_enabled()){
            cudaDeviceSynchronize();
            gettimeofday(&start_, NULL);
        }
        x = pre.view({ pre.size(0), -1 });
        //f_size +=x.nbytes();
        
        if(c10::Profile::is_enabled()){
            cudaDeviceSynchronize();
            gettimeofday(&end_, NULL);
            timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
            c10::GetPolicyMaker()->add_comptime(timeuse);
            x.grad_fn()->id_ = index;
            x.grad_fn()->ids.push_back(index-1);
            c10::GetPolicyMaker()->add_tensor(index,x.nbytes());
            c10::GetPolicyMaker()->add_stepdep({index-1,index});
        }else{
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            if(c10::Profile3::is_enabled()){
                const  c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
                c10::GetCompressProfile()->add_memory_load((stats.allocated_bytes)[0].current);
            }

        }
        index++;
        pre = x;
        
        if(c10::Profile::is_enabled()){
            cudaDeviceSynchronize();
            gettimeofday(&start_, NULL);
        }
        x = fc1->forward(pre);
        //f_size +=x.nbytes();
        
        if(c10::Profile::is_enabled()){
            cudaDeviceSynchronize();
            gettimeofday(&end_, NULL);
            timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
            c10::GetPolicyMaker()->add_comptime(timeuse);
            x.grad_fn()->id_ = index;
            x.grad_fn()->ids.push_back(index-1);
            c10::GetPolicyMaker()->add_tensor(index,x.nbytes());
            c10::GetPolicyMaker()->add_stepdep({index-1,index});
        }else{
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
             if(c10::Profile3::is_enabled()){
                const  c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
                c10::GetCompressProfile()->add_memory_load((stats.allocated_bytes)[0].current);
            }
        }
        index++;
        pre = x;
        if(c10::Profile::is_enabled()){
            cudaDeviceSynchronize();
            gettimeofday(&start_, NULL);
        }
        x = loss->forward(pre,target);
        //f_size +=x.nbytes();
        
        if(c10::Profile::is_enabled()){
            cudaDeviceSynchronize();
            gettimeofday(&end_, NULL);
            timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
            c10::GetPolicyMaker()->add_comptime(timeuse);
            x.grad_fn()->id_ = index;
            x.grad_fn()->ids.push_back(index-1);
            x.grad_fn()->ids.push_back(index);
            c10::GetPolicyMaker()->add_tensor(index,x.nbytes());
            c10::GetPolicyMaker()->add_stepdep({index-1,index});
        }else{
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            if(c10::Profile3::is_enabled()){
                const  c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
                c10::GetCompressProfile()->add_memory_load((stats.allocated_bytes)[0].current);
            }
        }
        index++;
        if(c10::Profile::is_enabled()){
            c10::GetPolicyMaker()->set_num(index);
        }
        at::GetTransferRun()->finish_forward();
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
    c10::Passive::set_enabled(true);
    //c10::Profile2::set_enabled(true);
    int batch_size = 128;
    int zjlab = 1;
    int policy_ = 1;
    double ratio = 100;
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
    torch::Device device(device_type);
    Net net;
    net.to(device);
    torch::optim::Adam admm(net.parameters(),torch::optim::AdamOptions(0.001));
    int iter;
    // parallel needed parameter
    cudaMallocHost(&cpuValueIndex, sizeof(int) * process);
    cudaMallocHost(&beginIndex, sizeof(int) * process);    
    
    for(int i=0;i<43;i++){
        net.policy[i] = 2;
    }

    std::vector<int> policy1,policy2;
    iter = 0;
    for(int i=0;i<5;i++)
    {
        for (auto& batch : *data_loader){
            if(iter==1){
                c10::Profile::set_enabled(true);
            }
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);
            struct timeval start_, end_;
            cudaDeviceSynchronize();
            gettimeofday(&start_, NULL );
            auto loss = net.forward(data,target);
            //std::cout<<loss<<std::endl;
            const c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
            // 记录当前网络内存最大值
            if(iter==1){
                // std::cout<<"Max memory usage is: "<<(stats.allocated_bytes)[0].current<<std::endl;
                c10::GetPolicyMaker()->add_remain_mem(double((stats.allocated_bytes)[0].current));
            }
            loss.backward();
            cudaDeviceSynchronize();
            gettimeofday(&end_, NULL );
            int timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
            std::cout<<timeuse<<std::endl;
            admm.step();
            if(iter==1)
            {
                int time_;
                c10::Profile::set_enabled(false);
                c10::Passive::set_enabled(false);
                c10::GetPolicyMaker()->init(ratio);
                if(policy_==1){
                    policy1 = c10::GetPolicyMaker()->make_policy(time_);
                    std::cout<<"iteration_time:"<<time_<<std::endl;
                    for(int po:policy1){
                        std::cout<<po<<",";
                    }
                    if(policy1.size()==0) return 0;
                }
                for(int po:policy1){
                   std::cout<<po<<",";
                }
                std::cout<<std::endl;
                if(policy_==2){
                    policy2 = c10::GetPolicyMaker()->capuchin_new(time_);
                    // policy2 = c10::GetPolicyMaker()->capuchin(time_);
                    std::cout<<"capuchin_time:"<<time_<<std::endl;
                    for(int po:policy2){
                        std::cout<<po<<",";
                    }
                    if(policy2.size()==0) return 0;
                }
                
                if(policy_==1&&policy1.size()){
                    net.policy = policy1;
                }
                if(policy2.size()&&policy_==2){
                    net.policy = policy2;
                }

            }
            iter++;
        }
        
    }
    return 0;
}
