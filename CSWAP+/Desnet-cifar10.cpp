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


enum Layer_type{
    CONV,
    POOL,
    ACT,
    BN,
    FC
};

bool is_super = false;
bool is_vdnn = false;


class Execution{
public:
    Execution(){ 
    }
    void init(){
        index = -1;
    }
    void init_(){
        policy.resize(360,0);
        begin_stage.resize(360,0);      
    }
    void pre(torch::Tensor& pre,Layer_type layer_type){
        index++;
        //std::cout<<index<<std::endl;
        if((pre.tensor_id>=0)&&(index==begin_stage[pre.tensor_id])){
            if((policy[pre.tensor_id]==2)||(is_super&&(layer_type==CONV))||is_vdnn){
                if(c10::Passive::is_enabled()){
                    if(c10::Profile4::is_enabled()){
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
        }
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&start_, NULL);
        }
    }

    void pre(vector<torch::Tensor>& inputs){
        index++;
        int length = inputs.size();
        dataptrs.clear();
        stream.clear();
        start_s.clear();
        start_1s.clear();
        end_1s.clear();
        end_s.clear();
        dataptrs.resize(length);
        stream.resize(length);
        start_s.resize(length);
        start_1s.resize(length);
        end_s.resize(length);
        end_1s.resize(length);
        for(int j=0;j<length;j++){
            if((inputs[j].tensor_id>=0)&&(index==begin_stage[inputs[j].tensor_id])){
                if(policy[inputs[j].tensor_id]==2){
                    if(c10::Passive::is_enabled()){
                        if(c10::Profile4::is_enabled()){
                            gettimeofday(&start_1s[j], NULL); 
                        }
                        sizes = inputs[j].nbytes();
                        dataptrs[j] = at::detail::getCUDAHooks().getPinnedMemoryAllocator()->allocate(sizes);
                        cudaStreamCreateWithFlags(&stream[j],cudaStreamNonBlocking);
                        if(c10::Profile4::is_enabled()){
                            c10::GetCompressProfile()->add_tensor_size(sizes);
                        }
                        cudaMemcpyAsync(dataptrs[j].get(),inputs[j].data_ptr(),sizes,cudaMemcpyDeviceToHost,stream[j]);
                        if(c10::Profile4::is_enabled()){
                            gettimeofday(&start_s[j], NULL);
                        }
                    }            
                }
            }
        }
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&start_, NULL);
        }
    }




    void post(torch::Tensor& pre,torch::Tensor& x,Layer_type layer_type){   
        //std::cout<<pre.tensor_id<<std::endl;   
        x.tensor_id = index;
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&end_, NULL);
            timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
            c10::GetPolicyMaker()->add_comptime(timeuse);
            x.grad_fn()->id_ = index;
            x.grad_fn()->ids.push_back(pre.tensor_id);
            //x.grad_fn()->ids.push_back(index);
            c10::GetPolicyMaker()->add_tensor(index,x.nbytes());
            c10::GetPolicyMaker()->add_stepdep({pre.tensor_id,index});
        }else{
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            if(c10::Profile3::is_enabled()){
                const  c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
                c10::GetCompressProfile()->add_memory_load((stats.allocated_bytes)[0].current);
            }
        }
        if((pre.tensor_id>=0)&&((policy[pre.tensor_id]==1)||(is_super&&(layer_type!=CONV)))&&(index==begin_stage[pre.tensor_id])){
            pre.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->data_ptr().clear();
            pre.grad_fn()->storage_impl_ = pre.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
            pre.grad_fn()->com_and_tran = 1;
                    //x.grad_fn()->is_recomp = true;     
        }else if((pre.tensor_id>=0)&&((policy[pre.tensor_id]==2)||(is_super&&(layer_type==CONV))||is_vdnn)&&(index==begin_stage[pre.tensor_id])){
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
                //std::cout<<"shout;";    
                if(c10::Profile4::is_enabled()){
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
        x.grad_fn()->input_tensor = pre.variable_data();
    }
    void post(vector<torch::Tensor>& inputs,torch::Tensor& x){
        int length = inputs.size();
        x.tensor_id = index;
        if(c10::Profile::is_enabled()){
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            gettimeofday(&end_, NULL);
            timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
            c10::GetPolicyMaker()->add_comptime(timeuse);
            x.grad_fn()->id_ = index;
            for(int j=0;j<length;j++){
                x.grad_fn()->ids.push_back(inputs[j].tensor_id);
            }
            c10::GetPolicyMaker()->add_tensor(index,x.nbytes());
            vector<int> ids{index};
            for(int j=0;j<length;j++){
                ids.push_back(inputs[j].tensor_id);
            }
            c10::GetPolicyMaker()->add_stepdep(ids);
        }else{
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            if(c10::Profile3::is_enabled()){
                const  c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
                c10::GetCompressProfile()->add_memory_load((stats.allocated_bytes)[0].current);
                        //std::cout<<(stats.allocated_bytes)[0].current<<"-";
            }
        }
        x.grad_fn()->is_cat = true;
        x.grad_fn()->multiple_input.input_tensors.resize(length);
        x.grad_fn()->multiple_input.pre_nodes.resize(length,nullptr);

        for(int j=0;j<length;j++){
            if((inputs[j].tensor_id>=0)&&(policy[inputs[j].tensor_id]==1)&&(index==begin_stage[inputs[j].tensor_id])){
                inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->data_ptr().clear();
                inputs[j].grad_fn()->storage_impl_ = inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
                inputs[j].grad_fn()->com_and_tran = 1;
            }else if((inputs[j].tensor_id>=0)&&(policy[inputs[j].tensor_id]==2)&&(index==begin_stage[inputs[j].tensor_id])){
                if(c10::Passive::is_enabled()){
                    if(c10::Profile4::is_enabled()){
                        gettimeofday(&end_s[j], NULL);
                        timeuse = 1000000 * ( end_s[j].tv_sec - start_s[j].tv_sec ) + end_s[j].tv_usec - start_s[j].tv_usec;
                        c10::GetCompressProfile()->add_compute_time(timeuse);
                    }
                    //cudaDeviceSynchronize();
                    cudaStreamSynchronize(stream[j]);
                    inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->set_data_ptr(std::move(dataptrs[j]));   
                    if(c10::Profile4::is_enabled()){ 
                        gettimeofday(&end_1s[j], NULL);
                        timeuse = 1000000 * ( end_1s[j].tv_sec - start_1s[j].tv_sec ) + end_1s[j].tv_usec - start_1s[j].tv_usec;
                        c10::GetCompressProfile()->add_transfer_time(timeuse);                       
                    }
                }else{
                    at::GetTransferRun()->add_item(inputs[j].nbytes(),inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl());
                }
                inputs[j].grad_fn()->com_and_tran = 2;
                inputs[j].grad_fn()->storage_impl_ = inputs[j].unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();

            }
            x.grad_fn()->multiple_input.input_tensors[j] = inputs[j].variable_data();
            x.grad_fn()->multiple_input.pre_nodes[j] = inputs[j].grad_fn();
        }
    }
public:
    int index;
    vector<int> policy;
    vector<int> begin_stage;

private:
    struct timeval start_, end_;
    struct timeval start_1, end_1;

    vector<struct timeval> start_s,end_s,start_1s,end_1s;

    int timeuse;
    size_t sizes;
    cudaStream_t stream1;
    at::DataPtr dataptr_;
    vector<cudaStream_t> stream;
    vector<c10::DataPtr> dataptrs;
};

Execution ex;

struct DenseLayer : torch::nn::Module{
    DenseLayer(int in_channels,int grow_rate,int bn_size){
        bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(in_channels));
        bn2 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(bn_size*grow_rate));
        conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels,bn_size*grow_rate,1).stride(1).bias(false));
        conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(bn_size*grow_rate,grow_rate,3).stride(1).padding(1).bias(false));
        register_module("conv1",conv1);
        register_module("conv2",conv2);
        register_module("batchnorm1",bn1);
        register_module("batchnorm2",bn2);
    }
    DenseLayer(){}

    torch::Tensor forward(torch::Tensor& x){
        torch::Tensor input = x;
        torch::Tensor pre = x;
        ex.pre(pre,BN);
        x = bn1->forward(pre);
        ex.post(pre,x,BN);
        x.grad_fn()->func_ = std::bind(&torch::nn::BatchNorm2dImpl::forward,bn1,std::placeholders::_1);
        pre = x;
        ex.pre(pre,ACT);
        x = torch::relu(pre);
        ex.post(pre,x,ACT);
        x.grad_fn()->func_ = torch::relu;
        pre = x;
        ex.pre(pre,CONV);
        x = conv1->forward(pre);
        ex.post(pre,x,CONV);
        x.grad_fn()->func_ = std::bind(&torch::nn::Conv2dImpl::forward,conv1,std::placeholders::_1);
        pre = x;
        ex.pre(pre,BN);
        x = bn2->forward(pre);
        ex.post(pre,x,BN);
        x.grad_fn()->func_ = std::bind(&torch::nn::BatchNorm2dImpl::forward,bn2,std::placeholders::_1);
        pre = x;
        ex.pre(pre,ACT);
        x = torch::relu(pre);
        ex.post(pre,x,ACT);
        x.grad_fn()->func_ = torch::relu;
        pre = x;
        ex.pre(pre,CONV);
        x = conv2->forward(pre);
        ex.post(pre,x,CONV);
        x.grad_fn()->func_ = std::bind(&torch::nn::Conv2dImpl::forward,conv2,std::placeholders::_1);

        pre = x;
        vector<torch::Tensor> inputs{input,pre};
        ex.pre(inputs);
        x = torch::cat({input,pre},1);
        ex.post(inputs,x);
        return x;
    }
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr};
    torch::nn::BatchNorm2d bn2{nullptr};
};

struct DenseBlock: torch::nn::Module{
    DenseBlock(int num_layers,int in_channels,int bn_size,int grow_rate){
        DenseLayers.resize(num_layers);
        for(int i=0;i<num_layers;i++){
            DenseLayers[i] = DenseLayer(in_channels+grow_rate*i,grow_rate,bn_size);
            register_module("denselayer"+to_string(i),std::make_shared<DenseLayer>(DenseLayers[i]));
        }
    }
    DenseBlock(){}

    torch::Tensor forward(torch::Tensor& x){
        for(int i=0;i<DenseLayers.size();i++){
            x = DenseLayers[i].forward(x);
        }
        return x;
    }
    vector<DenseLayer> DenseLayers;
};

struct Transition: torch::nn::Module{

    Transition(int in_channels,int out_channels){
        bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(in_channels));
        conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels,out_channels,1).stride(1).bias(false));
        avgpool = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(2).stride(2));
        register_module("conv",conv);
        register_module("batchnorm",bn);
    }
    Transition(){}
    torch::Tensor forward(torch::Tensor& x){
        torch::Tensor pre = x;
        ex.pre(pre,BN);
        x = bn->forward(pre);
        ex.post(pre,x,BN);
        x.grad_fn()->func_ = std::bind(&torch::nn::BatchNorm2dImpl::forward,bn,std::placeholders::_1);
        pre = x;
        ex.pre(pre,ACT);
        x = torch::relu(pre);
        ex.post(pre,x,ACT);
        x.grad_fn()->func_ = torch::relu;
        pre = x;
        ex.pre(pre,CONV);
        x = conv->forward(pre);
        ex.post(pre,x,CONV);
        x.grad_fn()->func_ = std::bind(&torch::nn::Conv2dImpl::forward,conv,std::placeholders::_1);
        pre = x;
        ex.pre(pre,POOL);
        x = avgpool->forward(pre);
        ex.post(pre,x,POOL);
        x.grad_fn()->func_ = std::bind(&torch::nn::AvgPool2dImpl::forward,avgpool,std::placeholders::_1);
        return x;
    }


    torch::nn::BatchNorm2d bn{nullptr};
    torch::nn::Conv2d conv{nullptr};
    torch::nn::AvgPool2d avgpool{nullptr};
};

struct Net: torch::nn::Module{
    Net(vector<int> block_config,int grow_rate=12,int bn_size=4,double theta=0.5,int num_classes=10){
        int num_init_features = 2*grow_rate;
        conv  = torch::nn::Conv2d(torch::nn::Conv2dOptions(3,num_init_features,3).stride(1).padding(1).bias(false));
        register_module("conv",conv);
        int num_features = num_init_features;
        denseblocks.resize(block_config.size());
        transitions.resize(block_config.size()-1);
        for(int i=0;i<block_config.size();i++){
            denseblocks[i] = DenseBlock(block_config[i],num_features,bn_size,grow_rate);
            register_module("denseblock"+to_string(i),std::make_shared<DenseBlock>(denseblocks[i]));
            num_features = num_features+grow_rate*block_config[i];
            if(i!=block_config.size()-1){
                transitions[i] = Transition(num_features,int(num_features*theta));
                register_module("transition"+to_string(i),std::make_shared<Transition>(transitions[i]));
                num_features = int(num_features*theta);
            }
        }
        bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(num_features));
        register_module("bn",bn);
        avgpool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1,1}));
        linear = torch::nn::Linear(num_features,10);
        register_module("linear",linear);
        loss = torch::nn::CrossEntropyLoss();
    }
    Net(){}

    torch::Tensor forward(torch::Tensor x,torch::Tensor target){
        struct timeval start_, end_;
        at::GetTransferRun()->init();
        ex.init();
        torch::Tensor pre = x;
        pre.tensor_id = -1;

        ex.pre(pre,CONV);
        x = conv->forward(pre);
        ex.post(pre,x,CONV);
        x.grad_fn()->func_ = std::bind(&torch::nn::Conv2dImpl::forward,conv,std::placeholders::_1);
        for(int i=0;i<denseblocks.size();i++){
            x = denseblocks[i].forward(x);
            if(i!=denseblocks.size()-1){
                x = transitions[i].forward(x);
            }
        }
        pre = x;
        ex.pre(pre,BN);
        x = bn->forward(pre);
        ex.post(pre,x,BN);
        x.grad_fn()->func_ = std::bind(&torch::nn::BatchNorm2dImpl::forward,bn,std::placeholders::_1);

        pre = x;
        ex.pre(pre,ACT);
        x = torch::relu(pre);
        ex.post(pre,x,ACT);
        x.grad_fn()->func_= torch::relu;

        pre = x;
        ex.pre(pre,POOL);
        x = avgpool->forward(pre);
        ex.post(pre,x,POOL);
        x.grad_fn()->func_ = std::bind(&torch::nn::AdaptiveAvgPool2dImpl::forward,avgpool,std::placeholders::_1);


        ex.index++;
        if(c10::Profile::is_enabled()){
            cudaDeviceSynchronize();
            gettimeofday(&start_, NULL);
        }
        x = x.view({ x.size(0), -1 });
        if(c10::Profile::is_enabled()){
            cudaDeviceSynchronize();
            gettimeofday(&end_, NULL);
            int timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
            c10::GetPolicyMaker()->add_comptime(timeuse);
            x.grad_fn()->id_ = ex.index;
            x.grad_fn()->ids.push_back(ex.index-1);
            c10::GetPolicyMaker()->add_tensor(ex.index,x.nbytes());
            c10::GetPolicyMaker()->add_stepdep({ex.index-1,ex.index});       
        }else{
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            if(c10::Profile3::is_enabled()){
                const  c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
                c10::GetCompressProfile()->add_memory_load((stats.allocated_bytes)[0].current);
                //std::cout<<(stats.allocated_bytes)[0].current<<"-";
            }
        }
        
        ex.index++;
        if(c10::Profile::is_enabled()){
            cudaDeviceSynchronize();
            gettimeofday(&start_, NULL);
        }

        x = linear->forward(x);
        if(c10::Profile::is_enabled()){
            cudaDeviceSynchronize();
            gettimeofday(&end_, NULL);
            int timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
            c10::GetPolicyMaker()->add_comptime(timeuse);
            x.grad_fn()->id_ = ex.index;
            x.grad_fn()->ids.push_back(ex.index-1);
            x.grad_fn()->ids.push_back(ex.index);
            c10::GetPolicyMaker()->add_tensor(ex.index,x.nbytes());
            c10::GetPolicyMaker()->add_stepdep({ex.index-1,ex.index});

            
        }else{
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            if(c10::Profile3::is_enabled()){
                const  c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
                c10::GetCompressProfile()->add_memory_load((stats.allocated_bytes)[0].current);
                //std::cout<<(stats.allocated_bytes)[0].current<<"-";
            }
        }

        ex.index++;
        if(c10::Profile::is_enabled()){
            cudaDeviceSynchronize();
            gettimeofday(&start_, NULL);
        }

        x = loss->forward(x, target);
        if(c10::Profile::is_enabled()){
            cudaDeviceSynchronize();
            gettimeofday(&end_, NULL);
            int timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
            c10::GetPolicyMaker()->add_comptime(timeuse);
            x.grad_fn()->id_ = ex.index;
            x.grad_fn()->ids.push_back(ex.index-1);
            x.grad_fn()->ids.push_back(ex.index);
            c10::GetPolicyMaker()->add_tensor(ex.index,x.nbytes());
            c10::GetPolicyMaker()->add_stepdep({ex.index-1,ex.index});

            
        }else{
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            if(c10::Profile3::is_enabled()){
                const  c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
                c10::GetCompressProfile()->add_memory_load((stats.allocated_bytes)[0].current);
                //std::cout<<(stats.allocated_bytes)[0].current<<"-";
            }
        }

        ex.index++;
        if(c10::Profile::is_enabled()){
            c10::GetPolicyMaker()->set_num(ex.index);
        }
        //std::cout<<ex.index<<std::endl;
        at::GetTransferRun()->finish_forward();
        return x;
    }

    torch::nn::Conv2d conv{nullptr};
    torch::nn::BatchNorm2d bn{nullptr};
    torch::nn::AdaptiveAvgPool2d avgpool{nullptr};
    torch::nn::Linear linear{nullptr};
    torch::nn::CrossEntropyLoss loss{nullptr};
    
    vector<DenseBlock> denseblocks;
    vector<Transition> transitions;


};

struct Normalize : public torch::data::transforms::TensorTransform<> {
    Normalize(float mean, float stddev)
        : mean_(torch::tensor(mean)), stddev_(torch::tensor(stddev)) {}
    torch::Tensor operator()(torch::Tensor input) {
        input.resize_({3,36,36});
        return input.sub_(mean_).div_(stddev_);
    }
    torch::Tensor mean_, stddev_;
};

int main(int argc,char* argv[]){
    //c10::Profile3::set_enabled(true);
    c10::Passive::set_enabled(true);
    //c10::Profile2::set_enabled(true);
    int batch_size = 128;
    int zjlab = 1;
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


    //c10::Passive::set_enabled(true);

    
    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Training on GPU" << std::endl;
        device_type = torch::kCUDA;
    } else {
        std::cout << "Training on CPU" << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    vector<int> block_config{16,16,16};
    Net net(block_config);
    net.to(device);
    ex.init_();
    for(int i=0;i<340;i++){
        ex.policy[i] = 2;
        //ex.begin_stage[i] = i+1;
    }

    
    if(zjlab){
        c10::GetPolicyMaker()->read_lastvisit("/nfs/home/schen/dcgan/build/squeezenet-cifar10/lastvisit.txt",ex.begin_stage);
    }else{
        c10::GetPolicyMaker()->read_lastvisit("/home/pytorch/memory_mamagement/build/desnet-cifar10/lastvisit.txt",ex.begin_stage);
    }
    
    torch::optim::Adam admm(net.parameters(),torch::optim::AdamOptions(0.002));
    

    /**
    for(int i=0;i<80;i++){
        ex.policy[i] = 2;
    }**/

    int iter;

    std::vector<int> policy1,policy2;
    iter = 0;
    //for(int epoch = 0; epoch < 5; epoch++){      //epoch
        //std::cout << "epoch: " << epoch << std::endl;
        
        //c10::Profile4::set_enabled(true);
        for (auto& batch : *data_loader){
            
            if(iter==3){
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
            std::cout << timeuse<< std::endl;

            //const  c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
            //c10::GetCompressProfile()->add_memory_load((stats.allocated_bytes)[0].current);
            //std::cout<<(stats.allocated_bytes)[0].current<<"-";
            //c10::GetCompressProfile()->to_txt2("/home/pytorch/memory_mamagement/build/desnet_log/"+to_string(iter)+".log");
            //c10::GetCompressProfile()->finish_iteration2();

            /**
            if(c10::Profile4::is_enabled()){
                c10::GetCompressProfile()->finish_iteration();
                if(is_compress){
                    c10::GetCompressProfile()->add_total_time2(timeuse);
                }else{
                    c10::GetCompressProfile()->add_total_time1(timeuse);
                }
            }**/
            
            if(iter==3){
                int time_;
                c10::Profile::set_enabled(false);
                c10::Passive::set_enabled(false);
                c10::GetPolicyMaker()->init(ratio);
                //c10::GetPolicyMaker()->print();
                //c10::GetPolicyMaker()->write_lastvisit("/home/pytorch/memory_mamagement/build/desnet-cifar10/lastvisit.txt");
                
                //c10::GetPolicyMaker()->print_time();
                //auto policy = c10::GetPolicyMaker()->capuchin(time_);
                
                if(policy_==1){
                    policy1 = c10::GetPolicyMaker()->make_policy(time_);
                    std::cout<<"iteration_time:"<<time_<<std::endl;
                    for(int po:policy1){
                        std::cout<<po<<",";
                    }
                    if(policy1.size()==0) return 0;
                    ex.policy = policy1;
                }
                if(policy_==2){
                    policy2 = c10::GetPolicyMaker()->capuchin(time_);
                    std::cout<<"capuchin_time:"<<time_<<std::endl;
                    for(int po:policy2){
                        std::cout<<po<<",";
                    }
                    if(policy2.size()==0) return 0;
                    ex.policy = policy2;
                }
                if(policy_==3){
                    c10::Passive::set_enabled(true);
                    std::cout<<"+,"<<std::endl;
                    is_vdnn = true;
                    for(int i=0;i<340;i++){
                        ex.policy[i] = 0;
                    }
                }
                if(policy_==4){
                    std::cout<<"+,"<<std::endl;
                    is_super = true;
                    for(int i=0;i<340;i++){
                        ex.policy[i] = 0;
                    }
                }
            }
            iter++;

            if(iter==10) break;
        }
    //}
    return 0;
}