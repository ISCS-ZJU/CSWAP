#include <torch/torch.h>
#include<torch/csrc/autograd/generated/Functions.h>
#include<torch/csrc/autograd/function.h>
#include <ATen/cuda/CUDAContext.h>
#include<typeinfo>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>
#include<functional>

void log_transfer_time(torch::Tensor& x){
    torch::Device device1(torch::kCUDA);
    torch::Device device2(torch::kCPU);
    FILE *f = fopen("log_time.txt","a+");
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
}

struct Net : torch::nn::Module {
    Net()
        : conv1(torch::nn::Conv2dOptions(1, 20, /*kernel_size=*/5)),
          conv2(torch::nn::Conv2dOptions(20, 50, /*kernel_size=*/5)),
          fc1(4 * 4 * 50, 500),
          fc2(500, 10) {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }

    torch::Tensor forward(torch::Tensor x) {
        //std::cout<<"----****----"<<std::endl;
        //struct timeval start_, end_;
        //int timeuse;
        //gettimeofday(&start_, NULL );
        //x.name_ = "input";
       std::function<torch::Tensor(const torch::Tensor&)> func;


        torch::Tensor x_ = conv1->forward(x);
        
        


        torch::Tensor x__ = torch::max_pool2d(x_, 2);

        //auto func_ = std::bind(torch::max_pool2d,std::placeholders::_1,2);
        //func = func_;
        //x__.grad_fn()->func_ = func;
        //x__.grad_fn()->storage_impl_ = x_.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
        //x__.grad_fn()->tensor_impl = x_.unsafeGetTensorImpl();
        //x__.grad_fn()->pre_node = x_.grad_fn();




        torch::Tensor x1 = torch::relu(x__);
        //x__.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->data_ptr().clear();
        x1.grad_fn()->func_ = torch::relu;
        x1.grad_fn()->storage_impl_ = x__.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
        //x1.grad_fn()->tensor_impl = x__.unsafeGetTensorImpl();
        //auto ittt = x__.unsafeGetTensorImpl()->version_counter();
        x1.grad_fn()->input_tensor = x__.variable_data();
        //auto tensor_impl = x1.grad_fn()->tensor_impl;
        //auto tensor__ = at::Tensor(tensor_impl->shallow_copy_and_detach(
        //    0,tensor_impl->allow_tensor_metadata_change()
        //));
        //tensor__.set_requires_grad(false);
        //auto tensor_ = (x1.grad_fn()->func_)(tensor__);

        //std::cout<<x1.grad_fn()->tensor_impl<<std::endl;
        x1.grad_fn()->pre_node = x__.grad_fn();

        
        size_t sizes = x__.nbytes();
        at::DataPtr dataptr = x__.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->set_data_ptr(at::getCPUAllocator()->allocate(sizes));
        cudaMemcpy(x__.data_ptr(),dataptr.get(),sizes,cudaMemcpyDeviceToHost);
        x1.grad_fn()->storage_impl_ = x__.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
        x1.grad_fn()->com_and_tran = 2;
        //std::cout<<"---"<<std::endl;
        //dataptr.get_deleter()(dataptr.get());
        //std::cout<<"---"<<std::endl;
        //dataptr.clear();
        
        //dataptr = x.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->set_data_ptr(at::cuda::getCUDADeviceAllocator()->allocate(sizes));
        //std::cout<<"---"<<std::endl;
        //cudaMemcpy(x.data_ptr(),dataptr.get(),sizes,cudaMemcpyHostToDevice);
        torch::Tensor x2 = conv2->forward(x1);
        x1.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->data_ptr().clear();
        x2.grad_fn()->com_and_tran = 1;
        x2.grad_fn()->func_ = std::bind(&torch::nn::Conv2dImpl::forward,conv2,std::placeholders::_1);
        x2.grad_fn()->storage_impl_ = x1.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
        //x2.grad_fn()->tensor_impl = x1.unsafeGetTensorImpl();
        x2.grad_fn()->pre_node = x1.grad_fn();
        x2.grad_fn()->input_tensor = x1.variable_data();



        torch::Tensor x3 = torch::max_pool2d(x2, 2);
        x2.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->data_ptr().clear();
        x3.grad_fn()->com_and_tran = 1;
        x3.grad_fn()->storage_impl_ = x2.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
        //x2.grad_fn()->tensor_impl = x1.unsafeGetTensorImpl();
        x3.grad_fn()->pre_node = x2.grad_fn();
        x3.grad_fn()->input_tensor = x2.variable_data();

        /**
        sizes = x2.nbytes();
        dataptr = x2.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->set_data_ptr(at::getCPUAllocator()->allocate(sizes));
        cudaMemcpy(x2.data_ptr(),dataptr.get(),sizes,cudaMemcpyDeviceToHost);
        x3.grad_fn()->storage_impl_ = x2.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
        **/

        torch::Tensor x4 = torch::relu(x3);
        /**
        sizes = x3.nbytes();
        dataptr = x3.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->set_data_ptr(at::getCPUAllocator()->allocate(sizes));
        cudaMemcpy(x3.data_ptr(),dataptr.get(),sizes,cudaMemcpyDeviceToHost);
        x4.grad_fn()->storage_impl_ = x3.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
        **/
        x4 = x4.view({x4.size(0), -1});
        torch::Tensor x5 = fc1->forward(x4);
        /**
        sizes = x4.nbytes();
        dataptr = x4.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->set_data_ptr(at::getCPUAllocator()->allocate(sizes));
        cudaMemcpy(x4.data_ptr(),dataptr.get(),sizes,cudaMemcpyDeviceToHost);
        x5.grad_fn()->storage_impl_ = x4.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
        **/
        torch::Tensor x6 = torch::relu(x5);  
        /**
        sizes = x5.nbytes();
        dataptr = x5.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->set_data_ptr(at::getCPUAllocator()->allocate(sizes));
        cudaMemcpy(x5.data_ptr(),dataptr.get(),sizes,cudaMemcpyDeviceToHost);
        x6.grad_fn()->storage_impl_ = x5.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
        **/
        torch::Tensor x7 = fc2->forward(x6);
        /**
        sizes = x6.nbytes();
        dataptr = x6.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->set_data_ptr(at::getCPUAllocator()->allocate(sizes));
        cudaMemcpy(x6.data_ptr(),dataptr.get(),sizes,cudaMemcpyDeviceToHost);
        x7.grad_fn()->storage_impl_ = x6.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
        **/
        torch::Tensor x8 = torch::log_softmax(x7, /*dim=*/1);
        /**
        sizes = x7.nbytes();
        dataptr = x7.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->set_data_ptr(at::getCPUAllocator()->allocate(sizes));
        cudaMemcpy(x7.data_ptr(),dataptr.get(),sizes,cudaMemcpyDeviceToHost);
        x8.grad_fn()->storage_impl_ = x7.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
        **/
        //std::cout<<"forward"<<std::endl;
        return x8;
    }

    torch::nn::Conv2d conv1;
    torch::nn::Conv2d conv2;
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
};


struct net : torch::nn::Module{
    torch::Tensor forward(torch::Tensor x){
        torch::Tensor x1 = fc1->forward(x);
        torch::Tensor x2 = fc2->forward(x);
        torch::Tensor x3 = fc3->forward(x);
    }
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
    torch::nn::Linear fc3;
};



struct Options {
    std::string data_root{"/nfs/home/jsxnh/data/MNIST/raw"};
    int32_t batch_size{64};
    int32_t epochs{20};
    double lr{0.01};
    double momentum{0.5};
    bool no_cuda{false};
    int32_t seed{1};
    int32_t test_batch_size{1000};
    int32_t log_interval{10};
};

template <typename DataLoader>
void train(
    int32_t epoch,
    const Options& options,
    Net& model,
    torch::Device device,
    DataLoader& data_loader,
    torch::optim::SGD& optimizer,
    size_t dataset_size) {
    model.train();
    size_t batch_idx = 0;
    for (auto& batch : data_loader) {
        //std::cout<<"*****"<<std::endl;
        auto data = batch.data.to(device), targets = batch.target.to(device);
        //std::cout<<data.toString()<<std::endl;
        //std::cout<<targets<<std::endl;
        //return ;
        //std::cout<<"type:"<<typeid(data).name()<<std::endl;
        //std::cout<<"---"<<std::endl;
        
        optimizer.zero_grad();
        
        auto output = model.forward(data);
        
        //std::cout<<output.is_cuda()<<std::endl;
        //std::cout<<"size:"<<output.sizes()<<std::endl;
        //std::cout<<"****"<<std::endl;
        //output.to()
        //out.to(torch::kCUDA);
        //torch::cross_out
        //torch::nn::AvgPool2d
        auto loss = torch::nll_loss(output, targets);
        //std::cout<<"end"<<std::endl;
        //std::cout<<loss.is_cuda()<<std::endl;
        loss.backward();
        //std::cout<<"step"<<std::endl;
        optimizer.step();

        if (batch_idx++ % options.log_interval == 0) {
            std::cout << "Train Epoch: " << epoch << " ["
                      << batch_idx * batch.data.size(0) << "/" << dataset_size
                      << "]\tLoss: " << loss.template item<float>() << std::endl;
        }
        //return ;
    }
}
/**
template <typename DataLoader>
void test(
    Net& model,
    torch::Device device,
    DataLoader& data_loader,
    size_t dataset_size) {
    torch::NoGradGuard no_grad;
    model.eval();
    double test_loss = 0;
    int32_t correct = 0;
    for (const auto& batch : data_loader) {
        auto data = batch.data.to(device);
        auto targets = batch.target.to(device);
        auto output = model.forward(data);
        test_loss += torch::nll_loss(
            output,
            targets,
           {},
            at::Reduction::Sum)
            .template item<float>();
        auto pred = output.argmax(1);
        correct += pred.eq(targets).sum().template item<int64_t>();
    }

    test_loss /= dataset_size;
    std::cout << "Test set: Average loss: " << test_loss
              << ", Accuracy: " << static_cast<double>(correct) / dataset_size << std::endl;
}**/

struct Normalize : public torch::data::transforms::TensorTransform<> {
    Normalize(float mean, float stddev)
        : mean_(torch::tensor(mean)), stddev_(torch::tensor(stddev)) {}
    torch::Tensor operator()(torch::Tensor input) {
        return input.sub_(mean_).div_(stddev_);
    }
    torch::Tensor mean_, stddev_;
};


void func1(int& x,int y){
    x++;
    y++;
    
}

class node{
public:
    std::function<void()> f;

};

void func2(node* n){
    n->f();
}
node* n = new node();
void func3(int& x,int y){
    int z = 3;
    n->f = [&z,y](){func1(z,y);};
}
void func4(){
    n->f();
}


auto main(int argc, const char* argv[]) -> int {
    /**
    struct timeval start_, end_;
    for(int i=10;i<=28;i++){
        size_t N = 1 << i;
        size_t nBytes = N * sizeof(float);
        float *x,*y;
        x = (float*)malloc(nBytes);
        y = (float*)malloc(nBytes);
        for (int i = 0; i < N; ++i)
        {
            x[i] = 10.0;
            //y[i] = 20.0;
        }

        std::cout<<"nbytes:"<<nBytes<<std::endl;
        float *d_x;
        cudaMalloc((void**)&d_x, nBytes);
        gettimeofday(&start_, NULL );
        cudaMemcpy((void*)d_x, (void*)x, nBytes, cudaMemcpyHostToDevice);
        gettimeofday(&end_, NULL );

        int timeuse = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
        std::cout<<"time:"<<timeuse<<std::endl;

        gettimeofday(&start_, NULL );
        cudaMemcpy((void*)y, (void*)d_x, nBytes, cudaMemcpyDeviceToHost);
        gettimeofday(&end_, NULL );
        int timeuse2 = 1000000 * ( end_.tv_sec - start_.tv_sec ) + end_.tv_usec - start_.tv_usec;
        std::cout<<"time:"<<timeuse2<<std::endl;

        cudaFree(d_x);
        free(x);
    }**/

    //torch::Tensor tensor = at::empty({3,3,4});
    //at::addmm()

    //torch::empty()

    /**
    torch::manual_seed(0);

    Options options;
    torch::DeviceType device_type;
    if (torch::cuda::is_available() && !options.no_cuda) {
        std::cout << "CUDA available! Training on GPU" << std::endl;
        device_type = torch::kCUDA;
    } else {
        std::cout << "Training on CPU" << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    Net model;
    model.to(device);
    //torch::data::datasets::
    auto train_dataset =
        torch::data::datasets::MNIST(
            options.data_root, torch::data::datasets::MNIST::Mode::kTrain)
        .map(Normalize(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());
    const auto dataset_size = train_dataset.size().value();

    auto train_loader = torch::data::make_data_loader(
        train_dataset, options.batch_size);
    
    auto test_dataset = torch::data::datasets::MNIST(
        options.data_root, torch::data::datasets::MNIST::Mode::kTest)
        .map(Normalize(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());
    auto test_size = test_dataset.size().value();
    auto test_loader = torch::data::make_data_loader(
        test_dataset,
        options.batch_size);
    
    torch::optim::SGD optimizer(
        model.parameters(),
        torch::optim::SGDOptions(options.lr).momentum(options.momentum));
    for (size_t epoch = 1; epoch <= options.epochs; ++epoch) {
        train(
            epoch, options, model, device, *train_loader, optimizer, dataset_size);
        //test(model, device, *test_loader, test_size);
    }
    **/

    int x = 1;
    int y = 2;
    func3(x,y);
    func4();
    std::cout<<x<<std::endl;

    return 0;
}