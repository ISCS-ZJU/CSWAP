#include <torch/torch.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include<fstream>
#include<string>

class TinyImagenet :public torch::data::datasets::Dataset<TinyImagenet>{
public:

    torch::data::Example<> get(size_t index) {
        return {images_[index], targets_[index]};
    }
    torch::optional<size_t> size() const {
        return images_.size(0);
    }
    TinyImagenet(std::string path,std::string label){
        images_ = torch::empty({10000, 3,224, 224}, torch::kByte);
        targets_ =  torch::empty(10000, torch::kInt32);
        std::ifstream reader(path,std::ios::binary);
        std::ifstream reader2(label,std::ios::binary);
        void* meta_data = malloc(32);
        reader.read(reinterpret_cast<char*>(meta_data),32);
        reader2.read(reinterpret_cast<char*>(meta_data),32);
        for(int i=0;i<10000;i++){
            reader2.read(reinterpret_cast<char*>(targets_.data_ptr())+i*4,4);
            reader.read(reinterpret_cast<char*>(images_.data_ptr())+i*150528,150528);
        }
        targets_ = targets_.to(torch::kInt64);
        images_ = images_.to(torch::kFloat32).div_(255);
        //std::cout<<images_<<std::endl;
    }
private:
    torch::Tensor images_,targets_;
};

struct Normalize : public torch::data::transforms::TensorTransform<> {
    Normalize(float mean, float stddev)
        : mean_(torch::tensor(mean)), stddev_(torch::tensor(stddev)) {}
    torch::Tensor operator()(torch::Tensor input) {
        return input.sub_(mean_).div_(stddev_);
    }
    torch::Tensor mean_, stddev_;
};

