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
#include<ATen/TensorIndexing.h>
#include<thread>
#include <ATen/core/grad_mode.h>
#include<unordered_map>
#include<algorithm>
#include<utility>
#include<math.h>
#include"Execution.h"

using namespace std;


vector<string> sqlit(string& s,char a){
        vector<string> res;
        int pre = 0;
        for(int i=0;i<s.size();i++){
            if(s[i]==a){
                if(i-pre>0)
                    res.push_back(s.substr(pre,i-pre));
                pre = i+1;
            }
        }
        if(pre<s.size()){
            res.push_back(s.substr(pre));
        }
        return res;

}


class TorchVocab{
public:


    

    TorchVocab(){}
    void init_(unordered_map<string,int>& counter,int max_size=0,int min_freq=1,vector<string> specials={"<pad>","<oov>"}){
        freqs = counter;
        this->min_freq = max(min_freq,1);
        itos = specials;
        if(max_size){
            this->max_size = max_size+itos.size();
        }else{
            this->max_size = -1;
        }
        for(string s:specials){
            freqs.erase(s);
        }

        vector<pair<int,string>> v;
        for(auto it=freqs.begin();it!=freqs.end();it++){
            v.push_back({it->second,it->first});
        }
        sort(v.begin(),v.end());
        for(int i=0;i<v.size();i++){
            if(v[i].first<this->min_freq||itos.size()==this->max_size) break;
            itos.push_back(v[i].second);
        }
        for(int i=0;i<itos.size();i++){
            stoi_[itos[i]] = i;
        }



    }

    int len(){
        return itos.size();
    }



    unordered_map<string,int> freqs;
    int min_freq;
    vector<string> itos;
    int max_size;
    unordered_map<string,int> stoi_;

};


class Vocab:public TorchVocab{
public:
    Vocab(){
        pad_index = 0;
        unk_index = 1;
        eos_index = 2;
        sos_index = 3;
        mask_index = 4;
    }/**
    Vocab(unordered_map<string,int>& counter,int max_size=0,int min_freq=1){
        
        super(counter,max_size,min_freq,{"<pad>", "<unk>", "<eos>", "<sos>", "<mask>"});
    }**/

    void init(string path,int max_size=0,int min_freq=1){
        fstream in(path,ios::in);
        unordered_map<string,int> counter;
        string line;
        while(getline(in,line)){
            vector<string> words = sqlit(line,'\t');
            for(string word:words){
                vector<string> items = sqlit(word,' ');
                for(string item:items){
                    counter[item]++;
                }
            }
        }
        init_(counter,max_size,min_freq,{"<pad>", "<unk>", "<eos>", "<sos>", "<mask>"});
    }



    int pad_index;
    int unk_index;
    int eos_index;
    int sos_index;
    int mask_index;


};


class BERTDataset{
public:
    vector<torch::Tensor> get(size_t index) {
        vector<string> res = random_set(index);
        vector<int> t1_label,t2_label;
        vector<int> t1_random = random_word(res[0],t1_label);
        vector<int> t2_random = random_word(res[1],t2_label);
        t1_random.push_back(vocab.eos_index);
        t1_random.insert(t1_random.begin(),vocab.sos_index);
        t2_random.push_back(vocab.eos_index);
        t1_label.push_back(vocab.pad_index);
        t1_label.insert(t1_label.begin(),vocab.pad_index);
        t2_label.push_back(vocab.pad_index);
        vector<int> segment_label;
        for(int i=0;i<t1_random.size();i++){
            if(segment_label.size()>=seq_len) break;
            segment_label.push_back(1);
        }
        for(int i=0;i<t2_random.size();i++){
            if(segment_label.size()>=seq_len) break;
            segment_label.push_back(2);
        }
        t1_random.insert(t1_random.end(),t2_random.begin(),t2_random.end());
        if(seq_len<t1_random.size()){
            t1_random.erase(t1_random.begin()+seq_len,t1_random.end());
        }
        t1_label.insert(t1_label.end(),t2_label.begin(),t2_label.end());
        if(seq_len<t1_label.size()){
            t1_label.erase(t1_label.begin()+seq_len,t1_label.end());
        }

        if(t1_random.size()<seq_len){
            int size = seq_len-t1_random.size();
            for(int i=0;i<size;i++){
                segment_label.push_back(vocab.pad_index);
                t1_random.push_back(vocab.pad_index);
                t1_label.push_back(vocab.pad_index);
            }
        }
        torch::Tensor input = torch::empty({t1_random.size()},torch::kInt64);
        for(int i=0;i<t1_random.size();i++){
            input[i] = t1_random[i];
        }
        torch::Tensor label = torch::empty({t1_label.size()},torch::kInt64);
        for(int i=0;i<t1_label.size();i++){
            label[i] = t1_label[i];
        }
        torch::Tensor s_label = torch::empty({segment_label.size()},torch::kInt64);
        for(int i=0;i<segment_label.size();i++){
            s_label[i] = segment_label[i];
        }
        torch::Tensor is_next = torch::empty({1},torch::kInt64);
        is_next[0] = atoi(res[2].c_str());
        //return input;
        //cout<<torch::cat({input,label,s_label})<<endl;
        return {input,label,s_label,is_next};
        //return {images_[index], targets_[index]};
    }
    int size() const {
        return corpus_lines;
    }

    

    vector<string> random_set(size_t index){
        string s1 = lines[index][0];
        string s2 = lines[index][1];
        srand(time(NULL));
        int x = rand()%100;
        if(x>=50){
            return {s1,s2,"1"};
        }else{
            srand(time(NULL));
            x = rand()%lines.size();
            return {s1,lines[x][1],"0"};
        }

    }

    vector<int> random_word(string& s,vector<int>& lables){
        vector<string> token = sqlit(s,' ');
        //vector<int> lables;
        vector<int> res(token.size());
        for(int i=0;i<token.size();i++){
            srand(time(NULL));
            int x = rand()%100;
            string toke = token[i];
            if(x<15){
                double prob = x*1.0/15;
                if(prob<0.8){
                    res[i] = vocab.mask_index;
                }else if(prob<0.9){
                    res[i] = rand()%vocab.len();
                }else{
                    if(vocab.stoi_.count(token[i])){
                        res[i] = vocab.stoi_[token[i]];
                    }else{
                        res[i] = vocab.unk_index;
                    }
                }
                if(vocab.stoi_.count(toke)){
                    lables.push_back(vocab.stoi_[toke]);
                }else{
                    lables.push_back(vocab.unk_index);
                }
            }else{
                if(vocab.stoi_.count(token[i])){
                    res[i] = vocab.stoi_[token[i]];
                }else{
                    res[i] = vocab.unk_index;
                }
                lables.push_back(0);

            }
        }
        return res;

    }


    BERTDataset(){}
    BERTDataset(string corpus_path,int seq_len=20,string encoding="utf-8",int corpus_lines=0,bool on_memory=true){
        this->corpus_path = corpus_path;
        this->vocab = vocab;
        this->seq_len = seq_len;
        this->encoding = encoding;
        this->corpus_lines = corpus_lines;
        this->on_memory = on_memory;
        vocab.init(corpus_path);
        fstream in(this->corpus_path,ios::in);
        string str;
        while(getline(in,str)){
            lines.emplace_back(sqlit(str,'\t'));
        }
        /**
        for(int i=0;i<lines.size();i++){
            for(int j=0;j<lines[i].size();j++){
                cout<<lines[i][j]<<endl;
            }
        }**/
        this->corpus_lines = lines.size();


    }

    string corpus_path;
    Vocab vocab;
    int seq_len;
    string encoding;
    int corpus_lines;
    bool on_memory;

    vector<vector<string>> lines;
    torch::Tensor images_,targets_;



};

class Dataloader{
public:
    Dataloader(){}
    Dataloader(BERTDataset& dataset,int batch_size){
        data = dataset;
        bsize = batch_size;
        cur = 0;
        if(bsize>data.size()){
            cout<<"bsize bigger than dataset";
        }
    }
    vector<torch::Tensor> iter(){
        int len = bsize;
        if(cur+len>data.size()){
            len = data.size()-cur;
        }

        vector<vector<torch::Tensor>> v;
        for(int i=0;i<len;i++){
            v.emplace_back(data.get(cur+i));
        }

        //cout<<v[0][0].sizes()[0]<<endl;
        torch::Tensor input = torch::empty({len,v[0][0].sizes()[0]},torch::kInt64);
        torch::Tensor label = torch::empty({len,v[0][1].sizes()[0]},torch::kInt64);
        torch::Tensor s_label = torch::empty({len,v[0][2].sizes()[0]},torch::kInt64);
        torch::Tensor is_next = torch::empty({len},torch::kInt64);
        for(int i=0;i<v.size();i++){
            //cout<<i<<endl;
            //cout<<v[i][0];
            //cout<<v[i][1];
            //cout<<v[i][2];
            input[i] = v[i][0];
            label[i] = v[i][1];
            s_label[i] = v[i][2];
            /**
            for(int j=0;j<v[0][0].sizes()[0];j++)
                input[i][j] = v[i][0][j];
            for(int j=0;j<v[0][1].sizes()[0];j++)
                label[i][j] = v[i][1][j];
            for(int j=0;j<v[0][2].sizes()[0];j++)
                s_label[i][j] = v[i][2][j];
            **/
            is_next[i] = v[i][3][0]; 
        }
        //cout<<input<<endl;
        //cout<<label<<endl;
        //cout<<s_label<<endl;

        cur+=len;
        cur%=data.size();
        return {input,label,s_label,is_next};
    }

private:
    BERTDataset data;
    int bsize;
    int cur;


};

Execution exe;



struct EmbeddingImpl : torch::nn::Module{
public:
    EmbeddingImpl(){}
    EmbeddingImpl(int num_embedding,int embedding_dim,int padding_idx=0){
        //torch::embedding()
        weight = torch::empty({num_embedding,embedding_dim});
        register_parameter("weight",weight);
        at::GradMode::set_enabled(false);
        weight.normal_(0,1);
        weight[padding_idx].fill_(0);
        at::GradMode::set_enabled(true);
        pad_idx = padding_idx;
    }

    torch::Tensor forward(torch::Tensor input){
        exe.pre(input,ACT);
        torch::Tensor x = torch::embedding(weight,input,pad_idx,false,false);
        exe.post(input,x,ACT);
        x.grad_fn()->forwardfunc = [this,&input]() -> torch::Tensor{ return torch::embedding(weight,input,pad_idx,false,false); };
        return x;
    }
    torch::Tensor weight;
    int pad_idx;

};
TORCH_MODULE(Embedding);


struct PositionalEmbeddingImpl : torch::nn::Module{
public:
    PositionalEmbeddingImpl(){}
    PositionalEmbeddingImpl(int d_model,int max_len=512){
        pe = torch::zeros({max_len,d_model}).to(torch::kFloat32);
        torch::Tensor position = torch::arange(0,max_len).to(torch::kFloat32).unsqueeze(1);
        torch::Tensor div_term = torch::arange(0,d_model,2).to(torch::kFloat32).mul(-1.0*log(10000.0)/d_model).exp();
        pe.index_put_({at::indexing::Slice(),at::indexing::Slice(0,c10::nullopt,2)},torch::sin(torch::mul(position,div_term)));
        pe.index_put_({at::indexing::Slice(),at::indexing::Slice(1,c10::nullopt,2)},torch::cos(torch::mul(position,div_term)));
        pe = pe.unsqueeze(0);
        register_buffer("pe",pe);
    }

    torch::Tensor forward(torch::Tensor& input){
        exe.pre(input,ACT);
        torch::Tensor x = pe.index({at::indexing::Slice(),at::indexing::Slice(c10::nullopt,input.sizes()[1],c10::nullopt)});
        exe.post(input,x,ACT,true);
        return x;
    }

    torch::Tensor pe;
};
TORCH_MODULE(PositionalEmbedding);

struct BERTEmbeddingImpl : torch::nn::Module{
public:
    BERTEmbeddingImpl(){}
    BERTEmbeddingImpl(int vocab_size,int embed_size,double dropout=0.1){
        token = Embedding(vocab_size,embed_size);
        segment = Embedding(3,embed_size);
        position = PositionalEmbedding(embed_size);
        register_module("token",token);
        register_module("segment",segment);
        register_module("position",position);
        drop = dropout;
    }


    torch::Tensor forward(torch::Tensor& sequence,torch::Tensor& segment_label){
        torch::Tensor x1 = token->forward(sequence);
        torch::Tensor x2 = position->forward(sequence);
        torch::Tensor x3 = segment->forward(segment_label);
        //torch::Tensor x = token->forward(sequence)+position->forward(sequence)+segment->forward(segment_label);
        exe.pre({x1,x2,x3},ACT);
        torch::Tensor x = x1+x2+x3;
        exe.post({x1,x2,x3},x,ACT);
        x.grad_fn()->forwardfunc = [&x1,&x2,&x3]() -> torch::Tensor{ return x1+x2+x3; };

        exe.pre(x,ACT);
        torch::Tensor res = torch::dropout(x,drop,true);
        exe.post(x,res);
        res.grad_fn()->forwardfunc = [&x,this]() -> torch::Tensor{ return torch::dropout(x,drop,true); };

        return res;

    }

    Embedding token;
    Embedding segment;
    PositionalEmbedding position;
    double drop;


};
TORCH_MODULE(BERTEmbedding);

struct Attention : torch::nn::Module{
public:
    Attention(){}
    torch::Tensor forward(torch::Tensor& query,torch::Tensor& key,torch::Tensor& value,torch::Tensor& mask,
    c10::optional<double> dropout = c10::nullopt){
        
        exe.pre({query,key},ACT);
        //std::cout<<exe.index<<":"<<exe.begin_stage[query.tensor_id]<<":"<<exe.begin_stage[key.tensor_id]<<":"<<key.tensor_id<<":"<<query.tensor_id<<endl;
        torch::Tensor scores = torch::matmul(query,key.transpose(-2,-1))/sqrt(query.size(-1));
        
        //std::cout<<torch::matmul(query,key.transpose(-2,-1)).is_cuda()<<endl;
        exe.post({query,key},scores,ACT);
        
        scores.grad_fn()->forwardfunc = [&query,&key]() -> torch::Tensor{ return torch::matmul(query,key.transpose(-2,-1))/sqrt(query.size(-1)); };
        
        exe.pre({scores,mask},ACT);
        
        //std::cout<<exe.index<<":"<<exe.begin_stage[scores.tensor_id]<<":"<<exe.begin_stage[mask.tensor_id]<<endl;
       
        torch::Tensor scores1 = scores.masked_fill(mask==0,-1e9);
        //std::cout<<"after"<<endl;
        //std::cout<<scores1.is_cuda()<<":";
        exe.post({scores,mask},scores1,ACT,true);
        scores1.grad_fn()->forwardfunc = [&scores,&mask]() -> torch::Tensor{ return scores.masked_fill(mask==0,-1e9); };

        exe.pre(scores1,ACT);
        torch::Tensor p_attn = torch::softmax(scores1,-1);
        exe.post(scores1,p_attn,ACT);
        p_attn.grad_fn()->forwardfunc = [&scores1]() -> torch::Tensor{ return torch::softmax(scores1,-1); };

        torch::Tensor p_attn_ = p_attn;
        if(dropout.has_value()){
            exe.pre(p_attn,ACT);
            p_attn_ = torch::dropout(p_attn,dropout.value(),true);
            exe.post(p_attn,p_attn_);
            p_attn_.grad_fn()->forwardfunc = [&p_attn,dropout]() -> torch::Tensor{ return torch::dropout(p_attn,dropout.value(),true); };
        }

        exe.pre({p_attn_,value},ACT);
        torch::Tensor res = torch::matmul(p_attn_,value);
        //std::cout<<p_attn_.is_cuda()<<endl;
        //std::cout<<res.is_cuda()<<endl;
        exe.post({p_attn_,value},res,ACT);
        res.grad_fn()->forwardfunc = [&p_attn_,&value]() -> torch::Tensor{ return torch::matmul(p_attn_,value); };
        return res;
    }

};

struct MultiHeadedAttentionImpl: torch::nn::Module{
public:
    MultiHeadedAttentionImpl(){}
    MultiHeadedAttentionImpl(int h,int d_model,double dropout=0.1){
        d_k = d_model/h;
        this->h = h;
        this->dropout = dropout;
        for(int i=0;i<4;i++){
            linears[i] = torch::nn::Linear(d_model,d_model);
            register_module("fc"+to_string(i),linears[i]);
        }

    }
    torch::Tensor forward(torch::Tensor& query,torch::Tensor& key,torch::Tensor& value,torch::Tensor& mask){
        
        int batch_size = query.size(0);

        exe.pre(query,ACT);
        torch::Tensor query1 = linears[0]->forward(query);
        exe.post(query,query1,ACT);
        query1.grad_fn()->forwardfunc = [this,&query]() -> torch::Tensor{ return linears[0]->forward(query); };

        exe.pre(key,ACT);
        torch::Tensor key1 = linears[1]->forward(key);
        exe.post(key,key1,ACT);
        key1.grad_fn()->forwardfunc = [this,&key]() -> torch::Tensor{ return linears[1]->forward(key); };

        exe.pre(value,ACT);
        torch::Tensor value1 = linears[2]->forward(value);
        exe.post(value,value1,ACT);
        value1.grad_fn()->forwardfunc = [this,&value]() -> torch::Tensor{ return linears[2]->forward(value); };
        //std::cout<<value1.tensor_id<<":"<<key1.tensor_id<<":"<<query1.tensor_id<<endl;
        exe.pre(query1,ACT);
        torch::Tensor query2 = query1.reshape({batch_size,-1,h,d_k}).transpose(1,2);
        exe.post(query1,query2,ACT,true);

        exe.pre(key1,ACT);
        torch::Tensor key2 = key1.reshape({batch_size,-1,h,d_k}).transpose(1,2);
        exe.post(key1,key2,ACT,true);

        exe.pre(value1,ACT);
        torch::Tensor value2 = value1.reshape({batch_size,-1,h,d_k}).transpose(1,2);
        exe.post(value1,value2,ACT,true);

        torch::Tensor x = attention.forward(query2,key2,value2,mask,dropout);
        //cout<<x.tensor_id<<endl;
        exe.pre(x,ACT);
        torch::Tensor x1 = x.transpose(1,2).contiguous().view({batch_size,-1,h*d_k});
        exe.post(x,x1,ACT,true);

        exe.pre(x1,ACT);
        torch::Tensor res = linears[3]->forward(x1);
        exe.post(x1,res,ACT);
        res.grad_fn()->forwardfunc = [this,&x1]() -> torch::Tensor{ return linears[3]->forward(x1); };

        return res;

    }

    int d_k;
    int h;
    double dropout;
    Attention attention;
    torch::nn::Linear linears[4]{nullptr,nullptr,nullptr,nullptr};

};
TORCH_MODULE(MultiHeadedAttention);

struct PositionwiseFeedForwardImpl : torch::nn::Module{
public:
    PositionwiseFeedForwardImpl(){}
    PositionwiseFeedForwardImpl(int d_model,int d_ff,double dropout=0.1){
        fc1 = torch::nn::Linear(d_model,d_ff);
        fc2 = torch::nn::Linear(d_ff,d_model);
        this->dropout = dropout;
        register_module("fc1",fc1);
        register_module("fc2",fc2);
    }
    torch::Tensor forward(torch::Tensor& input){
        exe.pre(input);
        torch::Tensor x = fc1->forward(input);
        exe.post(input,x);
        x.grad_fn()->forwardfunc = [this,&input]() -> torch::Tensor{ return fc1->forward(input); };

        exe.pre(x);
        torch::Tensor x1 = torch::gelu(x);
        exe.post(x,x1);
        x1.grad_fn()->forwardfunc = [&x]() -> torch::Tensor{ return torch::gelu(x); };
        
        exe.pre(x1);
        torch::Tensor x2 = torch::dropout(x1,dropout,true);
        exe.post(x1,x2);
        x2.grad_fn()->forwardfunc = [&x1,this]() -> torch::Tensor{ return torch::dropout(x1,dropout,true); };

        exe.pre(x2);
        torch::Tensor x3 = fc2->forward(x2);
        exe.post(x2,x3);
        x3.grad_fn()->forwardfunc = [this,&x2]() -> torch::Tensor{ return fc2->forward(x2); };
        return x3;
    }

    torch::nn::Linear fc1{nullptr},fc2{nullptr};
    double dropout;

};
TORCH_MODULE(PositionwiseFeedForward);

struct LayerNormImpl : torch::nn::Module{
public:
    LayerNormImpl(){}
    LayerNormImpl(int features,double eps=1e-6){
        a_2 = torch::ones(features);
        b_2 = torch::zeros(features);
        register_parameter("a_2",a_2);
        register_parameter("b_2",b_2);
        this->eps = eps;
    }

    torch::Tensor forward(torch::Tensor& input){
        exe.pre(input,ACT);
        torch::Tensor mean = input.mean(-1,true);
        exe.post(input,mean,ACT);
        mean.grad_fn()->forwardfunc = [&input]() -> torch::Tensor{ return input.mean(-1,true); };

        exe.pre(input,ACT);
        torch::Tensor st = input.std(-1,true,true);
        exe.post(input,st,ACT);
        st.grad_fn()->forwardfunc = [&input]() -> torch::Tensor{ return input.std(-1,true,true); };

        exe.pre({input,mean,st},ACT);
        torch::Tensor res = a_2*(input-mean)/(st+eps)+b_2;
        exe.post({input,mean,st},res,ACT);
        res.grad_fn()->forwardfunc = [&input,&mean,&st,this]() -> torch::Tensor{ return a_2*(input-mean)/(st+eps)+b_2; };
        return res;

    }
    torch::Tensor a_2;
    torch::Tensor b_2;
    double eps;


};
TORCH_MODULE(LayerNorm);


struct TransformerBlockImpl: torch::nn::Module{
public:
    TransformerBlockImpl(){}
    TransformerBlockImpl(int hidden,int attn_heads,int feed_forward_hidden,double dropout){
        attention = MultiHeadedAttention(attn_heads,hidden);
        feed_forward = PositionwiseFeedForward(hidden,feed_forward_hidden,dropout);
        norm1 = LayerNorm(hidden);
        norm2 = LayerNorm(hidden);
        register_module("attention",attention);
        register_module("feed_forward",feed_forward);
        register_module("norm1",norm1);
        register_module("norm2",norm2);
        this->dropout = dropout;

    }
    torch::Tensor forward(torch::Tensor& input,torch::Tensor& mask){
        torch::Tensor x = norm1->forward(input);
        x = attention->forward(x,x,x,mask);

        exe.pre(x,ACT);
        torch::Tensor x1 = torch::dropout(x,dropout,true);
        exe.post(x,x1,ACT,true);
        
        exe.pre({input,x1},ACT);
        torch::Tensor x2 = input+x1;
        exe.post({input,x1},x2,ACT);
        x2.grad_fn()->forwardfunc = [&input,&x1]() -> torch::Tensor{ return input+x1; };

        torch::Tensor y = norm2->forward(x2);

        y = feed_forward->forward(y);

        exe.pre(y);
        torch::Tensor y1 = torch::dropout(y,dropout,true);
        exe.post(y,y1);
        y1.grad_fn()->forwardfunc = [&y,this]() -> torch::Tensor{ return torch::dropout(y,dropout,true); };

        exe.pre({x2,y1});
        torch::Tensor y2 = x2+y1;
        exe.post({x2,y1},y2);
        y2.grad_fn()->forwardfunc = [&x2,&y1]() -> torch::Tensor{ return x2+y1; };

        exe.pre(y2);
        torch::Tensor y3 = torch::dropout(y2,dropout,true);
        exe.post(y2,y3);
        y3.grad_fn()->forwardfunc = [&y2,this]() -> torch::Tensor{ return torch::dropout(y2,dropout,true); };
        return y3;
    }

    MultiHeadedAttention attention;
    PositionwiseFeedForward feed_forward;
    LayerNorm norm1,norm2;
    double dropout;

};
TORCH_MODULE(TransformerBlock);

using namespace torch;
struct BERT : torch::nn::Module{
public:
    BERT(){}
    BERT(int vocab_size,int hidden=768,int n_layers=12,int attn_heads=12,double dropout=0.1){
        embedding = BERTEmbedding(vocab_size,hidden);
        register_module("embedding",embedding);
        transformer_blocks.resize(n_layers);
        for(int i=0;i<n_layers;i++){
            transformer_blocks[i] = TransformerBlock(hidden,attn_heads,hidden*4,dropout);
            register_module("transformer_block"+to_string(i),transformer_blocks[i]);
        }
        fc1 = torch::nn::Linear(hidden,2);
        fc2 = torch::nn::Linear(hidden,vocab_size);
        criterion = torch::nn::NLLLoss(torch::nn::NLLLossOptions().ignore_index(0));
        register_module("fc1",fc1);
        register_module("fc2",fc2);
    }
    torch::Tensor forward(torch::Tensor& input,torch::Tensor& segment_info,torch::Tensor& label,torch::Tensor& is_next){
        exe.preforward();

        is_next.tensor_id = -4;
        label.tensor_id = -3;
        segment_info.tensor_id = -2;
        input.tensor_id = -1;

        exe.pre(input);
        torch::Tensor mask = input.gt(0).unsqueeze(1).repeat({1,input.size(1),1}).unsqueeze(1);
        exe.post(input,mask,ACT,true);

        
        //mask.grad_fn()->forwardfunc = [&input]() -> torch::Tensor{ return input.gt(0).unsqueeze(1).repeat({1,input.size(1),1}).unsqueeze(1); };

        torch::Tensor x = embedding->forward(input,segment_info);
        
        for(int i=0;i<transformer_blocks.size();i++){
            x = transformer_blocks[i]->forward(x,mask);
        }
        
        
        exe.pre(x);
        torch::Tensor yy1 = fc1->forward(x.index({at::indexing::Slice(),0}));
        exe.post(x,yy1);
        yy1.grad_fn()->forwardfunc = [this,&x]() -> torch::Tensor{ return fc1->forward(x.index({at::indexing::Slice(),0})); };

        exe.pre(yy1);
        torch::Tensor y1 = torch::log_softmax(yy1,-1);
        exe.post(yy1,y1);
        y1.grad_fn()->forwardfunc = [&yy1]() -> torch::Tensor{ return torch::log_softmax(yy1,-1); };
       

        exe.pre(x);
        torch::Tensor yy2 = fc2->forward(x);
        exe.post(x,yy2);
        yy2.grad_fn()->forwardfunc = [this,&x]() -> torch::Tensor{ return fc2->forward(x); };
        //std::cout<<yy2.tensor_id<<endl;
        yy2.grad_fn()->func_ = std::bind(&torch::nn::LinearImpl::forward,fc2,std::placeholders::_1);

        exe.pre(yy2);
        torch::Tensor y2 = torch::log_softmax(yy2,-1);
        exe.post(yy2,y2);
        //std::cout<<y2.tensor_id<<endl;
        y2.grad_fn()->forwardfunc = [&yy2]() -> torch::Tensor{ return torch::log_softmax(yy2,-1); };
        y2.grad_fn()->func_ = [](Tensor a) -> torch::Tensor{ return torch::log_softmax(a,-1); };
        
        exe.pre({y1,is_next});
        torch::Tensor next_loss = criterion->forward(y1,is_next);
        exe.post({y1,is_next},next_loss,ACT,true);
        
        exe.pre({y2,label});
        torch::Tensor mask_loss = criterion->forward(y2.transpose(1,2),label);
        exe.post({y2,label},mask_loss,ACT,true);

        exe.pre({next_loss,mask_loss});
        torch::Tensor loss = next_loss+mask_loss;
        exe.post({next_loss,mask_loss},loss,ACT,true);

        exe.postforward();
        return loss;    
    }
   
   vector<TransformerBlock> transformer_blocks;
   BERTEmbedding embedding;
   torch::nn::Linear fc1{nullptr},fc2{nullptr};
   torch::nn::NLLLoss criterion{nullptr};

};

/**
class ScheduledOptim{
public:
    ScheduledOptim(){}
    ScheduledOptim(torch::optim::Adam& optimizer,int d_model,int n_warmup_steps){
        adam = optimizer;
        this->n_warmup_steps = n_warmup_steps;
        n_current_steps = 0;
        init_lr = pow(d_model,-0.5);


    }
    double _get_lr_scale(){
        return min(pow(n_current_steps,-0.5),pow(n_warmup_steps,-1.5)*n_current_steps);
    }

    void _update_learning_rate(){
        n_current_steps++;
        double lr = init_lr*_get_lr_scale();
        for(auto& x:adam.param_groups()){
            ((torch::optim::AdamOptions)x.options()).lr(lr);
        }
    }
    torch::optim::Adam adam;
    int n_warmup_steps;
    int n_current_steps;
    double init_lr;
};**/






int main(int argc,char* argv[]){

    double ratio = 0.6;
    int policy_ = 1;
    int batch_size = 32;

    if(argc>=2){
        batch_size = atoi(argv[1]);
    }
    
    if(argc>=3){
        policy_ = atoi(argv[2]);
    }
    if(argc>=4){
        ratio = atof(argv[3])/100.0;
    }

    string corpus_path = "/data/glue_data/QNLI/train.tsv";
    BERTDataset bd(corpus_path);
    BERT bert(bd.vocab.len(),256,8,8);
    
    Dataloader dataloader(bd,batch_size);
    //torch::autograd::AnomalyMode::set_enabled(true);
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
    bert.to(device);
    torch::optim::Adam adam(bert.parameters(),torch::optim::AdamOptions(0.0001).weight_decay(0.01));
    
    //c10::Profile3::set_enabled(true);
    exe.init_(300);
    c10::GetPolicyMaker()->read_lastvisit("/home/pytorch/memory_mamagement/build/logs/bert_lastvisit.txt",exe.begin_stage);
    //exe.init();
    
    c10::Passive::set_enabled(true);
    
    /**
    for(int i=0;i<236;i++){
        exe.policy[i] = 2;
    }
    exe.policy[9] = 0,exe.policy[10] = 0,exe.policy[11] = 0;
    exe.policy[37] = 0,exe.policy[38] = 0,exe.policy[39] = 0;
    exe.policy[65] = 0,exe.policy[66] = 0,exe.policy[67] = 0;
    exe.policy[93] = 0,exe.policy[94] = 0,exe.policy[95] = 0;
    exe.policy[121] = 0,exe.policy[122] = 0,exe.policy[123] = 0;
    exe.policy[149] = 0,exe.policy[150] = 0,exe.policy[151] = 0;
    exe.policy[177] = 0,exe.policy[178] = 0,exe.policy[179] = 0;
    exe.policy[207] = 0,exe.policy[206] = 0,exe.policy[205] = 0;
    **/
    

   //exe.policy[233] = 1;

    for(int i=0;i<5;i++){
        std::cout<<"iter:"<<i<<std::endl;

        
        if(i==2){
            c10::Profile::set_enabled(true);
        }

        vector<torch::Tensor> data = dataloader.iter();
        torch::Tensor input = data[0].to(device);
        torch::Tensor label = data[1].to(device);
        torch::Tensor segment_label = data[2].to(device);
        torch::Tensor is_next = data[3].to(device);

        torch::Tensor loss = bert.forward(input,segment_label,label,is_next);
        //cout<<v[0].sizes()<<":"<<v[1].sizes()<<endl;
        adam.zero_grad();
        loss.backward();
        adam.step();
        
        if(i==2){
            c10::Profile::set_enabled(false);
            c10::Passive::set_enabled(false);
            //c10::Profile3::set_enabled(false);
            c10::GetPolicyMaker()->init(ratio);
            //c10::GetPolicyMaker()->print();
            //c10::GetPolicyMaker()->write_lastvisit("logs/bert_lastvisit.txt");
            int time_;
            if(policy_==1){
                std::vector<int> policy1 = c10::GetPolicyMaker()->make_policy(time_);
                std::cout<<"iteration_time:"<<time_<<std::endl;
                for(int po:policy1){
                    std::cout<<po<<",";
                }
                //if(policy1.size()==0) return 0;
                //policy = policy1;
            }
            if(policy_==2){
                std::vector<int> policy2 = c10::GetPolicyMaker()->capuchin(time_);
                std::cout<<"capuchin_time:"<<time_<<std::endl;
                for(int po:policy2){
                    std::cout<<po<<",";
                }
                //if(policy2.size()==0) return 0;
                //policy = policy2;
            }
                
            if(policy_==3){
                c10::Passive::set_enabled(true);
                std::cout<<"+,"<<std::endl;
                bool is_vdnn = true;
               
            }
            if(policy_==4){
                std::cout<<"+,"<<std::endl;
                bool is_super = true;
                
            }

        }

        //cout<<loss<<endl;
        
        
        //c10::GetCompressProfile()->to_txt2("logs/bert_memoryload.txt");
        //c10::GetCompressProfile()->finish_iteration2();
        

    }
    
    

    return 0;
}
