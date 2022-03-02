# Accelerating Tensor Swapping in GPUs with Self-Tuning Compression (CSWAP+)


Data swapping between CPUs and GPUs is widely used to address the GPU memory shortage issue when training deep neural networks (DNNs) requiring a larger amount of memory than that a GPU may have. Data swapping may become a bottleneck when its latency is longer than the latency of DNN computations. Tensor compression in GPUs can reduce the data swapping time. However, existing works on compressing tensors in the virtual memory of GPUs have three major issues: lack of portability because its implementation requires additional (de)compression units in memory controllers, sub-optimal compression performance for varying tensor compression ratio and sizes, and poor adaptation to dense tensors because they only focus on sparse tensors.

We propose a self-tuning tensor compression framework, named CSWAP+, for improving the virtual memory management of GPUs. It uses GPUs for (de)compression directly and thus has high portability and is minimally dependent on GPU architecture features. Furthermore, it only applies compression on tensors that are deemed to be cost-effective considering their compression ratio, size, and the characteristics of compression algorithms at runtime. Finally, to adapt to DNN models with dense tensors, it also supports cost-effective lossy compression for dense tensors with nearly no model training accuracy degradation. We conduct the experiments through six representative memory-intensive DNN models. Compared to vDNN, CSWAP+ reduces tensor swapping latency by up to 50.9% and 46.1% with NVIDIA 100 GPU, for DNN models with sparse and dense tensors, respectively.


## System Requirements

1. CUDA Version >=
2. PyTorch Version >= 

## Build

```
mkdir build
....
....
```