# Efficient Non-Sampling Knowledge Graph Embedding

If you use the code, please cite our [paper](https://arxiv.org/abs/2104.10796):

```
 @inproceedings{li2021efficient,
   title={Efficient Non-Sampling Knowledge Graph Embedding},
   author={Li, Zelong and Ji, Jianchao and Fu, Zuohui and Ge, Yingqiang and Xu, Shuyuan and Chen, Chong and Zhang, Yongfeng},
   booktitle={Proceedings of the the Web Conference 2021},
   year={2021}
 }
```

and the following [paper](http://aclweb.org/anthology/D18-2024)

```
 @inproceedings{han2018openke,
   title={OpenKE: An Open Toolkit for Knowledge Embedding},
   author={Han, Xu and Cao, Shulin and Lv Xin and Lin, Yankai and Liu, Zhiyuan and Sun, Maosong and Li, Juanzi},
   booktitle={Proceedings of EMNLP},
   year={2018}
 }
```

This package is mainly contributed by [Zelong Li](https://github.com/lzl65825), [Jianchao Ji](https://github.com/jianchaoji), [Yongfeng Zhang](https://github.com/evison).

## Overview

This is the implementation of the framework mentioned in our paper, Efficient Non-Sampling Knowledge Graph Embedding. Some parts of our work are based on the PyTorch version of [OpenKE](https://github.com/thunlp/OpenKE), which is an open source framework for Knowledge Embedding implemented with PyTorch.

We apply our framework to four embedded models, DistMult, SimplE, ComplEx, and TransE. The experimental results show that, compared with the model based on negative sampling, the model based on non-sampling has much higher training speed and better or competitive performance. 

## Parameter settings

- Dataset: FB15K237, WN18RR
- train_time: 2000
- alpha: 0.0001
- coefficient of regularization: [0.1, 0.01, 0.001, 0.0001]
- weight_decay: [0.1, 0.3, 0.5, 0.7]
- dim: 200
- pos_para (i.e,, c+): 1
- neg_para (i.e., c-): [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]

## Models

Models Based on Negative Sampling (OpenKE, PyTorch):

*	RESCAL
*  DistMult, ComplEx, Analogy
*  TransE, TransH, TransR, TransD
*  SimplE
*	RotatE

Models Based on Non-Sampling (NS):

* DistMult, SimplE, ComplEx, TransE

We welcome any issues and requests for model implementation and bug fix.

## Installation

To run the model based on negative sampling, please follow the [installation](https://github.com/thunlp/OpenKE#installation) of OpenKE.

1. Install [PyTorch](https://pytorch.org/get-started/locally/)

2. Clone the project:
```bash
git clone OpenKE-PyTorch https://github.com/rutgerswiselab/NS-KGE
cd Non-sampling-KB-embedding
cd openke
```
3. Compile C++ files
```bash
bash make.sh
```
4. Quick Start
```bash
cd ../
cp examples/train_NS_transe_FB15K237.py ./
python train_NS_transe_FB15K237.py
```

## Experiments

We have provided the state-of-the-art performace (Hits@10 (filter)) on FB15K237 and WN18RR.

|Model			|	FB15K237	|	WN18RR	
|:-:		|:-:	|:-:  |
|DistMult	|0.345	|0.461|
|NS-DistMult	|0.373	|0.462|
|SimplE	|0.355	|0.463|
|NS-SimplE	|0.370	|0.459|
|ComplEx	|0.415	|0.474|
|NS-ComplEx	|0.390|0.485
|TransE |0.316	|0.145|
|NS-TransE	|0.447	|0.437|


<strong> We are still trying more hyper-parameters and more training strategies for knowledge graph models. </strong> Hence, this table is still in change. We welcome everyone to help us update this table and hyper-parameters.
