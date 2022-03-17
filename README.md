# BayesFT: Bayesian Optimization for Fault Tolerant Neural Network Architecture

To deploy deep learning algorithms on resource-limited scenarios, an emerging device-resistive random access memory (ReRAM) has been regarded as promising via analog computing. However, the practicability of ReRAM is primarily limited due to the weight drifting of ReRAM neural networks due to multi-factor reasons, including manufacturing, thermal noises, and etc. In this paper, we propose a novel Bayesian optimization method for fault tolerant neural network architecture (BayesFT). For neural architecture search space design, instead of conducting neural architecture search on the whole feasible neural architecture search space, we first systematically explore the weight drifting tolerance of different neural network components, such as dropout, normalization, number of layers, and activation functions in which dropout is found to be able to improve the neural network robustness to weight drifting. Based on our analysis, we propose an efficient search space by only searching for dropout rates for each layer. Then, we use Bayesian optimization to search for the optimal neural architecture robust to weight drifting. Empirical experiments demonstrate that our algorithmic framework has outperformed the state-of-the-art methods by up to 10 times on various tasks, such as image classification and object detection.


## How to use it

First write the config file for a specific experiment  and then run main.py with `-c` to specify the path to access your config file.

In ./experiments/ directory, we've provided several examples as reference. The config files specify the dataset, hyper-parameters, training methods, models, etc.

## Requirement 
Pytorch >= 1.6.0

## Citing BayesFT


```BibTeX
@INPROCEEDINGS{9586115,  author={Ye, Nanyang and Mei, Jingbiao and Fang, Zhicheng and Zhang, Yuwen and Zhang, Ziqing and Wu, Huaying and Liang, Xiaoyao},  booktitle={2021 58th ACM/IEEE Design Automation Conference (DAC)},   title={BayesFT: Bayesian Optimization for Fault Tolerant Neural Network Architecture},   year={2021},  volume={},  number={},  pages={487-492},  doi={10.1109/DAC18074.2021.9586115}}
```





