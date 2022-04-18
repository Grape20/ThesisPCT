# Official Implementation of PCT
## Prerequisites
- python >= 3.9.7

Please make sure you have the following libraries installed:
- numpy>=1.20.3
- pandas>=1.3.4
- torch>=1.10.2
- torchvision>=0.11.3

## Datasets
- [Office-31](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/)
- [Office-Home](https://www.hemanthdv.org/OfficeHome-Dataset/)
- [DomainNet](http://ai.bu.edu/M3SDA/) 

We provide direct download links in the script. However, for file larger than 100 MB (OfficeHome - Art and RealWorld), please manually download them from the following links--[Art](https://drive.google.com/file/d/18_F4TMEwP3yJcxTwhgH3FFc7OYtSJxAj/view?usp=sharing) and [RealWorld](https://drive.google.com/file/d/1xZPldApUAkx4qzsRIi00qfnzeX13HWCe/view?usp=sharing)--and extract them appropriately.

## Usage
- ` ` - data folder path  
- `-s` - source domain name  
- `-t` - target domain name  
- `-a` - architecture of feature extractor, default resnet18
- `--epochs` - number of epochs in each experiment 
- `-b` - batch size in source domain
- `--bs_tgt` - batch size in target domain 
- `-p` - print frequency(print for p batches)
- `-i` - number of batches in each epoch 
- `--trade-off` - trade-off hyper-parameter between classification loss and bi-directional transport loss, that is $\lambda_t$, default 1, higher $\lambda_t$ transport loss weights more   
- `--s_par` - trade-off hyper-parameter between two directional transport loss, that is $\lambda_b$, default 0.5, higher $\lambda_b$ transport $t\to\mu$ loss weights more
- `-beta` - learning rate/ momentum parameter to learn proportions in the target domain ( `beta=0` corresponds to using a uniform prior), default 0.001
- `-e` - number of experiments
- `--auto_bs` - auto batch size if True, default False 
- `--ratio_ts` - the ratio of target batch size to source when auto batch size, default 3

Below, we provide example commands to run our method.
```shell script
# Train on Office-31 Amazon -> Webcam task using ResNet18.
# Assume you put the datasets under the path `../data/office-31

python Proto_DA-ours/proto.py data/office31 -s A -t D --epochs 32 -i 10 -p 5 --auto_bs True

```

## Citation
We adapt our code base from the v0.1 of [PCT](https://github.com/korawat-tanwisuth/Proto_DA).

## PCT

> @inproceedings{tanwisuth2021prototype,  
>  title={A Prototype-Oriented Framework for Unsupervised Domain Adaptation},  
>  author={Korawat Tanwisuth and Xinjie Fan and Huangjie Zheng and Shujian Zhang and Hao Zhang and Bo Chen and Mingyuan Zhou},  
> booktitle = {NeurIPS 2021: Neural Information Processing Systems},   
> month={Dec.},  
> Note = {(the first three authors contributed equally)},  
> year = {2021}  
> }  

