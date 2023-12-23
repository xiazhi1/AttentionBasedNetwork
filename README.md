# AttentionBasedNetwork

## 项目简介

该项目主要目的是实现基于注意力机制的深度学习模型优化，设计目标是开发出一个基于深度学习模型的应用软件，能实现对乳腺癌CT图片的分析功能。该项目主要基于[Attention_Branch_Network](https://github.com/machine-perception-robotics-group/attention_branch_network)

本项目基于华中科技大学电子信息与通信学院软件课设要求搭建，共由三位同学合作进行开发，主要分工如下：

[xiazhi1](https://github.com/xiazhi1) 负责算法的实现,模型的训练与测试，以及GUI与模型的连接测试

[TToooooom](https://github.com/TToooooom) 负责GUI界面的设计和收集数据集


[Lizicakeee](https://github.com/Lizicakeee) 负责收集训练数据集以及完成数据集的初始化

## 配置说明

运行如下指令配置环境
```
conda env create -f environment.yml
```

测试运行GUI.py，即可弹出GUI窗口，请点击选取图像选取图片然后进行识别，模型会返回输入图像的注意力图

训练阶段运行imagenet.py，训练指令请参考[Attention_Branch_Network](https://github.com/machine-perception-robotics-group/attention_branch_network)


```
# 一个训练示例
python3 imagenet.py -a resnet152 --data ../../dataset/imagenet_data/ --epochs 90 --schedule 31 61 --gamma 0.1 -c checkpoints/imagenet/resnet152 --gpu-id 4,5,6,7 --test-batch 100
# 一个测试示例
python3 imagenet.py -a resnet152 --data ../../../../dataset/imagenet_data/ --epochs 90 --schedule 31 61 --gamma 0.1 -c checkpoints/imagenet/resnet152 --gpu-id 4,5,6 --test-batch 10 --evaluate --resume checkpoints/imagenet/resnet152/model_best.pth.tar
```

经过验证，本模型在[cbis-ddsm](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)数据集上对乳腺癌CT图片的分类准确率能达到62%左右，请注意这里我们对数据集进行了额外处理，将其转换为imagenet数据集格式，处理代码请参考build_dataset.py

## 参考资料

1. [Attention_Branch_Network](https://github.com/machine-perception-robotics-group/attention_branch_network)
2. [cbis-ddsm](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)