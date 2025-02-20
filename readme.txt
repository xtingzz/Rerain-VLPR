课题：基于去雨算法的车牌识别系统
作者：钟希婷
实现功能：图像去雨+车牌识别
实现原理：基于训练好的PReNet模型进行图像去雨，车牌识别流程依次包括有图像预处理、车牌定位、车牌字符分割和基于SVM模型的字符识别。


系统运行：运行surface.py
车牌去雨数据集：见https://www.kaggle.com/code/xitting/ccpd-addrain-datasets/output

注：
PReNet模型是在Kaggle平台上进行训练的，训练代码：prenet-ccpd.ipynb；
SVM模型训练代码：train.py

