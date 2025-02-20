# 合成雨滴

import cv2
import numpy as np
from skimage import exposure, io
import random
import matplotlib.pyplot as plt

# 生成噪声图像
def get_noise(img, value=10):
    noise = np.random.uniform(0, 256, img.shape[0:2])
    # 控制噪声水平，取浮点数，只保留最大的一部分作为噪声
    v = value * 0.01
    noise[np.where(noise < (256 - v))] = 0

    # 噪声做初次模糊
    k = np.array([[0, 0.1, 0],
                  [0.1, 8, 0.1],
                  [0, 0.1, 0]])

    noise = cv2.filter2D(noise, -1, k)
    # cv2.imshow('rain_noise',noise)
    # cv2.waitKey()
    return noise


# 将噪声加上运动模糊,模仿雨滴
def rain_blur(noise, length=10, angle=0,w=1):
    #这里由于对角阵自带45度的倾斜，逆时针为正，所以加了-45度的误差，保证开始为正
    trans = cv2.getRotationMatrix2D((length/2, length/2), angle-45, 1-length/100.0)
    dig = np.diag(np.ones(length))   #生成对焦矩阵
    k = cv2.warpAffine(dig, trans, (length, length))  #生成模糊核
    k = cv2.GaussianBlur(k,(w,w),0)    #高斯模糊这个旋转后的对角核，使得雨有宽度
    #k = k / length                         #是否归一化
    blurred = cv2.filter2D(noise, -1, k)    #用刚刚得到的旋转后的核，进行滤波
    #转换到0-255区间
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    # cv2.imshow('rain_effct',blurred)
    # cv2.waitKey()
    return blurred

# 输入雨滴噪声和图像
def alpha_rain(rain,img,beta = 0.8):
    #将二维雨噪声扩张为三维单通道
    #并与图像合成在一起形成带有alpha通道的4通道图像
    rain = np.expand_dims(rain,2)
    rain_effect = np.concatenate((img,rain),axis=2)  #add alpha channel
    # 对每个通道先保留雨滴噪声图对应的黑色（透明）部分，再叠加白色的雨滴噪声部分（有比例因子）
    rain_result = img.copy()    #拷贝一个掩膜
    rain = np.array(rain,dtype=np.float32)     #数据类型变为浮点数，后面要叠加，防止数组越界要用32位
    rain_result[:,:,0]= rain_result[:,:,0] * (255-rain[:,:,0])/255.0 + beta*rain[:,:,0]
    rain_result[:,:,1] = rain_result[:,:,1] * (255-rain[:,:,0])/255 + beta*rain[:,:,0]
    rain_result[:,:,2] = rain_result[:,:,2] * (255-rain[:,:,0])/255 + beta*rain[:,:,0]
    # 将图片调暗
    # rain_result = exposure.adjust_gamma(rain_result, 2)
    cv2.imshow('rain_effct_result',rain_result)
    cv2.waitKey()
    return rain_result



img = cv2.imread('./test/07.jpg')
# value大小控制雨滴数量，length大小控制雨水水痕长度，angle大小来控制雨下落的角度，w来控制雨点粗细程度
noise = get_noise(img,value=20)   # value:20-50
rain = rain_blur(noise,length=134, angle=15,w=7)   # length:125-129,132-134,140-142  angle:-45-45  w:5,9,15
rain_result = alpha_rain(rain,img,beta=0.9)    # beta:0.8-0.95
cv2.imwrite('./rain_test/07.png', rain_result)
# plt.figure()
#
# plt.subplot(211)
# b,g,r = cv2.split(img)
# img = cv2.merge((r,g,b))
# plt.imshow(img)
#
# plt.subplot(212)
# b,g,r = cv2.split(rain_result)
# rain_result = cv2.merge((r,g,b))
# plt.imshow(rain_result)
#
# plt.legend()
# plt.show()


