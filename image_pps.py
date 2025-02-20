# 图像预处理

import cv2

MAX_WIDTH = 800  # 原始图片最大宽度
MAX_HEIGHT = 600  # 原始图片最大高度
blur = 3   # 高斯内核大小
filename = "./test/4.jpg"



# 读入图片
def img_read(filename):
    img = cv2.imread(filename)  # np.ndarray BGR uint8
    # cv2.imshow('input_image', img)   # 显示原图
    # key=cv2.waitKey()
    return img


# 图像尺寸变换
def img_resize(img):
    pic_hight, pic_width = img.shape[:2]   # 获取图片大小
    # print(img.shape[:2])
    if pic_width > MAX_WIDTH:   # 限制大小
        resize_rate = MAX_WIDTH / pic_width
        img = cv2.resize(img, (MAX_WIDTH, int(pic_hight * resize_rate)), interpolation=cv2.INTER_AREA)
    # print(img.shape[:2])
    if pic_hight > MAX_HEIGHT:   # 限制大小
        resize_rate = MAX_HEIGHT / pic_hight
        img = cv2.resize(img, (int(pic_width * resize_rate), MAX_HEIGHT), interpolation=cv2.INTER_AREA)
    # print(img.shape[:2])
    return img

# 高斯去噪
def img_GaussianBlur(img):
    img = cv2.GaussianBlur(img, (blur, blur), 0, 0, cv2.BORDER_DEFAULT)  # 高斯滤波
    # cv2.imshow('input_image', img)   # 显示原图
    # key=cv2.waitKey()
    return img

# 图像灰度化
def img_gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('input_image', img)   # 显示原图
    # key=cv2.waitKey()
    return img


# img = img_read(filename)
# img = img_resize(img)
# img1 = img_GaussianBlur(img)
# img2 = img_gray(img)
