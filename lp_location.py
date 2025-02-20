# 车牌定位

import cv2
import copy
import numpy as np
import image_pps
import rain_removal_test



Min_Area = 1000  # 车牌区域允许最大面积
filename = "./rain_test/08.png"   #7：黄


# 查找可能区域，输出为找到的多个轮廓
def img_edge(img):
    img = copy.deepcopy(img)
    img = image_pps.img_gray(img)     # 先转为灰度图
    # 开运算后的图像与原图进行融合，去除小点、毛刺等噪声
    kernel = np.ones((25,25), np.uint8)
    img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)   # 开运算
    img = cv2.addWeighted(img, 1, img_opening, -1, 0)    #融合
    # cv2.imshow('fuse_image', img)   # 融合后的图像
    # key=cv2.waitKey()

    # 找到图像边缘
    ret, thresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  #二值化
    img = cv2.Canny(thresh_img, 100, 200)  #canny边缘检测
    # cv2.imshow('canny_image', img)   # canny边缘检测后的图像
    # key=cv2.waitKey()

     # 数学形态学处理：腐蚀（erode）和膨胀（dilate）
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
    #x方向进行闭运算
    img = cv2.dilate(img, kernelX)
    img = cv2.erode(img, kernelX)
    #y方向进行开运算
    img = cv2.erode(img, kernelY)
    img = cv2.dilate(img, kernelY)
    # cv2.imshow('math_image', img)   # 数学形态学处理后的图像
    # key=cv2.waitKey()

    # 由于部分图像得到的轮廓边缘不整齐，因此再进行一次膨胀操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    img = cv2.dilate(img, kernel, iterations=3)
    # cv2.imshow('ropen_image', img)   # 再膨胀后的图像
    # key=cv2.waitKey()
    # # 查找图像边缘整体形成的所有可能区域，车牌就在其中一个区域中
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# 根据比例排除非车牌的矩形区域
def relocate_byrate(img, plates):
    img = copy.deepcopy(img)
    # 先排除面积较小的区域
    temp1_plates = []
    for plate in plates:
        if cv2.contourArea(plate) > Min_Area:
            temp1_plates.append(plate)
            print(cv2.contourArea(plate))
    # 排除宽高比不符的区域
    temp2_plates = []
    for temp1_plate in temp1_plates:
        temp_rect = cv2.minAreaRect(temp1_plate)  # 框选 生成最小外接矩形 返回值（中心(x,y), (宽,高), 旋转角度）
        rect_width, rect_height = temp_rect[1]
        # 计算宽高比
        if rect_width < rect_height:
            rect_width, rect_height = rect_height, rect_width
        wh_ratio = rect_width / rect_height
        # 车牌正常情况下宽高比在2 - 5.5之间
        print("宽高比：", wh_ratio)
        if wh_ratio > 2 and wh_ratio < 5.5:
            temp2_plates.append(temp1_plate)   #选出符合宽高比的轮廓
            # rect_vertices = cv2.boxPoints(temp_rect)
            # rect_vertices = np.int0(rect_vertices)
    # 框选并画出矩形区域
    for temp2_plate in temp2_plates:
        row_min,col_min = np.min(temp2_plate[:,0,:],axis=0)
        row_max, col_max = np.max(temp2_plate[:, 0, :], axis=0)
        cv2.rectangle(img, (row_min,col_min), (row_max, col_max), (0,255,0), 2)  # 画出矩形区域
        # card_img = img[col_min:col_max,row_min:row_max,:]  # 截取出矩形区域
    # cv2.imshow("rect_img", img)
    # cv2.waitKey(0)

    return temp2_plates

# 根据颜色排除非车牌的矩形区域
def relocate_bycolor(img, temp_plates):
    img = copy.deepcopy(img)
    # 用颜色识别出车牌区域
    max_mean = 0
    r = []
    for temp_plate in temp_plates:
        # 截取出矩形区域
        row_min,col_min = np.min(temp_plate[:,0,:],axis=0)
        row_max, col_max = np.max(temp_plate[:,0, :], axis=0)
        # cv2.rectangle(img, (row_min,col_min), (row_max, col_max), (0,255,0), 2)  # 画出矩形区域
        block = img[col_min:col_max,row_min:row_max,:]  # 截取出矩形区域
        # 从rgb颜色域转换到hsv颜色域
        hsv =  cv2.cvtColor(block, cv2.COLOR_BGR2HSV)
        # 寻找蓝色车牌区域
        low = np.array([100, 43, 46])
        up = np.array([140, 255, 255])
        # 将在蓝色范围外的颜色像素值置为0，范围内的置为255
        result = cv2.inRange(hsv, low, up)
        # 用计算均值的方式找蓝色最多的区块
        mean = cv2.mean(result)
        # print(mean)   # 每个可能矩形块的均值
        if mean[0] > max_mean:
            max_mean = mean[0]
            car_col_min = col_min
            car_row_min = row_min
            car_col_max = col_max
            car_row_max = row_max
            car_plate = temp_plate
    # 画出识别结果
    # print("最大均值：",max_mean)
    # cv2.rectangle(img, (car_row_min,car_col_min), (car_row_max, car_col_max), (0,255,0), 2)  # 画出车牌矩形区域
    # cv2.imshow("rect_img", img)
    # cv2.waitKey(0)
    car_img = img[car_col_min:car_col_max,car_row_min:car_row_max,:]  # 截取出车牌区域
    # cv2.imshow("car_plate", car_img)
    # cv2.waitKey(0)
    return car_img, car_plate


# 对倾斜车牌进行校正
def tilt_correction(img, car_img, car_plate):
    img = copy.deepcopy(img)
    # 矩形区域可能是倾斜的矩形，需要矫正，便于字符分割
    rect = cv2.minAreaRect(car_plate)  # 框选 生成最小外接矩形 返回值（中心(x,y), (宽,高), 旋转角度）
    print("倾斜角度：", rect[2])
    if 0 < abs(rect[2]) < 5 or 85 < abs(rect[2]) < 95:  # 创造角度，使得左、高、右、低拿到正确的值
        # angle = 1
        print("不需要校正")
        tc_img = car_img
        # cv2.imshow("tcw_plate", tc_img)
        # cv2.waitKey(0)
    else:
        print("需要校正")
        angle = rect[2]
        rect = (rect[0], (rect[1][0] + 5, rect[1][1] + 5), angle)  # 扩大范围，避免车牌边缘被排除

        box = cv2.boxPoints(rect)  # 获取矩形四个顶点
        pic_hight, pic_width = img.shape[:2]   # 获取图片大小
        heigth_point = right_point = [0, 0]
        left_point = low_point = [pic_width, pic_hight]
        for point in box:
            if left_point[0] > point[0]:
                left_point = point
            if low_point[1] > point[1]:
                low_point = point
            if heigth_point[1] < point[1]:
                heigth_point = point
            if right_point[0] < point[0]:
                right_point = point

        if left_point[1] <= right_point[1]:  # 正角度
            new_right_point = [right_point[0], heigth_point[1]]
            pts2 = np.float32([left_point, heigth_point, new_right_point])  # 字符只是高度需要改变
            pts1 = np.float32([left_point, heigth_point, right_point])
            M = cv2.getAffineTransform(pts1, pts2)
            dst = cv2.warpAffine(img, M, (pic_width, pic_hight))
            point_limit(new_right_point)
            point_limit(heigth_point)
            point_limit(left_point)
            tc_img = dst[int(left_point[1]):int(heigth_point[1]), int(left_point[0]):int(new_right_point[0])]
            # cv2.imshow("tcr_img", tc_img)
            # cv2.waitKey(0)
        elif left_point[1] > right_point[1]:  # 负角度
            new_left_point = [left_point[0], heigth_point[1]]
            pts2 = np.float32([new_left_point, heigth_point, right_point])  # 字符只是高度需要改变
            pts1 = np.float32([left_point, heigth_point, right_point])
            M = cv2.getAffineTransform(pts1, pts2)
            dst = cv2.warpAffine(img, M, (pic_width, pic_hight))
            point_limit(right_point)
            point_limit(heigth_point)
            point_limit(new_left_point)
            tc_img = dst[int(right_point[1]):int(heigth_point[1]), int(new_left_point[0]):int(right_point[0])]
            # cv2.imshow("tcn_img", tc_img)
            # cv2.waitKey(0)

    return tc_img

def point_limit(point):
    if point[0] < 0:
        point[0] = 0
    if point[1] < 0:
        point[1] = 0

def cp_location(img):
    contours = img_edge(img)
    car_plates = relocate_byrate(img, contours)
    car_img, car_plate = relocate_bycolor(img, car_plates)
    tc_img = tilt_correction(img, car_img, car_plate)
    return tc_img


# 定位模块测试
if __name__ == '__main__':
    in_img = image_pps.img_read(filename)
    pre_img = image_pps.img_resize(in_img)
    model = rain_removal_test.load_rainmodel()
    rimg = rain_removal_test.single_process(pre_img, model)
    pre_img = image_pps.img_resize(rimg)
    tc_img = cp_location(pre_img)
    # cv2.imshow("car_tc_img", tc_img)
    # cv2.waitKey(0)

