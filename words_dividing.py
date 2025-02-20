# 字符分割

import cv2
import copy
import numpy as np
import image_pps
import lp_location
import matplotlib.pyplot as plt
import rain_removal_test

filename = "./rain_test/08.png"   #7：黄 2:绿

# 根据设定的阈值和图片直方图，找出波峰，用于分隔字符
def find_waves(threshold, histogram):
    # print("阈值：",threshold)
    # print(histogram)
    up_point = -1 #上升点
    is_peak = False
    if histogram[0] > threshold:
        up_point = 0
        is_peak = True
    wave_peaks = []
    for i,x in enumerate(histogram):
        if is_peak and x < threshold:
            if i - up_point > 2:
                is_peak = False
                wave_peaks.append((up_point, i))
        elif not is_peak and x >= threshold:
            is_peak = True
            up_point = i
    if is_peak and up_point != -1 and i - up_point > 4:
        wave_peaks.append((up_point, i))
    # print(wave_peaks)
    return wave_peaks


# 车牌图像灰度化，确定上下边框，并去除上下无用的边缘部分，输出二值化图像
def remove_plate_upanddown_border(card_img):
    plate_gray_Arr = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
    ret, plate_binary_img = cv2.threshold( plate_gray_Arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU )
    row_histogram = np.sum(plate_binary_img, axis=1)   #数组的每一行求和
    row_min = np.min( row_histogram )
    row_average = np.sum(row_histogram) / plate_binary_img.shape[0]
    row_threshold = (row_min + row_average) / 2

    wave_peaks = find_waves(row_threshold, row_histogram)  # 根据水平直方图找出波峰
    if len(wave_peaks) == 0:
        print("peak less 0:")
        return 0

    wave = max(wave_peaks, key=lambda x:x[1]-x[0])
    max_wave_dis = wave[1] - wave[0]
    #判断是否是左侧车牌边缘，去除左侧车牌边缘
    if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis/3 and wave_peaks[0][0] == 0:
        wave_peaks.pop(0)

    #接下来挑选跨度最大的波峰
    wave_span = 0.0
    for wave_peak in wave_peaks:
        span = wave_peak[1]-wave_peak[0]
        if span > wave_span:
            wave_span = span
            selected_wave = wave_peak
    # 截取出去除上下边缘后的图像
    plate_binary_img = plate_binary_img[selected_wave[0] : selected_wave[1], :]
    # cv2.imshow("plate_binary_img", plate_binary_img)
    # cv2.waitKey(0)
    return plate_binary_img


# 分割车牌字符，输出字符图像
def divide_words(img):
    img = copy.deepcopy(img)
    # cv2.imshow("y_img", img)
    # cv2.waitKey(0)
    part_cards = []
    # 车牌灰度化+去除车牌上下边缘
    gray_img = remove_plate_upanddown_border(img)
    # cv2.imshow("re_b_img", gray_img)
    # cv2.waitKey(0)


    #查找垂直投影波峰
    row_num, col_num= gray_img.shape[:2]
    # #去掉车牌上下边缘1个像素，避免白边影响阈值判断
    gray_img = gray_img[1:row_num-1]
    y_histogram = np.sum(gray_img, axis=0)
    # # 画垂直投影直方图
    # x = np.arange(len(y_histogram))
    # fig, (ax1) = plt.subplots(1,1);
    # ax1.fill_between(x, 0, y_histogram)
    # plt.show()

    y_min = np.min(y_histogram)
    y_average = np.sum(y_histogram)/y_histogram.shape[0]
    y_threshold = (y_min + y_average)/10#U和0要求阈值偏小，否则U和0会被分成两半
    wave_peaks = find_waves(y_threshold, y_histogram)    # 找出波峰
    # for wave in wave_peaks:
    # 	cv2.line(gray_img, pt1=(wave[0], 5), pt2=(wave[1], 5), color=(0, 0, 255), thickness=2)
    # cv2.imshow("weak",gray_img)
    # cv2.waitKey()


    #车牌字符数应大于6
    if len(wave_peaks) <= 6:
        print("peak less 1:", len(wave_peaks))
        # continue

    wave = max(wave_peaks, key=lambda x:x[1]-x[0])
    max_wave_dis = wave[1] - wave[0]
    #判断是否是左侧车牌边缘
    if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis/2 and wave_peaks[0][0] < 4:    #改：原:max_wave_dis/3,  wave_peaks[0][0]==0
        wave_peaks.pop(0)
        # print("pop边缘")

    #判断组合分离汉字
    cur_dis = 0
    # 对于第一个间隔小于max_wave_dis * 0.6，第一个加第二个间隔之和大于max_wave_dis * 0.6的情况，将第一个与第二个间隔之和判断为组合汉字
    for i,wave in enumerate(wave_peaks):
        if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
            break
        else:
            cur_dis += wave[1] - wave[0]
    if i > 0:
        wave = (wave_peaks[0][0], wave_peaks[i][1])
        wave_peaks = wave_peaks[i+1:]
        wave_peaks.insert(0, wave)
        # print("组合")

    #去除车牌上的分隔点
    if wave_peaks[2]:
        point = wave_peaks[2]
        if point[1] - point[0] < max_wave_dis/3:
            point_img = gray_img[:,point[0]:point[1]]
            if np.mean(point_img) < 255/5:
                wave_peaks.pop(2)

        if len(wave_peaks) <= 6:
            print("peak less 2:", len(wave_peaks))
            # continue

    # 分割字符
    # print(wave_peaks)
    i = 0
    for wave in wave_peaks:
        if wave[0] == 0:   # 起点为0的字符
            cj = gray_img[:, wave[0]:wave[1]+1]
            part_cards.append(cj)
        elif wave[1] == col_num:  # 终点为最右侧的字符
            cj = gray_img[:, wave[0]-1:wave[1]]
            part_cards.append(cj)
        else:
            cj = gray_img[:, wave[0]-1:wave[1]+1]  # 其他字符（左右边界各扩展1）
            part_cards.append(cj)
        # 显示分割字符图像
    #     cv2.imshow('word'+str(i), cj)
    #     cv2.waitKey()
    #     i += 1
    # cv2.destroyAllWindows()

    return part_cards



# 字符分割模块测试
if __name__ == '__main__':

    in_img = image_pps.img_read(filename)
    pre_img = image_pps.img_resize(in_img)
    model = rain_removal_test.load_rainmodel()
    rimg = rain_removal_test.single_process(pre_img, model)
    tc_img = lp_location.cp_location(rimg)
    # cv2.imshow('carplate', tc_img)
    # cv2.waitKey(0)
    part_cards = divide_words(tc_img)

    # n: int = len(part_cards)
    # f = plt.figure()
    # for i in range(n):
    #     f.add_subplot(1, n, i + 1)
    #     plt.imshow(part_cards[i])
    #     plt.show(block=True)


