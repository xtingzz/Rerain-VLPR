# 车牌字符识别

import cv2
import train
# import copy
import numpy as np
import image_pps
import lp_location
import words_dividing
import rain_removal_test


SZ = 20  # 训练图片长宽
filename = "./rain_test/04.png"   #7：黄   2：绿
PROVINCE_START = 1000


class CardPredictor:
    # 加载模型
    def load_svm(self):
        #识别英文字母和数字
        self.model = train.SVM(C=1, gamma=0.5)#SVM(C=1, gamma=0.5)
        #识别中文
        self.modelchinese = train.SVM(C=1, gamma=0.5)#SVM(C=1, gamma=0.5)
        self.model.load("module\\svm.dat")
        self.modelchinese.load("module\\svmchinese.dat")
    # 字符识别
    def plate_recognation(self, part_cards):
        p = 0
        predict_result = []
        for i, part_card in enumerate(part_cards):
            #可能是固定车牌的铆钉
            if np.mean(part_card) < 255/5:
                # print("a point")
                p = p + 1
                continue
            part_card_old = part_card
            #w = abs(part_card.shape[1] - SZ)//2
            w = part_card.shape[1] // 3
            part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value = [0,0,0])
            part_card = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)
            # cv2.imshow("part_card", part_card)
            # cv2.waitKey(0)
            #cv2.imwrite("u.jpg", part_card)
            #part_card = deskew(part_card)
            part_card = train.preprocess_hog([part_card])#preprocess_hog([part_card])
            pt = i - p
            if pt == 0:
                resp = self.modelchinese.predict(part_card)#第一个字符调用中文svm模型
                charactor = train.provinces[int(resp[0]) - PROVINCE_START]
            else:
                resp = self.model.predict(part_card)#其他字符调用字母数字svm模型
                charactor = chr(resp[0])


            #判断最后一个数是否是车牌边缘，去除车牌边缘
            if charactor == "1" and i == len(part_cards)-1:
                if part_card_old.shape[0]/part_card_old.shape[1] >= 8:#假设车牌边缘被认为是1，1太细，认为是边缘
                    continue
            if pt > 6:   # 去除车牌规定字符数目以外的干扰
                continue

            predict_result.append(charactor)
            # print(charactor)

        return predict_result#识别到的字符、定位的车牌图像、车牌颜色


# 字符识别模块测试
if __name__ == '__main__':
    img = image_pps.img_read(filename)
    img = image_pps.img_resize(img)   # 限制大小
    model = rain_removal_test.load_rainmodel()
    rimg = rain_removal_test.single_process(img, model)
    car_tc_img = lp_location.cp_location(rimg)    # 车牌图像定位
    cv2.imshow("car_tc_img", car_tc_img)
    cv2.waitKey(0)
    part_cards = words_dividing.divide_words(car_tc_img)   # 车牌字符分割
    # 车牌字符识别
    c = CardPredictor()
    c.load_svm()    # 加载训练好的模型
    predict_result = c.plate_recognation(part_cards)
    print(predict_result)
