# 系统界面显示及运行
# 主函数


import tkinter as tk
from tkinter.filedialog import *
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
# import threading
import time
import numpy as np

import train
import image_pps
import lp_location
import words_dividing
import word_regn
import rain_removal_test

class Surface(ttk.Frame):
    pic_path = ""
    viewhigh = 400
    viewwide = 400
    update_time = 0
    thread = None
    thread_run = False
    # camera = None
    # color_transform = {"blue": ("蓝牌", "#6666FF")}   # 颜色转换字典

    def __init__(self, win):    # 初始化窗口
        ttk.Frame.__init__(self, win)
        frame_left = ttk.Frame(self)
        frame_right1 = ttk.Frame(self)
        frame_right2 = ttk.Frame(self)
        win.title("基于去雨算法的车牌识别系统")   # 窗口标题
        # win.state("zoomed")   # 窗口的默认状态，zoomed表示全屏，缺省为非全屏
        self.pack(fill=tk.BOTH, expand=tk.YES, padx="5", pady="5")
        frame_left.pack(side=LEFT, expand=1, fill=BOTH)
        frame_right1.pack(side=TOP, expand=1, fill=tk.Y)
        frame_right2.pack(side=RIGHT, expand=0)

        # 初始化Label控件
        ttk.Label(frame_left, text='原始图像：').grid(column=0, row=0, sticky=tk.W)
        ttk.Label(frame_left, text='定位区域：').grid(column=0, row=2, sticky=tk.W)
        ttk.Label(frame_right1, text='去雨图像：').grid(column=0, row=0, sticky=tk.W)
        ttk.Label(frame_right1, text='识别结果：').grid(column=0, row=2, sticky=tk.W)
        # 初始化Button控件
        from_pic_ctl = ttk.Button(frame_right2, text="来自图片", width=20, command=self.from_pic)


        self.image_ctl = ttk.Label(frame_left)   # 原始图像框
        self.image_ctl.grid(column=0, row=1, sticky=tk.W)
        self.roi_ctl = ttk.Label(frame_left)     # 车牌定位图像框
        self.roi_ctl.grid(column=0, row=3, sticky=tk.W)
        self.re_ctl = ttk.Label(frame_right1)     # 去雨图像框
        self.re_ctl.grid(column=0, row=1, sticky=tk.W)
        # ttk.Label(frame_right1, text='识别结果：').grid(column=0, row=2, sticky=tk.W)
        self.r_ctl = ttk.Label(frame_right1, text="")     # 字符识别结果框
        self.r_ctl.grid(column=0, row=3, sticky=tk.W)
        from_pic_ctl.pack(anchor="se", pady="5")    # button框
        self.predictor = Carplater()
        # self.predictor.train_svm()

    def get_imgtk(self, img_bgr):   # 处理读入的图像
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # 将BGR图转换为RGB图
        img = Image.fromarray(img)     # 实现array到image的转换
        imgtk = ImageTk.PhotoImage(image=img)   # 创建一个Tkinter兼容的照片图像
        # 获取读入图像的宽高
        wide = imgtk.width()
        high = imgtk.height()
        # 如果读入的图像大小超过显示范围，则对图像进行按比例的resize
        if wide > self.viewwide or high > self.viewhigh:
            wide_factor = self.viewwide / wide
            high_factor = self.viewhigh / high
            factor = min(wide_factor, high_factor)
            wide = int(wide * factor)
            if wide <= 0:
                wide = 1
            high = int(high * factor)
            if high <= 0:
                high = 1
            img = img.resize((wide, high), Image.ANTIALIAS)
            imgtk = ImageTk.PhotoImage(image=img)
        return imgtk

    def from_pic(self):     # 从图片识别车牌
        self.thread_run = False
        self.pic_path = askopenfilename(title="选择识别图片", filetypes=[("jpg图片", "*.jpg"), ("png图片", "*.png")])
        if self.pic_path:
            img_bgr = cv2.imdecode(np.fromfile(self.pic_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.imgtk = self.get_imgtk(img_bgr)    # 读取图片及尺寸限制
            self.image_ctl.configure(image=self.imgtk)      # 配置窗口控件（Widgets）
            result, rrain_img, car_img = self.predictor.predict(img_bgr)     # 识别图像中的车牌信息
            self.show_roi(result, rrain_img, car_img)    # 在窗口显示识别的车牌信息

    def show_roi(self, result, rimg,img):  # 显示车牌图像
        if result:   # 如果识别出车牌
            # 显示车牌图像
            self.imgtk_re = self.get_imgtk(rimg)    # 读取图片及尺寸限制
            self.re_ctl.configure(image=self.imgtk_re, state='enable')      # 配置窗口控件（Widgets）
            # re = cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)
            # re = Image.fromarray(re)
            # self.imgtk_re = ImageTk.PhotoImage(image=re)
            # self.re_ctl.configure(image=self.imgtk_re, state='enable')   # 显示车牌定位图像
            # 显示车牌图像
            roi = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            roi = Image.fromarray(roi)
            self.imgtk_roi = ImageTk.PhotoImage(image=roi)
            self.roi_ctl.configure(image=self.imgtk_roi, state='enable')   # 显示车牌定位图像
            # 显示识别结果
            self.r_ctl.configure(text=str(result))   # 分字符显示车牌号
            self.update_time = time.time()
        elif self.update_time + 8 < time.time():    # 如果未识别出车牌则不显示该部分
            self.re_ctl.configure(state='disabled')
            self.roi_ctl.configure(state='disabled')
            self.r_ctl.configure(text="")
            # self.color_ctl.configure(state='disabled')

    @staticmethod
    def vedio_thread(self):
        self.thread_run = True
        predict_time = time.time()
        while self.thread_run:
            _, img_bgr = self.camera.read()
            self.imgtk = self.get_imgtk(img_bgr)
            self.image_ctl.configure(image=self.imgtk)
            if time.time() - predict_time > 2:
                r, roi, color = self.predictor.predict(img_bgr)
                self.show_roi(r, roi, color)
                predict_time = time.time()
        print("run end")


class Carplater:
    def __init__(self):
        # 去雨模型加载
        self.model = rain_removal_test.load_rainmodel()
        # 车牌识别模型加载
        self.c = word_regn.CardPredictor()
        self.c.load_svm()    # 加载训练好的svm模型
    def train_svm(self):
        train.train_svm()
    def predict(self, img):
        img = image_pps.img_resize(img)   # 限制大小
        rrain_img = rain_removal_test.single_process(img, self.model)
        car_tc_img = lp_location.cp_location(rrain_img)    # 车牌图像定位
        # cv2.imshow("car_tc_img", car_tc_img)
        # cv2.waitKey(0)
        part_cards = words_dividing.divide_words(car_tc_img)   # 车牌字符分割
        # 车牌字符识别
        predict_result = self.c.plate_recognation(part_cards)

        return predict_result, rrain_img, car_tc_img    # 返回识别结果，车牌图像





def close_window():     # 关闭窗口
    print("destroy")
    if surface.thread_run:
        surface.thread_run = False
        surface.thread.join(2.0)
    win.destroy()


if __name__ == '__main__':
    win = tk.Tk()   # 生成初始化窗口
    # win.configure(bg='blue')
    surface = Surface(win)  # 新建一个自定义类Surface（在本文件内定义），作用有窗口的初始化和功能的定义
    win.protocol('WM_DELETE_WINDOW', close_window)  # 定义窗口关闭事件
    win.mainloop()  # 进入窗口消息循环
