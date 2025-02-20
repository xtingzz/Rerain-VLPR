# PreNet网络测试
# 部分代码来源于： https://github.com/csdwren/PReNet

import cv2
import os
import argparse
import glob
import numpy as np
import torch
from torch.autograd import Variable
from networks import *
import time
from math import exp
# from skimage.metrics import structural_similarity as ssim
from skimage.measure import compare_ssim
import matplotlib.pyplot as plt



# 导入相关库

parser = argparse.ArgumentParser(description="PReNet_Test")
parser.add_argument("--logdir", type=str, default="rain_model", help='path to model and log files')
parser.add_argument("--data_path", type=str, default="rain_test", help='path to training data')
parser.add_argument("--save_path", type=str, default="rain_results", help='path to save results')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
opt = parser.parse_args()

# if opt.use_GPU:
#     os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id



def batch_process(model_path):
    os.makedirs(opt.save_path, exist_ok=True)
    # Build model
    print('Loading model ...\n')
    model = PReNet(opt.recurrent_iter, opt.use_GPU)
    # print_network(model)
    # if opt.use_GPU:
    #     model = model.cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, model_path), map_location='cpu'))
    model.eval()

    time_test = 0
    count = 0
    for img_name in os.listdir(opt.data_path):
        if is_image(img_name):
            img_path = os.path.join(opt.data_path, img_name)

            # input image
            y = cv2.imread(img_path)
            b, g, r = cv2.split(y)
            y = cv2.merge([r, g, b])
            #y = cv2.resize(y, (int(500), int(500)), interpolation=cv2.INTER_CUBIC)

            y = normalize(np.float32(y))
            y = np.expand_dims(y.transpose(2, 0, 1), 0)
            y = Variable(torch.Tensor(y))

            # if opt.use_GPU:
            #     y = y.cuda()

            with torch.no_grad(): #
                # if opt.use_GPU:
                #     torch.cuda.synchronize()
                start_time = time.time()

                out, _ = model(y)
                out = torch.clamp(out, 0., 1.)

                # if opt.use_GPU:
                #     torch.cuda.synchronize()

                # calculate dealtime
                end_time = time.time()
                dur_time = end_time - start_time
                time_test += dur_time

                print(img_name, ': ', dur_time)

            if opt.use_GPU:
                save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())   #back to cpu
            else:
                save_out = np.uint8(255 * out.data.numpy().squeeze())

            save_out = save_out.transpose(1, 2, 0)
            b, g, r = cv2.split(save_out)
            save_out = cv2.merge([r, g, b])

            cv2.imwrite(os.path.join(opt.save_path, img_name), save_out)

            count += 1

    print('Avg. time:', time_test/count)


def load_rainmodel():
    # Build model
    print('Loading model ...\n')
    model = PReNet(opt.recurrent_iter, opt.use_GPU)
    print_network(model)
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'PreNet_latest.pth'), map_location='cpu'))
    model.eval()
    return model


# test to single image
def single_process(img, model):
    os.makedirs(opt.save_path, exist_ok=True)

    # time_test = 0

    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])

    img = normalize(np.float32(img))
    img = np.expand_dims(img.transpose(2, 0, 1), 0)
    img = Variable(torch.Tensor(img))

    with torch.no_grad():
        # start_time = time.time()
        out, _ = model(img)
        out = torch.clamp(out, 0., 1.)

        # if opt.use_GPU:
        #     torch.cuda.synchronize()

        # # calculate dealtime
        # end_time = time.time()
        # dur_time = end_time - start_time
        # time_test += dur_time
        # print(img_name, ': ', dur_time)

    # # ssim
    # b, g, r = cv2.split(oimg)
    # oimg = cv2.merge([r, g, b])
    # oimg = normalize(np.float32(oimg))
    # oimg = np.expand_dims(oimg.transpose(2, 0, 1), 0)
    # oimg = Variable(torch.Tensor(oimg))
    # loss_f = SSIM()
    # loss = loss_f(oimg, out)
    # print(f"[ test_SSIM_loss by SSIM = {loss:.5f}")
    # loss = loss_f(oimg, img)
    # print(f"[ in_SSIM_loss by SSIM = {loss:.5f}")

    if opt.use_GPU:
        save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())   #back to cpu
    else:
        save_out = np.uint8(255 * out.data.numpy().squeeze())

    save_out = save_out.transpose(1, 2, 0)
    b, g, r = cv2.split(save_out)
    save_out = cv2.merge([r, g, b])
    return save_out




def test_ssim(img, output):

    # # 法一：ssim
    # loss = ssim(img, output,multichannel=True)
    # loss = -loss
    # print(f"[ test_SSIM_loss by ssim = {loss:.5f}")
    # 法二：compare_ssim
    loss =  compare_ssim(img, output, multichannel = True)
    print(f"[ test_SSIM_loss by compare_ssim = {loss:.5f}")




def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def is_image(img_name):
    if img_name.endswith(".jpg") or img_name.endswith(".bmp") or img_name.endswith(".png"):
        return True
    else:
        return  False

def normalize(data):
    return data / 255.


# ssim
# SSIM损失函数实现
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


input_img_path = "./rain_test/02.png"
label_img_path = "./test/02.jpg"
model_path = "PreNet_latest.pth"


if __name__ == "__main__":
    # 单项测试
    img = cv2.imread(input_img_path)
    cv2.imshow('y',img)
    cv2.waitKey()
    norain_img = cv2.imread(label_img_path)
    model = load_rainmodel()
    rrain_img = single_process(img, model)
    cv2.imshow('result',rrain_img)
    cv2.waitKey()
    test_ssim(norain_img, rrain_img)


    plt.figure()
    plt.subplot(131)
    plt.title('norain_image')
    b,g,r = cv2.split(norain_img)
    norain_img = cv2.merge((r,g,b))
    plt.imshow(norain_img)

    plt.subplot(132)
    plt.title('rain_image')
    b,g,r = cv2.split(img)
    img = cv2.merge((r,g,b))
    plt.imshow(img)

    plt.subplot(133)
    plt.title('rerain_image')
    b,g,r = cv2.split(rrain_img)
    rain_result = cv2.merge((r,g,b))
    plt.imshow(rain_result)

    plt.legend()
    plt.show()

    # # 整体测试
    # batch_process(model_path)



