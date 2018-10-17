import os

import torch.nn as nn
import torch
from PIL import Image
from skimage.transform import resize
from skimage import io,color
import numpy as np
import pandas as pd
import constant
import datetime

from image_preprocessing import generate_img_name


def get_images(rootdir):
    list_file = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    x = []
    for i in range(0, len(list_file)):
        path = os.path.join(rootdir, list_file[i])
        if os.path.isfile(path) and path[-4:] == '.jpg':
            x.append(path)
    return x


def predict_data(original_image_path,image_list):
    arr = original_image_path[-20 :-4]
    # 打开原始图片.jpg
    im = Image.open(original_image_path)
    # 左侧图片切割
    left_area = (constant.DX0, constant.DY0, constant.DX1, constant.DY1)
    # 中间图片切割
    middle_area = (constant.DX1, constant.DY0, constant.DX2, constant.DY1)
    # 右侧图片切割
    right_area = (constant.DX2, constant.DY0, constant.DX3, constant.DY1)
    x = [im.crop(left_area),im.crop(middle_area),im.crop(right_area)]
    for img in x:
        name = generate_img_name(constant.CHECK_IMAGE_PATH)
        img.save(name)
        rgb = io.imread(name)

        img = np.array(img)
        rgb = color.rgb2gray(rgb)
        rgb = resize(rgb, (570, 1200))
        image_list.append(rgb)


def data_preprocessing(rootdir):

    image_list = []
    origin_images=get_images(rootdir)
    for img_path in origin_images:
            rgb = io.imread(img_path)
            gray = color.rgb2gray(rgb)
            img = resize(gray, (570, 1200))
            image_list.append(img)

    # 70%图像被当做训练集
    train_data = np.array(image_list)
    # 查看矩阵或者数组的维数
    train_label = np.array(constant.PREDICT_SUCC_LABEL)
    train_label = train_label.reshape(train_label.shape[0], -1)
    train_data = torch.from_numpy(train_data).float()
    train_label = torch.from_numpy(train_label).long()

    return train_data,train_label
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 142 * 300, 2)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x    # return x for visualization

if __name__=='__main__':
    predict_data('./original_images/success_images/20180002200030_64_P45_75011772607501177260O02101802994000_9710a233dc8f4c3784bbb10dc119c9ce.jpg',[])


    print('开始时间', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    cnn = CNN()
    cnn.load_state_dict(torch.load('net.pkl'))
    test_x,test_y = data_preprocessing(constant.ORIGINAL_SUCC_IMAGE_PATH)
    true = 0
    amount = 0
    for i,x in enumerate(test_x):
        x.unsqueeze_(0)
        x.unsqueeze_(0)
        test_output = cnn(x)[0]
        pre_y = torch.max(test_output, 1)[1].data.squeeze().numpy()
        if pre_y==test_y[i].numpy():
            true += 1
        else:
            print(i)
        amount += 1
    acc = true/amount
    print(acc)
    print('训练结束时间', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))