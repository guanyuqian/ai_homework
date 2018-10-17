import math

import torch.nn as nn
import torch

import constant
import datetime

from gpu_train import CNN
from image_preprocessing import generate_img_name, data_preprocessing, predict_data_preprocessing


if __name__ == '__main__':

    print('开始时间', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    cnn = CNN()
    cnn = cnn.cuda()
    cnn.load_state_dict(torch.load(constant.LAST_MODEL_PATH))
    test_x, test_y ,images_name= predict_data_preprocessing(constant.PREDICT_IMAGES_PATH_LIST)
    true = 0
    amount = 0
    for i, x in enumerate(test_x):
        x.unsqueeze_(0)
        x.unsqueeze_(0)
        x = x.cuda()
        test_output = cnn(x)[0]
        pre_y = torch.max(test_output, 1)[1].data.cpu().squeeze().numpy()
        if pre_y == test_y[i].numpy():
            true += 1
        else:
            print("F",images_name[i])
        amount += 1
    acc = true / amount
    print(acc)
    print('训练结束时间', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
