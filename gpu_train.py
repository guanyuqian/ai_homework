import torch
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd
import numpy as np
from skimage import io,color
from skimage.transform import resize

import constant
from image_preprocessing import data_preprocessing, predict_data_preprocessing

EPOCH = 30# train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.0001              # learning rate
if_use_gpu = 1

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
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
            nn.Dropout(0.5),
        )
        self.out = nn.Linear(32*int(constant.WIDTH_PX/4)*int(constant.HEIGHT_PX/4), 2)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x    # return x for visualization


if __name__ == '__main__':
    cnn = CNN()
    if if_use_gpu:
        cnn = cnn.cuda()
    print(cnn)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters

    # train_x,train_y,test_x,test_y= data_preprocessing(constant.TRAIN_IMAGES_PATH_LIST)
    # print(len(train_x),len(test_x))
    train_x, train_y, images_name = predict_data_preprocessing(constant.TRAIN_IMAGES_PATH_LIST)

    # optimizer
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    # loss_fun
    loss_func = nn.CrossEntropyLoss()
    # training loop
    for epoch in range(EPOCH):
        for step, x in enumerate(train_x):
            x.unsqueeze_(0)
            x.unsqueeze_(0)
            batch_x = Variable(x)
            batch_y = Variable(train_y[step])
            if if_use_gpu:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
            # 输入训练数据
            output = cnn(batch_x)[0]
            # 计算误差
            loss = loss_func(output, batch_y)
            # 清空上一次梯度
            optimizer.zero_grad()
             # 误差反向传递
            loss.backward()
            # 优化器参数更新
            optimizer.step()
            # if step % 50 == 0:
            #     amount = 0
            #     true = 0
            #     y1 = []
            #     y2 = []
            #     for i,x in enumerate(test_x):
            #         x.unsqueeze_(0)
            #         x.unsqueeze_(0)
            #         if if_use_gpu:
            #             x = x.cuda()
            #         test_output= cnn(x)[0]
            #         pred_y = torch.max(test_output, 1)[1].data.cpu().squeeze().numpy()
            #         if(pred_y == test_y[i].numpy()):
            #             true += 1
            #         amount += 1
            #     accuracy = true/amount
            #     print('Epoch: ', epoch, '| train loss: %.8f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)
    torch.save(cnn.state_dict(),constant.LAST_MODEL_PATH)