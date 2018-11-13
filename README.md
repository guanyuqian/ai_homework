产品缺陷检测

实验报告

关钰千 \|10142510128\| 人工智能基础 \| 2018年10月10日

实验目的
========

-   掌握人工智能相关基础知识

-   掌握人工智能中图像识别的基础知识

-   掌握机器学习编程方法

实验内容
========

输入输出定义
------------

输入规则：给定一组产品焊接图片

输出规则：区分输入中的合格和不合格产品，并指出不合格产品中的缺陷点

判定规则
--------

焊接成功：当几个焊点都焊接在两个工件的连接处时，说明该焊接成功

产品合格：一张输入的图片中的所有焊接都成功则称该产品合格

产品不合格：一张输入产品图片中出现一次以上焊接不成功则称该产品不合格

设计思想
========

检测思路
--------

把产品图片中的所有焊接分离，对每个焊接进行检测，最后根据每个焊接的判定结果输出。

特征点提取
----------

由于产品照片机位相似，是否合格的特征在一定的固定区域以内，故用简单的人工特征点放大，把产品图片的特征的区域分离出来进行检测，减少不相关干扰。

丰富数据集（可选）
------------------

由于实验给的数据集过少，所以这里对经过预处理的图像进行旋转、镜像等操作得到新的数据集，这可能会让训练结果更加好，但是最后使用原数据集的操作就已经达到100%的正确率（检测集较少的原因），没有更多数据表明图像处理丰富的数据集可以让训练结果更好，所以最后的实验就没有使用。

判定思路
--------

利用深度学习的方法，搭建深度学习框架，把根据上述思路进行预处理的图像作为数据集，训练出一个用于产品缺陷检测的神经网络。

实验环境
========

-   运行环境：Python

-   深度学习框架：PyTorch

-   图像处理库：PIL，Skimage

-   开发环境：Windows 10，GTX 1063，PyCharm，intel i5 7500

实验过程
========

工程目录
--------

1.  │  constant.py  

2.  │  gpu_train.py  

3.  │  image_preprocessing.py  

4.  │  net_last.pkl  

5.  │  net_no_validation_acc1.0.pkl  

6.  │  net_validation0.3_acc1.0.pkl  

7.  │  predict.py  

8.  ├─original_images  

9.    ├─check_images  

10.   ├─failure_images  

11.   └─success_images  

图像分类
--------

图片存于original_images文件夹内，其中check_images中放置我们的预测集，failure_images中放置不合格的产品图片，success_images中放置合格的产品图片。

标签处理
--------

图片按照标签命名，如图片中焊接分别为成功-失败-失败，则该图片命名为100
(x).jpg（x是整数，防止重复命名）。

常数类实现（constant.py）
-------------------------

鉴于编程的良好规范，常量都定义在constant.py中，并且进行大写和不可修改的规定。

图像预处理( image_preprocessing.py)
-----------------------------------

-   根据图片中的三个焊接大致区域，将图片分出三份

-   根据特征点位于焊接的大致区域，将焊接图片焊接点大致区域裁剪出来

-   采用 镜像\*180度翻转 丰富数据集  

神经网络训练（ gpu_train.py  ）
-------------------------------

输入数据处理：把数据打乱，分为70%的训练集和30%的测试集，由于数据集较少，所以预测集没有意义，就没有分离出预测集。

起初采用的是TensorFlow的MLP进行建模训练，最后得出的效果并不是很好，之后用了Pytorch的cnn进行建模训练。训练时用测试集得出的超参数如下代码


1.  EPOCH = 30\# train the training data n times

2.  BATCH_SIZE = 50  

3.  LR = 0.0001              \# learning rate  

4.  if_use_gpu = 1  

5.  **class** CNN(nn.Module):  

6.      **def** __init__(self):  

7.          super(CNN, self).__init__()  

8.          self.conv1 = nn.Sequential(         \# input shape (1, 28, 28)  

9.              nn.Conv2d(  

10.                 in_channels=1,              \# input height  

11.                 out_channels=16,            \# n_filters  

12.                 kernel_size=5,              \# filter size  

13.                 stride=1,                   \# filter movement/step  

14.                 padding=2,                 

15.             ),                              \# output shape (16, 28, 28)  

16.             nn.ReLU(),                      \# activation  

17.             nn.MaxPool2d(kernel_size=2),   

18.             nn.Dropout(0.2)  

19.         )  

20.         self.conv2 = nn.Sequential(         \# input shape (16, 14, 14)  

21.             nn.Conv2d(16, 32, 5, 1, 2),     \# output shape (32, 14, 14)  

22.             nn.ReLU(),                      \# activation  

23.             nn.MaxPool2d(2),                \# output shape (32, 7, 7)  

24.             nn.Dropout(0.5),  

25.         )  

26.         self.out = nn.Linear(32\*int(constant.WIDTH_PX/4)\*int(constant.HEIGHT_PX/4), 2)  

27.     **def** forward(self, x):  

28.         x = self.conv1(x)  

29.         x = self.conv2(x)  

30.         x = x.view(x.size(0), -1)           

31.         output = self.out(x)  

32.         **return** output, x    \# return x for visualization 

神经网络提升
------------

使用Pytorch的cnn模型，最后已经让验证集的准确率达到100%。为了让模型进一步提升，让验证集的那30%的数据也参与训练。由于最后的出来的结果始终都是100%，数据集太少，所以之前想扩展数据集的方式也成了可选方案，并未采用，因为无法验证扩展数据集是否可以训练出更好的神经网络。

产品检测（predict.py）
----------------------

把产品图片放入check_images 中（放入别的文件夹也可以，不过需要自行配置），然后运行程序，程序会加载之前训练好的神经网络，对预处理后的产品图片进行检测，最后把有缺陷的产品输出，并且用红框标记失败的焊接。

实验结果
========

经过实验过程中的神经网络训练，最后得到训练集*识别率*达到*100%*，识别时间100张图片，程序开始运行加载神经网络图片预处理缺陷检测检测结果导出，总共花费10S时间，即每张图片的*识别速度*是*0.1S*。实验的代码我放在了github上（<https://github.com/guanyuqian/ai_homework>），检测的过程如下图所示：

实验总结
========

关于方法选择
------------

在老师布置下作业的时候，考虑实现方法，就有传统的机器学习和深度学习两条路径，由于老师给的数据集太少，曾一度想用传统方法实现。之后拜读了Andrew
L. Beam的《[You can probably use deep learning even if your data isn't that
big](http://beamandrew.github.io/deeplearning/2017/06/04/deep_learning_works.html)》，觉得用深度学习试一试，最后没想到得出的结果还不错，让我更能理解了利用神经网络的深度学习的强大之处。

数据集大小的思考
----------------

关于数据集我之前会因为老师给的数据集太少而相关数据集通过变换而丰富的方法，虽然最后因为没有变换的数据集得出的神经网络就达标，但是我相信在面对更多的产品检测的情况，丰富数据集的方式还是很有效果的，这个就等今后的工作中验证了。

深度学习入门的思考
------------------

搭建深度学习的过程中也学习了“[莫烦python](https://morvanzhou.github.io/)”的深度学习框架搭建，让我意识到了深度学习的入门门槛是很低的，也许这就是深度学习现在那么火的原因吧。

深度学习的基础知识掌握
----------------------

之后在优化过程中的超参数的设置却让我十分为难，因为超参数的修改需要了解深度学习底层的基础知识，还有结合以往的深度学习经验。这样让我意识到深度学习这方向要想深入下去，还是得了解机器学习的基础知识，所以陈伟婷老师的“机器学习基础”还是很有必要学习的。