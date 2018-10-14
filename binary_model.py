#训练神经网络
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import datetime
from PIL import Image
from skimage.transform import resize
import matplotlib.pyplot as plt


# 从CSV文件中读取数据，并返回2个数组。分别是自变量x和因变量y。方便TF计算模型。
import constant


def read_csv():
    zc_dataframe = pd.read_csv(constant.TRAIN_SET, sep=constant.CSV_SEP)
    x = []
    y = []
    for zc_index in zc_dataframe.index:
        zc_row = zc_dataframe.loc[zc_index]
        x.append(zc_row[constant.CSV_IMAGE_NAME])
        y.append(zc_row[constant.CSV_IMAGE_RESULT])
    return (x,y)

#获取多层感知器
def get_mlp():
    # 使用序列模型
    model = keras.Sequential()
    # 隐藏层
    model.add(keras.layers.Dense(512, activation='relu', input_shape=(684000,)))
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    # 输出层
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    # 显示一下模型结构
    model.summary()
    # tf.train.GradientDescentOptimizer(0.01) 由于keras保存model时只能保存keras的优化器，所以编译时必须使用keras的
    opt = tf.keras.optimizers.SGD()
    # 编译模型
    model.compile(optimizer=opt,
                  loss=keras.losses.binary_crossentropy,
                  metrics=[keras.metrics.binary_accuracy])
    return model


# 构建基于卷积神经网络的多类别分类模型
def get_cnn(n_classes):
    # 输入图片的维度
    img_rows, img_cols = 570, 1200
    # 卷积滤镜的个数
    nb_filters, nb_filters2 = 32, 64
    # 最大池化，池化核大小
    pool_size = (2, 2)
    # 卷积核大小
    kernel_size = (3, 3)
    # rgb彩色图输入规格 (img_rows, img_cols,3)
    input_shape = (img_rows, img_cols, 1)
    # 全连接层
    Dense = keras.layers.Dense
    # 卷积层
    Convolution2D = keras.layers.Convolution2D
    # 池化层
    MaxPooling2D = keras.layers.MaxPooling2D
    # 扁平层,多维的输入一维化，常用在从卷积层到全连接层的过渡
    Flatten = keras.layers.Flatten
    # 防止过拟合层，设置百分比来随机断开已连接的百分比的神经元
    Dropout = keras.layers.Dropout
    # 开始自定义组合神经网络结构
    model = keras.Sequential()
    model.add(Convolution2D(nb_filters, kernel_size, activation='relu', input_shape=input_shape))
    model.add(Convolution2D(nb_filters, kernel_size, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(nb_filters2, kernel_size, activation='relu'))
    model.add(Convolution2D(nb_filters2, kernel_size, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def use_cnn():
    batch_size = 128
    nb_classes = 10  # 分类数
    nb_epoch = 12  # 训练轮数

# 读取所有图片，随机选取70%作为训练集和30%作为验证集
def preprocessing_format_binary_data():
    x, y = read_csv()
    temp = np.array([x, y])
    temp = temp.transpose()
    np.random.shuffle(temp)
    x = list(temp[:, 0])
    y = list(temp[:, 1])
    y = [int(i) for i in y]
    width_px = 570
    height_px = 1200
    image_list = []
    for img_path in x:
        img = Image.open(img_path)
        img = img.resize((width_px, height_px))
        # img = np.array(plt.imread(img_path))
        img = np.array(img)
        img = resize(img, (width_px, height_px), mode='constant')
        img = img.reshape(1, -1)
        image_list.append(img)
    # 训练集
    train_data = np.array(image_list[0:int(len(image_list) * 0.7)])
    train_data = train_data.reshape(train_data.shape[0], -1)
    train_label = np.array(y[0:int(len(image_list) * 0.7)])
    train_label = train_label.reshape(train_label.shape[0], -1)
    # 验证集
    validation_data = np.array(image_list[int(len(image_list) * 0.7):len(image_list)])
    validation_data = validation_data.reshape(validation_data.shape[0], -1)
    validation_label = np.array(y[int(len(image_list) * 0.7):len(image_list)])
    validation_label = validation_label.reshape(validation_label.shape[0], -1)
    return train_data, train_label, validation_data, validation_label

if __name__ == '__main__':
    data, labels, val_data, val_labels  = preprocessing_format_binary_data()
    print('开始时间', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    # 传统全连接模型
    model = get_mlp()
    # 卷积模型
    # model = get_cnn(2)
    model.fit(data, labels, epochs=4, batch_size=32,
              validation_data=(val_data, val_labels))
    # model.fit(data, labpredictels, batch_size=32,epochs=5)
    # 保存模型
    print('训练结束时间', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    model.save(constant.MODEL_PATH)
    print('结束时间', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
# data, labels, val_data, val_labels = preprocessing_format_categorical_data()
