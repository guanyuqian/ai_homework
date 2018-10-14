# 用训练好的模型测试数据
from tensorflow import keras
from PIL import Image
import os
import numpy as np
from skimage.transform import resize

import constant


def get_images(rootdir):
    list_file = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    x = []
    for i in range(0, len(list_file)):
        path = os.path.join(rootdir, list_file[i])
        if os.path.isfile(path) and path[-4:] == '.jpg':
            x.append(path)
    return x


def predict_data(model,original_image_path,image_list):
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
        img = img.resize((constant.WIDTH_PX, constant.HEIGHT_PX))
        img = np.array(img)
        img = resize(img, (constant.WIDTH_PX, constant.HEIGHT_PX), mode='constant')
        img = img.reshape(1, -1)
        image_list.append(img)


if __name__ == '__main__':
    model = keras.models.load_model(constant.MODEL_PATH)
    x = get_images(constant.CHECK_IMAGE_PATH)
    image_list = []
    for img_path in x:
        predict_data(model, img_path,image_list)
    x = np.array(image_list)
    x = x.reshape(x.shape[0], -1)
    result = model.predict(x, batch_size=32)
    print(result)