import constant
import os, uuid
from PIL import Image
from skimage.transform import resize
from skimage import io, color
import numpy as np
import torch

'''
生成唯一图片名称
'''


def generate_img_name(save_path):
    id = str(uuid.uuid1())
    return save_path + id + ".jpg"


'''
获取路径下面所有jpg图片
'''


def get_images(rootdir):
    list_file = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    x = []
    for i in range(0, len(list_file)):
        path = os.path.join(rootdir, list_file[i])
        path=path.replace('\\', '/')
        if os.path.isfile(path) and path[-4:] == '.jpg':
            x.append(path)
    return x


'''
    将图片分割成三份并根据图片名称生成相应label
'''


def cut_img(original_image_path, image_list, images_label, images_name, need_synthesis_images):
    # 根据文件名打标记
    images_label.append(int(original_image_path.split("/")[-1][0]))
    images_label.append(int(original_image_path.split("/")[-1][1]))
    images_label.append(int(original_image_path.split("/")[-1][2]))
    # 打开原始图片.jpg
    im = Image.open(original_image_path)
    # 左侧图片切割
    left_area = (constant.DX0 + constant.GAP, constant.DY0, constant.DX1 - constant.GAP, constant.DY1)
    # 中间图片切割
    middle_area = (constant.DX1 + constant.GAP, constant.DY0, constant.DX2 - constant.GAP, constant.DY1)
    # 右侧图片切割
    right_area = (constant.DX2 + constant.GAP, constant.DY0, constant.DX3 - constant.GAP, constant.DY1)
    image_device = [im.crop(left_area), im.crop(middle_area), im.crop(right_area)]

    # 添加文件名字
    images_name.append(os.path.basename(original_image_path) +constant.LEFT)
    images_name.append(os.path.basename(original_image_path) +constant.MIDDLE)
    images_name.append(os.path.basename(original_image_path) + constant.RIGHT)

    # if (need_synthesis_images):
    #     synthesis_images(image_device, images_label, images_name, original_image_path)

    for img in image_device:
        # img.save(generate_img_name(constant.SYNTHESIS_IMAGE_PATH));
        img = np.array(img)
        img = color.rgb2gray(img)
        img = resize(img, (constant.WIDTH_PX, constant.HEIGHT_PX))
        image_list.append(img)

    return image_list, images_label

'''
    图像丰富处理
'''
def synthesis_images(image_device, images_label, images_name, original_image_path):
    #  图像丰富处理
    image_device.append(image_device[0].transpose(Image.ROTATE_180))
    image_device.append(image_device[1].transpose(Image.ROTATE_180))
    image_device.append(image_device[2].transpose(Image.ROTATE_180))
    #  图像丰富处理
    image_device.append(image_device[0].transpose(Image.FLIP_LEFT_RIGHT))
    image_device.append(image_device[1].transpose(Image.FLIP_LEFT_RIGHT))
    image_device.append(image_device[2].transpose(Image.FLIP_LEFT_RIGHT))
    image_device.append(image_device[3].transpose(Image.FLIP_LEFT_RIGHT))
    image_device.append(image_device[4].transpose(Image.FLIP_LEFT_RIGHT))
    image_device.append(image_device[5].transpose(Image.FLIP_LEFT_RIGHT))
    # 添加标签文件名字
    images_label.append(int(original_image_path.split("/")[-1][0]))
    images_label.append(int(original_image_path.split("/")[-1][1]))
    images_label.append(int(original_image_path.split("/")[-1][2]))

    images_label.append(int(original_image_path.split("/")[-1][0]))
    images_label.append(int(original_image_path.split("/")[-1][1]))
    images_label.append(int(original_image_path.split("/")[-1][2]))
    images_label.append(int(original_image_path.split("/")[-1][0]))
    images_label.append(int(original_image_path.split("/")[-1][1]))
    images_label.append(int(original_image_path.split("/")[-1][2]))
    # 添加文件名字
    images_name.append(os.path.basename(original_image_path) + " left ROTATE_180")
    images_name.append(os.path.basename(original_image_path) + " middle ROTATE_180")
    images_name.append(os.path.basename(original_image_path) + " right ROTATE_180")

    images_name.append(os.path.basename(original_image_path) + " left FLIP_LEFT_RIGHT")
    images_name.append(os.path.basename(original_image_path) + " middle FLIP_LEFT_RIGHT")
    images_name.append(os.path.basename(original_image_path) + " right FLIP_LEFT_RIGHT")
    images_name.append(os.path.basename(original_image_path) + " left ROTATE_180 FLIP_LEFT_RIGHT")
    images_name.append(os.path.basename(original_image_path) + " middle ROTATE_180 FLIP_LEFT_RIGHT")
    images_name.append(os.path.basename(original_image_path) + " right ROTATE_180 FLIP_LEFT_RIGHT")

'''
    训练预处理数据
'''
def data_preprocessing(images_path_list):
    image_list = []
    images_label = []

    image_list, images_label, images_name = get_imags_and_labels(image_list, images_label, images_path_list, 1, 1)
    print(image_list[0].shape)
    # 70%图像被当做训练集
    train_data = np.array(image_list[0:int(len(image_list) * 0.7)])
    train_data = torch.from_numpy(train_data).float()
    train_label = np.array(images_label[0:int(len(images_label) * 0.7)])
    train_label = train_label.reshape(train_label.shape[0], -1)
    train_label = torch.from_numpy(train_label).long()

    # 验证集 剩下30%
    validation_data = np.array(image_list[int(len(image_list) * 0.7):len(image_list)])
    validation_data = torch.from_numpy(validation_data).float()
    validation_label = np.array(images_label[int(len(images_label) * 0.7):len(images_label)])
    validation_label = validation_label.reshape(validation_label.shape[0], -1)
    validation_label = torch.from_numpy(validation_label).long()



    return train_data, train_label, validation_data, validation_label


'''
    预测预处理数据
'''


def predict_data_preprocessing(images_path_list):
    image_list = []
    images_label = []
    image_list, images_label, images_name = get_imags_and_labels(image_list, images_label, images_path_list, 0, 0)

    # 70%图像被当做训练集
    train_data = np.array(image_list)
    train_data = torch.from_numpy(train_data).float()
    train_label = np.array(images_label)
    train_label = train_label.reshape(train_label.shape[0], -1)
    train_label = torch.from_numpy(train_label).long()

    return train_data, train_label, images_name


'''
    根据多个路径，提取路径中的图片，生成标签和图片 的训练验证集
'''


def get_imags_and_labels(image_list, images_label, images_path_list, ramdomData, need_synthesis_images):
    images_name = []
    for index, images_path in enumerate(images_path_list):
        # get_device_imgs_and_labels
        origin_images = get_images(images_path)
        # 将图片分成三份
        for img_path in origin_images:
            cut_img(img_path, image_list, images_label, images_name, need_synthesis_images)

    # 打乱顺序函数
    if (ramdomData):
        images_list_and_label = np.array([image_list, images_label])
        images_list_and_label = images_list_and_label.transpose()
        np.random.shuffle(images_list_and_label)
        image_list = list(images_list_and_label[:, 0])
        images_label = list(images_list_and_label[:, 1])
    return image_list, images_label, images_name


if __name__ == '__main__':
    data_preprocessing(constant.TRAIN_IMAGES_PATH_LIST)
