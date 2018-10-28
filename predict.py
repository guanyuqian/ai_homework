import os

import cv2
import torch
from PIL import Image, ImageDraw

import constant
import datetime

from gpu_train import CNN
from image_preprocessing import generate_img_name, data_preprocessing, predict_data_preprocessing, get_images

debug = 0


def get_imags_from_images_path_list(images_path_list):
    images_list = []
    for index, images_path in enumerate(images_path_list):
        # get_device_imgs_and_labels
        images_list.extend(get_images(images_path))
        # 将图片分成三份

    return images_list


def draw_failed_img(image_path, result,image_save_path):
    gred = cv2.imread(image_path, 0)
    img = Image.fromarray(cv2.cvtColor(gred, cv2.COLOR_GRAY2RGB))
    draw = ImageDraw.Draw(img)
    if result[0]==0:
        draw.rectangle(constant.LEFT_AREA, None, 'red', width = 10)
    if result[1]==0:
        draw.rectangle(constant.MIDDLE_AREA,None, 'red', width = 10)
    if result[2]==0:
        draw.rectangle(constant.RIGHT_AREA, None, 'red', width = 10)
    img.save(image_save_path+os.path.basename(image_path))

if __name__ == '__main__':

    print('开始时间', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    cnn = CNN()
    cnn = cnn.cuda()
    cnn.load_state_dict(torch.load(constant.LAST_MODEL_PATH))
    test_x, test_y, images_name = predict_data_preprocessing(constant.PREDICT_IMAGES_PATH_LIST)
    true = 0
    amount = 0
    result = [[0 for col in range(3)] for row in range(int(len(images_name) / 3))]
    images_path = get_imags_from_images_path_list(constant.PREDICT_IMAGES_PATH_LIST)
    for i, x in enumerate(test_x):
        x.unsqueeze_(0)
        x.unsqueeze_(0)
        x = x.cuda()
        test_output = cnn(x)[0]
        pre_y = torch.max(test_output, 1)[1].data.cpu().squeeze().numpy()
        result[int(i / 3)][i % 3] = pre_y
        if pre_y == 0:
            print("Failed:" + images_name[i])
        if i%3==2 and (result[int(i / 3)][0]==0 or result[int(i / 3)][1]==0 or  result[int(i / 3)][2]==0 ):
            draw_failed_img(images_path[int(i/3)], result[int(i / 3)], constant.RESULT_IMAGE_PATH)

        if debug:
            if pre_y == test_y[i].numpy():
                true += 1

            else:
                print("F", images_name[i])
                result[int(i / 3)][i % 3] = 0
            amount += 1
    if debug:
        acc = true / amount
        print(acc)

    print('训练结束时间', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
