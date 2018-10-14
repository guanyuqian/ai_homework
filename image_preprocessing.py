import constant
from PIL import Image, ImageEnhance
import pandas as pd
import os, uuid


## 把样本图像分成3份，增加样本数量 ##
def cut_image(image_path, image_label, segment_image_list):
    arr = image_path[-7:-4]
    # 打开原始图片.jpg
    im = Image.open(image_path)
    # 左侧图片切割
    left_area = (constant.DX0, constant.DY0, constant.DX1, constant.DY1)
    # 中间图片切割
    middle_area = (constant.DX1, constant.DY0, constant.DX2, constant.DY1)
    # 右侧图片切割
    right_area = (constant.DX2, constant.DY0, constant.DX3, constant.DY1)
    # 把图片和标签对应放入segment_image_list中
    segment_image_list.append([im.crop(left_area), image_label[0]])
    segment_image_list.append([im.crop(middle_area), image_label[1]])
    segment_image_list.append([im.crop(right_area), image_label[2]])


# 丰富图片测试集合
def generate_img_name(file_path):
    return file_path + str(uuid.uuid1()) + ".jpg"


def synthesis_train_images(segment_image_list, synthesis_image_list):
    for img_and_label in segment_image_list:
        img = img_and_label[0]
        label = img_and_label[1]
        save_image(img, label, synthesis_image_list)
        # 反转
        fimg = flip(img)
        save_image(fimg, label, synthesis_image_list)
        for sharpness in floatrange(0.0, 60, 2):
            # 改变边缘清晰度（锐度）模糊：<1  锐化 >1
            simg = change_sharpness(fimg, sharpness)
            name = generate_img_name(constant.SYNTHESIS_IMAGE_PATH)
            synthesis_image_list[0].append(name)
            synthesis_image_list[1].append(label)
            simg.save(name)
            # 改变亮度
            for brightness in floatrange(0.2, 2, 2):
                bimg = change_brightness(simg, brightness)
                name = generate_img_name(constant.SYNTHESIS_IMAGE_PATH)
                synthesis_image_list.append({name, label})
                bimg.save(name)
        '''
        # 改变对比度,对比度增强
        for contrast in floatrange(0.1, 1, 2):
            cimg = change_contrast_ratio(img, contrast)
            name = generate_img_name(constant.SYNTHESIS_IMAGE_PATH)
            synthesis_image_list.append({name, label})
            cimg.save(name)
            # 改变饱和度,颜色
            for color in floatrange(0.1, 1, 2):
                clrimg = change_color(cimg, color)
                name = generate_img_name(constant.SYNTHESIS_IMAGE_PATH)
                synthesis_image_list.append({name, label})
                clrimg.save(name)
               
            
        
            
           
        for sharpness in floatrange(0.0, 1, 2):
            # 180旋转
            rimg = reverse180(img)
            name = generate_img_name(constant.SYNTHESIS_IMAGE_PATH)
            synthesis_image_list[0].append(name)
            synthesis_image_list[1].append(label)
            rimg.save(name)
             '''

    print("synthesis_train_images done\n")


def save_image(img, label, synthesis_image_list):
    name = generate_img_name(constant.SYNTHESIS_IMAGE_PATH)
    synthesis_image_list[0].append(name)
    synthesis_image_list[1].append(label)
    img.save(name)


# 改变饱和度,颜色
def change_color(img, color=0.1):
    return ImageEnhance.Color(img).enhance(color)


# 改变对比图,对比度增强
def change_contrast_ratio(img, contrast=0.1):
    return ImageEnhance.Contrast(img).enhance(contrast)


# 改变亮度
def change_brightness(img, brightness=0.2):
    # return img.point(lambda p: p - 0.1) #darker < 1.0 < lighter
    return ImageEnhance.Brightness(img).enhance(brightness)


# 改变边缘清晰度（锐度）模糊：<1  锐化 >1
def change_sharpness(img, sharpness=0.0):
    return ImageEnhance.Sharpness(img).enhance(sharpness)


# 180旋转+翻转
def reverse180_and_flip(img):
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img.transpose(Image.ROTATE_180)
    # img.rotate(180)


# 180旋转+翻转
def reverse180(img):
    return img.transpose(Image.ROTATE_180)
    # img.rotate(180)


# 翻转
def flip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)
    # img.rotate(180)


def floatrange(start, stop, steps):
    ''' Computes a range of floating value.

        Input:
            start (float)  : Start value.
            end   (float)  : End value
            steps (integer): Number of values

        Output:
            A list of floats

        Example:
            #>>> print floatrange(0.25, 1.3, 5)
            [0.25, 0.51249999999999996, 0.77500000000000002, 1.0375000000000001, 1.3]
    '''
    return [start + float(i) * (stop - start) / (float(steps) - 1) for i in range(steps)]


# 字典中的key值即为csv中列名
def write_csv(synthesis_image_list,
              csv_file_path):
    dataframe = pd.DataFrame(
        {constant.CSV_IMAGE_NAME: synthesis_image_list[0], constant.CSV_IMAGE_RESULT: synthesis_image_list[1]})
    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv(csv_file_path, index=False, sep=constant.CSV_SEP)
    print("write_csv done\n")


if __name__ == '__main__':
    segment_image_list = []  # 分成3段后的图片集合以及标签{image:label}
    synthesis_image_list = [[], []]  # 段图像变换后的图像集合及标签{image_file:label}
    list_file = os.listdir(constant.ORIGINAL_FAIL_IMAGE_PATH)
    for i in range(0, len(list_file)):
        image_path = os.path.join(constant.ORIGINAL_FAIL_IMAGE_PATH, list_file[i])
        if os.path.isfile(image_path) and image_path[-4:] == '.jpg':
            cut_image(image_path, constant.FAILURE_IMG_LABEL[i], segment_image_list)
    synthesis_train_images(segment_image_list, synthesis_image_list)

    list_file = os.listdir(constant.ORIGINAL_SUCC_IMAGE_PATH)  # 列出文件夹下所有的目录与文件
    image_list_label = [1, 1, 1]
    for i in range(0, len(list_file)):
        image_path = os.path.join(constant.ORIGINAL_SUCC_IMAGE_PATH, list_file[i])
        if os.path.isfile(image_path) and image_path[-4:] == '.jpg':
            cut_image(image_path, image_list_label, segment_image_list)
    synthesis_train_images(segment_image_list, synthesis_image_list)
    write_csv(synthesis_image_list, constant.TRAIN_SET)
