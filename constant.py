import os


class Const:
    class ConstError(TypeError):
        pass

    class ConstCaseError(ConstError):
        pass

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstError("Can't change const value!")
        if not name.isupper():
            raise self.ConstCaseError('const "%s" is not all letters are capitalized' % name)
        self.__dict__[name] = value


import sys

sys.modules[__name__] = Const()

import constant

# 裁剪图片偏移量
constant.WIDTH_PX = 570
constant.HEIGHT_PX = 1200
constant.DX0 = 380
constant.DX1 = constant.DX0 + constant.WIDTH_PX
constant.DX2 = constant.DX1 + constant.WIDTH_PX
constant.DX3 = constant.DX2 + constant.WIDTH_PX
constant.DY0 = 548
constant.DY1 = constant.DY0 + constant.HEIGHT_PX

# 路径
constant.ORIGINAL_SUCC_IMAGE_PATH = './original_images/success_images/'
constant.ORIGINAL_FAIL_IMAGE_PATH = './original_images/failure_images/'
constant.SEGMENT_IMAGE_PATH = './segment_images/'
constant.SYNTHESIS_IMAGE_PATH = './synthesis_images/'
constant.CHECK_IMAGE_PATH = './original_images/check_images/'
constant.TRAIN_SET = './cvs/train_set.csv'

# 测试集
constant.MODEL_PATH = 'binary_model.h5'
constant.CNN_PATH = 'cnn_model.h5'
constant.PREDICT_LABEL = [
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    1, 1, 0,
    1, 0, 0,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1
]


constant.PREDICT_SUCC_LABEL = [
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1
]
# CSV相关
constant.CSV_IMAGE_RESULT = 'img_result'
constant.CSV_IMAGE_NAME = 'img_name'
constant.CSV_SEP = ','

constant.FAILURE_IMG_LABEL = [
    [0, 0, 0], [1, 1, 0],
    [1, 1, 0], [0, 0, 0],
    [0, 0, 0], [0, 0, 0],
    [1, 0, 0], [1, 0, 0],
    [1, 1, 0], [0, 0, 0],
    [1, 0, 0], [0, 0, 0],
    [1, 0, 0], [0, 0, 0],
    [0, 0, 0], [1, 0, 0],
    [0, 0, 0], [0, 0, 0],
    [1, 0, 0], [1, 0, 0],
    [0, 0, 0], [0, 0, 0],
    [1, 0, 0], [1, 0, 0],
    [1, 0, 0], [1, 0, 0],
    [0, 0, 0], [0, 0, 0],
    [1, 0, 0], [1, 1, 0],
    [1, 0, 0], [1, 1, 0],
    [1, 0, 0], [0, 0, 0],
    [1, 0, 0], [0, 0, 0],
    [1, 1, 0], [0, 0, 0],
    [1, 0, 0], [1, 1, 0],
    [0, 0, 0], [1, 0, 0],
    [0, 0, 0], [0, 0, 0],
    [1, 0, 0], [0, 0, 0],
    [0, 0, 0], [0, 0, 0],
    [0, 0, 0], [1, 0, 0]
]
