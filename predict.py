import glob
import os
import PIL
from tkinter import filedialog
import subprocess
import cv2
import torch
from PIL import ImageDraw
import constant
import datetime
from train import CNN
from image_preprocessing import predict_data_preprocessing
import warnings



from tkinter import *
if __name__ == '__main__':
    def selectPath():
        path.set(filedialog.askdirectory(initialdir=path.get()))


    def delete_file_folder(path):

        #   read all the files under the folder
        fileNames = glob.glob(path + r'\*')

        for fileName in fileNames:
            try:
                #           delete file
                os.remove(fileName)
            except:
                try:
                    #               delete empty folders
                    os.rmdir(fileName)
                except:
                    #               Not empty, delete files under folders
                    delete_file_folder(fileName)



    def draw_image(left,right,mid,images_name):
        gred = cv2.imread(images_name, 0)
        img = PIL.Image.fromarray(cv2.cvtColor(gred, cv2.COLOR_GRAY2RGB))
        draw = ImageDraw.Draw(img)
        if(left):
            print(images_name+" left");
            draw.rectangle(constant.LEFT_AREA, None, 'red', width=10);
        if(mid):
            print(images_name+" mid");
            draw.rectangle(constant.MIDDLE_AREA, None, 'red', width=10);
        if(right):
            print(images_name+" right");
            draw.rectangle(constant.RIGHT_AREA, None, 'red', width=10);
        img.save(path2.get() + os.path.basename(images_name))

    def predict():
        log = '程序开始时间'+str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))+'\n'
        print('程序开始时间', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print('载入模型')
        cnn = CNN()
        if constant.USE_GPU:
            cnn = cnn.cuda()
        cnn.load_state_dict(torch.load(constant.LAST_MODEL_PATH))
        test_x, test_y, images_name = predict_data_preprocessing([path.get()])
        left=0
        mid=0
        right=0
        delete_file_folder(path2.get())
        log = '开始检测时间'+str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))+'\n'
        print('开始检测时间', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        for i, x in enumerate(test_x):
            x.unsqueeze_(0)
            x.unsqueeze_(0)
            if constant.USE_GPU:
                x = x.cuda()
            test_output = cnn(x)[0]
            pre_y = torch.max(test_output, 1)[1].data.cpu().squeeze().numpy()
            if pre_y == 0:
                if(i%3==0):
                    left = 1
                elif i%3==1:
                    mid = 1
                else:
                    right = 1

                # print("Failed:" + images_name[i])
                log += ("Failed:" + images_name[i]) + '\n'
            if i%3==2:
                if left|right|mid:
                    draw_image(left,right,mid,path.get()+images_name[i][0:-4])
                left = 0
                mid = 0
                right = 0
        print('结束时间', str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        log += '结束时间'+ datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n'
        f = open(path2.get()+'/result.txt', 'w')
        f.write(log)
        f.close()
        subprocess.Popen('explorer "%s"' % path2.get().replace('/', '\\'))
    root = Tk()
    root.title("产品缺陷检测")
    #窗口大小
    #width ,height= 540, 800
    #窗口居中显示
    #root.geometry('%dx%d+%d+%d' % (width,height,(root.winfo_screenwidth() - width ) / 2, (root.winfo_screenheight() - height) / 2))
    warnings.filterwarnings('ignore')
    path = StringVar()
    path.set(constant.PREDICT_IMAGES_PATH_LIST[0])
    path2 = StringVar()
    path2.set(constant.RESULT_IMAGE_PATH)
    Label(root, text="目标路径:").grid(row=0, column=0)
    entry=Entry(root, textvariable=path,width=50).grid(row=0, column=1)
    Button(root, text="路径选择", command=selectPath).grid(row=0, column=2)

    Label(root, text="输出路径:").grid(row=1, column=0)
    entry2 = Entry(root, textvariable=path2, width=50).grid(row=1, column=1)
    Button(root, text="路径选择", command=selectPath).grid(row=1, column=2)
    button = Button(root, text="开始检测", command=predict).grid(row=2)
    log_label = StringVar()
    log_label.set ("日志信息：\n")
    Label(root, textvariable=log_label).grid(row=1, column=0)
    root.mainloop()
