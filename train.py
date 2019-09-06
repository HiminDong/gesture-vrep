import matplotlib

from keras.preprocessing.image import ImageDataGenerator,img_to_array
from keras.optimizers import Adam
from keras.utils import to_categorical
from imutils import paths

import matplotlib.pyplot as plt
import numpy as np
import os,cv2,sys
import argparse
import random,pdb
from model import class_model


pre_labels = ['n','o','0','1','2','3','4','5','s']
norm_size = 128   ## 模型输入图片尺寸
CLASS_NUM = 9    ## 总类别数
init_lr = 0.001  ## 初始学习率
EPOCHS = 100  ## 训练次数
BS = 32  ## batch


def load_data(path):
    imgs_path = []
    labels = []
    data = []
    with open(path,'r') as fil:
        for line in fil.readlines():
            imgs_path.append(line.strip('\n').split(' ')[0])
            labels.append(pre_labels.index(line.strip('\n').split(' ')[1]))

    for img_pth in imgs_path:
        im = cv2.imread(img_pth)
        im = cv2.resize(im,(norm_size,norm_size))   # norm_size
        im = img_to_array(im)
        data.append(im)

    assert(len(data) == len(labels))

    data = np.array(data,dtype = 'float')/255.0
    labels = np.array(labels)

    labels = to_categorical(labels,num_classes=CLASS_NUM) # CLASS_NUM
    return data,labels

def train(aug,train_data,train_label,test_data,test_label,show_acc=True):
    model = class_model.creat_model(width=norm_size,height=norm_size,depth=3,classes=CLASS_NUM)
    optor = Adam(lr = init_lr, decay=init_lr/EPOCHS) ##优化器
    model.compile(loss='categorical_crossentropy',optimizer = optor,
            metrics=['accuracy'])  ## 指定模型损失函数，优化器

    ###### train model #######3

    print('traing model')
    H = model.fit_generator(aug.flow(train_data,train_label,batch_size=BS),validation_data=(test_data,test_label),steps_per_epoch=len(train_data)//BS,epochs=EPOCHS,verbose=1)  ###训练模型

    model.save('model/class.model')
    if show_acc:   ### 显示训练过程中loss,准确率的变化
        plt.style.use('ggplot')
        plt.figure()
        N = EPOCHS   # total epochs
        plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
        plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
        plt.title("Training Loss and Accuracy on gesture classifier")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig('loss_acc.jpg')



if __name__ == '__main__':
    train_txt = 'data/train.txt'
    test_txt = 'data/test.txt'
    train_data,train_label = load_data(train_txt)
    test_data,test_label = load_data(test_txt)
    #### 数据增强 ####
    aug = ImageDataGenerator(rotation_range=30,width_shift_range=0.1,
            height_shift_range=0.1,shear_range=0.2,zoom_range=0.2,
            horizontal_flip=True,fill_mode='nearest')

    train(aug,train_data,train_label,test_data,test_label)
