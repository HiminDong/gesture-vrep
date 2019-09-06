import cv2
import os
import random,pdb
import matplotlib.pyplot as plt

labels = ('n','o','0','1','2','3','4','5','s')

def process(image):
#    image = cv2.imread('data/images/0060.jpg')
    #pdb.set_trace()
    ##YCrCb空间，搜索一下介绍
    img = cv2.cvtColor(image,cv2.COLOR_BGR2YCR_CB)
    Y,Cb,Cr = cv2.split(img)
    ## 二值化
    ret,im = cv2.threshold(Cb,133,255,cv2.THRESH_BINARY)
    cv2.imshow('ff',im)
    ### 直方图选择合适二值化阈值
    #plt.figure()
    #n,bins,patchs = plt.hist(Cb.ravel(),256)
    #plt.show()
    ### 轮廓提取
    #contours,hierarchy = cv2.findContours(im,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #contours = [contour for contour in contours if cv2.contourArea(contour)>1000]
    #cv2.drawContours(image,contours,-1,(255,0,0),3)
    #cv2.imshow('im',)
   # cv2.waitKey(0)
    return im
    
    
    

def split_data():   # 拆分数据集为训练集和验证集
    data_f = open('data/data_label.txt','r')
    train_f = open('data/train.txt','w')
    test_f = open('data/test.txt','w')
    
    data_list = data_f.readlines()
    random.shuffle(data_list)
    
    for train in data_list[:int(0.75*len(data_list))]:
        train_f.write(train)
    for test in data_list[int(0.75*len(data_list)):]:
        test_f.write(test)

def create_label():
    vs = cv2.VideoCapture(0)
    cv2.namedWindow('frame',0)
    txt = open('data/data_label.txt','w')
    count = 1
    for label in labels:
        key = cv2.waitKey(1) & 0xFF
        print('please make this gesture: ',label)
        while True:
            key = cv2.waitKey(1) & 0xFF
            rec,frame = vs.read()
            cv2.putText(frame,'data window',(50,145),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),5)

            # 选取摄像头，框选部分
            cv2.rectangle(frame,(50,150),(350,450),(0,0,255),2)
            cv2.imshow('frame',frame)

            if key == ord('c'):  #保存当前框图片为数据集
                img_box = frame[150:450,50:350]
                img_box = process(img_box)
                cv2.imwrite('data/images/{}.jpg'.format(str(count).zfill(4)),img_box)
                txt.write('data/images/{}.jpg'.format(str(count).zfill(4))+' '+label+'\n')
                print(count)
                count += 1
            elif key == ord('n'):  #下一个label 准备数据集
                break
            elif key == ord('q'):  #退出并分割数据集
                txt.close()
                split_data()
                os._exit(0)
    vs.release()

if __name__ == '__main__':
    process()
    #create_label()
