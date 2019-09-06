from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

from keras import backend as K

class class_model:
    def creat_model(width,height,depth,classes):
        model = Sequential()
        in_shape = (height,width,depth)
        #### 第一层网络：卷积层-激活函数-polling层
        model.add(Conv2D(20,(3,3),padding='same',input_shape=in_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        ### 第二层： 结构和第一层相同
        model.add(Conv2D(30,(3,3),padding='same',input_shape=in_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        ### 第三层： 结构相同
        model.add(Conv2D(50,(5,5),padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        ### 第四层： 全连接层
        model.add(Flatten())
        model.add(Dense(50))
        model.add(Activation('relu'))

        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model

