from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils
import numpy as np
import cv2,os
import time
norm_size = 128
model_path = 'model/class.model'

if not os.path.exists(model_path):
    print('the pre_trained model is not exist')
else:
    model = load_model(model_path)

def predict(img_box):
    img = cv2.resize(img_box,(norm_size,norm_size))
    img = img.astype('float') / 255.0
    img = img_to_array(img)
    img = np.expand_dims(img,axis=0)

    result = model.predict(img)[0]

    proba = np.max(result)   ## 当前预测概率
    label = np.where(result==proba)[0]  ## 当前预测label

    return proba,label
if __name__ == '__main__':
    img_box = cv2.imread('data/images/0315.jpg')
    t1 = time.time()
    proba,label = predict(img_box)
    t2 = time.time()
    print(proba,label,t2-t1)
