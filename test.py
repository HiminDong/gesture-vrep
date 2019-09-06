from label import process
from inference import predict
import cv2,pdb
from imutils.video import FPS
import sys

labels = ('None','start','0','1','2','3','4','5','stop')
vs = cv2.VideoCapture(0)
cv2.namedWindow('frame',0)
cap_fps = vs.get(cv2.CAP_PROP_FPS)
print('video fps: ',cap_fps)
result = 'None'
fps = FPS().start()
while True:
    key = cv2.waitKey(1) & 0xFF
    rec,frame = vs.read()
    cv2.putText(frame,result,(50,145),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
    cv2.rectangle(frame,(50,150),(350,450),(0,0,255),2)
    cv2.imshow('frame',frame)

    img_box = frame[150:450,50:350]
    img_box = cv2.cvtColor(process(img_box),cv2.COLOR_GRAY2BGR)
#    img_box = process(img_box)
    proba,label = predict(img_box)
    if proba > 0.8:
        result = labels[label[0]]
    else:
        result = 'None'
    if key == ord('q'):
        break
    fps.update()
    fps.stop()
vs.release()

