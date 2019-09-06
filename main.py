import vrep
from label import process
from inference import predict
import cv2
from imutils.video import FPS
import sys
import numpy as np
import math
import time
import pdb

RAD2EDG = 180 / math.pi
tstep = 0.005

# 配置关节信息
jointNum = 6
baseName = 'puma560'
jointName = 'puma560_joint'


def link_vrep(link = False):# 连接vrep

    print('link vrep....')
    # 关闭潜在的链接
    vrep.simxFinish(-1)

    clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)  # Connect to V-REP
    if clientID != -1:
        link = True
        print('Success Connected to remote API server')
    vrep.simxSetFloatingParameter(clientID, vrep.sim_floatparam_simulation_time_step,
                                  tstep, vrep.simx_opmode_oneshot)
    vrep.simxSynchronous(clientID, True)  ###打开同步模式
    vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot)
    return clientID,link

def control_vrep(clientID,count):
    # 读取Base 和 joint 的handle
    jointHandle = np.zeros((jointNum,), dtype=np.int)
    for i in range(jointNum):
        _, returnHandle = vrep.simxGetObjectHandle(clientID, jointName + str(i + 1), vrep.simx_opmode_blocking)
        jointHandle[i] = returnHandle

    _, baseHandle = vrep.simxGetObjectHandle(clientID, baseName, vrep.simx_opmode_blocking)
    print('Handles available')

    # 首次读取关节的初始值，以streaming的形式
    jointConfigure = np.zeros((jointNum,))
    for i in range(jointNum):
        _, jpos = vrep.simxGetJointPosition(clientID, jointHandle[i], vrep.simx_opmode_streaming)
        jointConfigure[i] = jpos

    vrep.simxSynchronousTrigger(clientID)  # 让仿真走一步

    ### 控制指定关节 ####
    vrep.simxPauseCommunication(clientID, True)

    vrep.simxSetJointTargetPosition(clientID, jointHandle[count],
                                    jointConfigure[count] +1/ RAD2EDG, vrep.simx_opmode_oneshot)

    vrep.simxPauseCommunication(clientID, False)

    vrep.simxSynchronousTrigger(clientID)
    vrep.simxGetPingTime(clientID)  # 使得该仿真步


if __name__ == '__main__':
    clientId,link =link_vrep()
    if link:
        labels = ('None','start','0','1','2','3','4','5','stop')
        vs = cv2.VideoCapture(0)
        cv2.namedWindow('frame',0)
        cap_fps = vs.get(cv2.CAP_PROP_FPS)
        print('video fps: ',cap_fps)
        result = 'None'
        fps = FPS().start()
        START = False
        while True:
            key = cv2.waitKey(1) & 0xFF
            rec,frame = vs.read()
            cv2.putText(frame,result,(50,145),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            cv2.rectangle(frame,(50,150),(350,450),(0,0,255),2)
            cv2.imshow('frame',frame)

            img_box = frame[150:450,50:350]
            img_box = cv2.cvtColor(process(img_box),cv2.COLOR_GRAY2BGR)
            proba,label = predict(img_box)
            if proba > 0.8:
                result = labels[label[0]]
                if result == 'start':
                    START = True
                if START and 8>label[0]>1:
                    control_vrep(clientId,label[0]-2)
                if result == 'stop':
                    START = False
            else:
                result = 'None'
            if key == ord('q'):
                break
            fps.update()
            fps.stop()
        vs.release()




print ('Program ended')

