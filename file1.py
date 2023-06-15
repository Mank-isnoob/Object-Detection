import cv2
import numpy as np
classNames=[]
classFile='coco.names'

with open(classFile,'r') as f:
    classNames=f.read().split('\n')
print(classNames)

configPath='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath='frozen_inference_graph.pb'

#img=cv2.imread('cars.jpg')
cap=cv2.VideoCapture(0)
#img=cv2.resize(img,(0,0),fx=0.5,fy=0.5)

net=cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)



while True:
    success,img=cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=0.45)


    if len(classIds)!=0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId - 1], (box[0] + 10, box[0] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0),
                        2)

    cv2.imshow('image', img)
    cv2.waitKey(1)
