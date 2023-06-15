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

net=cv2.dnn_DetectionModel(weightsPath,configPath)  #this initializes the model and sets the path for weights and configuration file
net.setInputSize(320,320)  #this sets the pixel to be 320x320
net.setInputScale(1.0/127.5) #the input image is divided by 127.5 to scale it out
net.setInputMean((127.5,127.5,127.5)) #to choose a center
net.setInputSwapRB(True) #to swap RED and BLUE color as we prefer RGB but input images are in BGR



while True:
    success,img=cap.read() #the frame is read continously and stored in img, success is indicating if it was possible to read the frame or not
    classIds, confs, bbox = net.detect(img, confThreshold=0.45) # this sets the threshold for truth in the obejct detection
    #this function detects the classID in the image and returns the value to the variables


    if len(classIds)!=0: #if the classID exixts
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):  # this makes the box around the object and displays text over the image
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId - 1], (box[0] + 10, box[0] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0),
                        2)

    cv2.imshow('image', img) # this displays the image
    cv2.waitKey(1) # this waits for the image to be closed by the user
