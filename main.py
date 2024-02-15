# Using pre-trained deep learning architecture 
# Using openCV to load images from above architecture 
import cv2 as cv 
import matplotlib.pyplot as pl

configFile='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozenModel='frozen_inference_graph.pb'

model=cv.dnn_DetectionModel(frozenModel,configFile)
classLabels=[]
with open('labels.txt','rt') as f:
    classLabels=f.read().rstrip('\n').split('\n')
# print(classLabels)
# Setting up the configuration 
model.setInputSize(320,320)
model.setInputScale(1.0/127.5)#255/2 
model.setInputMean(127.5)#mobilenet takes input as [-1,1]
model.setInputSwapRB(True)#performs bgr->rgb automatically 


# #Reading an image (in config file , the img size config is 320*320 )
# img=cv.imread('img.jpg') #bgr
# # imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
# pl.imshow(img)
# pl.show()

# print(classIndex)


#Video-------
cap =cv.VideoCapture(0)
while True:
    ret , frame = cap.read()
    classIndex,confidence,bbox=model.detect(frame,confThreshold=0.43)
    for classInd,conf,boxes in zip(classIndex,confidence,bbox):
  
      cv.rectangle(frame,boxes,(255,0,0),2)
      cv.putText(frame,classLabels[(classInd-1)],(boxes[0]+10,boxes[1]+40),cv.FONT_HERSHEY_PLAIN,3,(0,255.0),3)
      cv.imshow('Detection',frame)

      if cv.waitKey(1) & 0xFF == ord("q"):
        break

# release the camera and close the window
cap.release()
cv2.destroyAllWindows()

    



