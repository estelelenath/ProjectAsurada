from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np

# external functions

# front camera capture
#cap_front = cv2.VideoCapture(0)
# rear camera capture
#cap_rear = cv2.VideoCapture(1)

# for video Mode (for video mode recommended video width and height setting deactivate...)
cap_front = cv2.VideoCapture("D:\ProjectAsurada\ProjectAsurada\VideoSample\FrontCameraTestnnn.mp4")
cap_rear = cv2.VideoCapture("D:\ProjectAsurada\ProjectAsurada\VideoSample\MulticamTestRearnn.mp4")

# video width setting for front camera
#cap_front.set(3, 1280)
# video height setting for front camera
#cap_front.set(4, 720)

# video width setting for rear camera
#cap_rear.set(3, 1280)
# video height setting for rear camera
#cap_rear.set(4, 720)

# option for object detection data set
model_n = YOLO('/YOLO_WEIGHTS/yolov8n.pt')
#model_m = YOLO('/YOLO_WEIGHTS/yolov8m.pt')

classNames = []
 #class 배열 만들기
with open("coco.names", "r") as f:
    classNames = [line.strip() for line in f.readlines()]
# 읽어온 coco 파일을 whitespace(공백라인)를 제거하여 classes 배열 안에 넣는다.
# strip() : whitespace(띄워쓰기, 탭, 엔터)를 없애는 것, 중간에 끼어있는 것은 없어지지 않는다.

while True:
    success, img_front = cap_front.read()
    success, img_rear = cap_rear.read()
    
# Vehicle Detection with ML_Model with input video    
    results_front = model_n(img_front, stream=True)
    results_rear = model_n(img_rear, stream=True)



# Vehicle Detection for front camera
    for rf in results_front:
        boxesf = rf.boxes
        for boxf in boxesf:

            # Bounding Box
            x1,y1,x2,y2 = boxf.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img_front, (x1,y1), (x2,y2), (0, 0, 255, 127), -1)
            w, h = x2 - x1, y2 - y1
            # rt = -1 -> fullfilled rectangle, 0~ -> normal thickness
            cvzone.cornerRect(img_front, (x1, y1, w, h), rt=1, colorR=(0, 0, 255))         
            # Confidence
            confidence = math.ceil((boxf.conf[0] * 100)) / 100
            
            # Class Name
            cls = int(boxf.cls[0])

            cvzone.putTextRect(img_front, f'{classNames[cls]} {confidence}', (max(0, x1), max(35, y1))) # , scale = 3, thickness = 3

# Vehicle Detection for rear camera
    for rr in results_rear:
        boxesr = rr.boxes
        for boxr in boxesr:
            x3,y3,x4,y4 = boxr.xyxy[0]
            x3,y3,x4,y4 = int(x3), int(y3), int(x4), int(y4)
            # last parameter -1 -> fullfilled rectangle, 0~ -> normal thickness
            cv2.rectangle(img_rear, (x3,y3), (x4,y4), (0, 0, 255), 1)
            # alpha = 0.4
            # opacity_rear = cv2.addWeighted(img_rear, alpha, img_rear, 1-alpha,0)

    cv2.imshow("img_front",img_front)
    cv2.imshow("img_rear",img_rear)

    

    if cv2.waitKey(1) == ord('q'):
        break

cap_front.release()
cap_rear.release()
cv2.destroyAllWindows()