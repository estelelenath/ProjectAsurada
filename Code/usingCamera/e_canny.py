from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np

# external functions
# Lane Detection (Canny Edge Detection)
def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blur before canny edge detection, there are a lot of noise, kernel size affect the noise reduce
    kernel = 5
    blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    canny = cv2.Canny(gray,50,150)
    return canny

def region_of_interest(canny):
    height = canny.shape[0]
    width = canny.shape[1]
    mask = np.zeros_like(canny)
    # size or form modify after.... ****
    #triangle = np.array([[
    #(200, height),
    #(800, 350),
    #(1200, height),]], np.int32)
    triangle = np.array([[
      (300,height/2), # Top-left corner
      (100, height), # Bottom-left corner            
      (900,height), # Bottom-right corner
      (700,height/2) # Top-right corner
    ,]], np.int32)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image

# 어떤 두 점이 직선의 관계를 가지는 부분을 찾는 것! 직선 -> lane이 된다는 생각, 따라서 허프 서클이면 원을 찾는 함수일것이고.. 그런 개념..
# https://bkshin.tistory.com/entry/OpenCV-23-%ED%97%88%ED%94%84-%EB%B3%80%ED%99%98Hough-Transformation
def houghLines(cropped_canny):
    return cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, 
        np.array([]), minLineLength=40, maxLineGap=5)

def addWeighted(frame, line_image):
    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)
 
def display_lines(img,lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),10)
    return line_image

def make_points(image, line):
    slope, intercept = line
    y1 = int(image.shape[0])
    y2 = int(y1*3.0/5)      
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return [[x1, y1, x2, y2]]

def average_slope_intercept(image, lines):
    left_fit    = []
    right_fit   = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1,x2), (y1,y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0: 
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    left_fit_average  = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line  = make_points(image, left_fit_average)
    right_line = make_points(image, right_fit_average)
    averaged_lines = [left_line, right_line]
    return averaged_lines

# front camera capture
#cap_front = cv2.VideoCapture(0)
# rear camera capture
#cap_rear = cv2.VideoCapture(1)

# for video Mode (for video mode recommended video width and height setting deactivate...)
cap_front = cv2.VideoCapture("D:\ProjectAsurada\ProjectAsurada\VideoSample\FrontCameraTestn.mp4")
cap_rear = cv2.VideoCapture("D:\ProjectAsurada\ProjectAsurada\VideoSample\MulticamTestRearn.mp4")

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

# Lane Detection
    canny_image = canny(img_rear)
    cropped_canny = region_of_interest(canny_image)

    lines = houghLines(cropped_canny)
    #averaged_lines = average_slope_intercept(img_rear, lines)
    #line_image = display_lines(img_rear, averaged_lines)
    #combo_image = addWeighted(img_rear, line_image)

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
    #cv2.imshow("canny_image",canny_image)
    #cv2.imshow("cropped_canny",cropped_canny)
    cv2.imshow("canny_image",lines)
    #cv2.imshow("canny_image",averaged_lines)
    #cv2.imshow("canny_image",line_image)
    #cv2.imshow("canny_image",combo_image)
    

    if cv2.waitKey(1) == ord('q'):
        break

cap_front.release()
cap_rear.release()
cv2.destroyAllWindows()