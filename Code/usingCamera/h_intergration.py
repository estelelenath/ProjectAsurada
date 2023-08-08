from ultralytics import YOLO
import cv2
import cvzone
import math
import pandas
import numpy as np
import glob
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tracking_function import*

# Settings
# # Object Detection
# YOLO (https://github.com/alanzhichen/yolo8-ultralytics)
##########################################################################################
#   Model   #   Size(pixels)    #   mAP^val 50-95  #   Speed CPU ONNX(ms)  #    FLOPs    #
#   YOLO8n  #   640             #   37.3           #   80.4                #    8.7      #
#   YOLO8s  #   640             #   44.9           #   128.4               #    28.6     #
#   YOLO8m  #   640             #   50.2           #   234.7               #    78.9     #
#   YOLO8l  #   640             #   52.9           #   375.2               #    165.2    #
#   YOLO8x  #   640             #   53.9           #   479.1               #    257.8    #
##########################################################################################
model = YOLO('yolov8n.pt')      # load the official pretrained model (recommended for training)
# model = YOLO('yolov8s.pt')    #
# model = YOLO('yolov8m.pt')    #
# model = YOLO('yolov8l.pt')    #
# model = YOLO('yolov8x.pt')    #

# Functions
## Test Functions
## Mouse Cursor Coordinate
# : show the current mouse cursor coordinate in the window.
#   normally it should be deactive, but in the test or must be checked the coordinate of window
#   , then should be active to below cv2.nameWindow and cv2.setMouseCallback
def cursor_Coordinate(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :
        cursor_Coordinate = [x, y]
        print(cursor_Coordinate)
# Cursor Coordinate of front Camera
#cv2.namedWindow('camera_front_input')                           # in ' ' should be filled by name of display window
#cv2.setMouseCallback('camera_front_input', cursor_Coordinate)   # in ' ' should be filled by name of display window
# Cursor Coordinate of rear Camera
#cv2.namedWindow('camera_rear_input')                           # in ' ' should be filled by name of display window
#cv2.setMouseCallback('camera_rear_input', cursor_Coordinate)   # in ' ' should be filled by name of display window

## Object Detection
# Based on (https://github.com/freedomwebtech/yolov8counting-trackingvehicles)
# Based on (https://github.com/murtazahassan)
# Make a Class Array
open_coco = open("coco.txt", "r")
read_coco = open_coco.read()
class_list = read_coco.split("\n")


## Object Tracking
# Initialize count
count_f = 0                         # intialize for front camera
count_r = 0                         #
#tracking for front and rear video
tracker_f=Tracker()                 # Tracking function call for front camera
tracker_r=Tracker()

## Speed Estimation


## Distance Estimation


## Lane Detection


# Video Settings
fps = 24.0 # Viedo Frame
frame_size = (1080, 720) # Video Size
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Define the codec using VideoWriter_fourcc

## Video Input Part
### Webcam Mode
####################################################
# front camera capture
# camera_front_raw_input = cv2.VideoCapture(0)
# rear camera capture
# camera_rear_raw_input = cv2.VideoCapture(1)
####################################################

### Video Mode
####################################################
# front camera capture
video_front_raw_input = cv2.VideoCapture("D:\ProjectAsurada\ProjectAsurada\VideoSample\Front_driving.mp4")
frame_size_raw_front = (int(video_front_raw_input.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_front_raw_input.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#fps_video_front_raw_input = video_front_raw_input.get(cv2.CAP_PROP_FPS)    # Input video FPS Check
# rear camera capture
video_rear_raw_input = cv2.VideoCapture("D:\ProjectAsurada\ProjectAsurada\VideoSample\Rear-driving.mp4")
frame_size_rear = (int(video_rear_raw_input.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_rear_raw_input.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#fps_video_rear_raw_input = video_rear_raw_input.get(cv2.CAP_PROP_FPS)
####################################################
## Video Output Part
### Webcam Mode

### Video Mode
output_front = cv2.VideoWriter('video_front_output.mp4', fourcc, fps, frame_size) # for video export
output_rear = cv2.VideoWriter('video_rear_output.mp4', fourcc, fps, frame_size)
#output_front_test = cv2.VideoWriter('output_video_front_test.mp4', fourcc, fps, frame_size)
#output_rear_test = cv2.VideoWriter('output_video_rear_test.mp4', fourcc, fps, frame_size)

while True:
    ret_front, frame_front = video_front_raw_input.read()    # ret = return, if ret false, loop will be closing
    ret_rear, frame_rear = video_rear_raw_input.read()
    #############################if...else#############################
    #if ret_front and ret_rear:   # Check if frames were read correctly
    #    ...
    #    if cv2.waitKey(1) == ord('q'):
    #        break
    #else:
    #    break
    ###################################################################
    # Video Frame Resize
    video_front_resize_input = cv2.resize(frame_front, frame_size)
    video_rear_resize_input = cv2.resize(frame_rear, frame_size)
    #...

    # Object Detection and Tracking
    count_f += 1
    count_r += 1
    ####################################################################
    # performance optimization using data reduction
    #if count_f % 3 != 0:
    #    continue
    ####################################################################

    ##########################Front Camera_Start########################
    #result_f = model.predict(video_front_resize_input, stream=True)
    result_f = model.predict(video_front_resize_input)
    resbb_f = result_f[0].boxes.boxes
    px = pandas.DataFrame(resbb_f).astype("float")

    list = []  # in List, save the each frame information of detected object's x1,x2,y1,y2 value

    # index is in each frame, and indexing of detected object
    # in row value or above px value,
    # 0~3 values are about coordinate of detected rectangle box,
    # 4th value is about
    # 5th value is class_list's id, if it is 2, it means a car
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        #print("class_list", c) -> if car, then show car(confirmed!)
        if 'car' or 'motorcycle' or 'bus' or 'truck' in c:
            list.append([x1, y1, x2, y2])
    bbox_id = tracker_f.update(list)
    for bbox in bbox_id:
        x3, y3, x4, y4, sd, id, nr = bbox  # appended 4/4
        # x3, y3, x4, y4, id = bbox
        # sd : space difference
        # id : class_id
        # nr : vehicle identification nr(unsupported)
        # print("space_difference", sd)
        # relative_speed = space_difference / delta_t
        # in real World. (delta_s / delta_t) / scale_factor
        # real world speed, not necessary, main thema is how dangerous ->
        # -> box size chnaging / delta time(1/frame) -> rate of changing
        # if pos. -> closer, if neg. it further
        # rate of changing / distance = rate of approach(=1/approaching time)
        # if time is shorter than human react -> dangerous

        # if(sd<0):
        # print("SPEED", -(math.sqrt(abs(sd / (fps ** 2)))))
        # scalar_factor =
        # real_speed_on_km_h = -(math.sqrt(abs(sd / (fps ** 2)))) * scalar_factor
        # else:
        # print("SPEED", math.sqrt(abs(sd / (fps ** 2))))
        # scalar_factor =
        # real_speed_on_km_h = (math.sqrt(abs(sd / (fps ** 2)))) * scalar_factor

        # distance from other car to my car -> x_dist = cx - my car x coordinate, y_dist = cy - my car y coordinate
        # x_dist = cx - my car x coordinate
        # y_dist = cy - my car y coordinate

        # based on dist and speed->risiko measure!

        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        cv2.circle(video_front_resize_input, (cx, cy), 4, (0, 0, 255), -1)
        if cx > 540:
            cv2.rectangle(video_front_resize_input, (x3, y3), (x4, y4), (0, 0, 255), 1)
            cv2.putText(video_front_resize_input, f'{id}{nr}', (max(0, cx), max(35, cy)), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
        else:
            plt.rectangle
            cv2.putText(video_front_resize_input, f'{id}{nr}', (max(0, cx), max(35, cy)), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
        #cv2.putText(video_front_resize_input, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        # print("bbox", bbox)

    ##########################Front Camera_End##########################

    ##########################Rear Camera_Start#########################

    ##########################Rear Camera_End###########################

    # Output Video
    ## Original Video(just resize)
    cv2.imshow("camera_front_input", video_front_resize_input)
    cv2.imshow("camera_rear_input", video_rear_resize_input)
    #cv2.imshow("camera_front_resize_input_test", camera_front_resize_input_test)
    #cv2.imshow("camera_rear_resize_input_test", camera_rear_resize_input_test)

    ## Video Export
    output_front.write(video_front_resize_input)
    output_rear.write(video_rear_resize_input)
    #output_front.write(camera_front_resize_input)  # Test Video Front
    #output_rear.write(camera_rear_resize_input)    # Test Video Rear

    if cv2.waitKey(1) == ord('q'):
        break

video_front_resize_input.release()
video_rear_resize_input.release()
output_front.release()
output_rear.release()

cv2.destroyAllWindows()