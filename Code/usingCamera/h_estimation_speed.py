from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
# https://moon-coco.tistory.com/entry/OpenCV%EC%B0%A8%EC%84%A0-%EC%9D%B8%EC%8B%9D
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# pip install youtube_dl && pip install pafy


# external functions

# front camera capture
#cap_front = cv2.VideoCapture(0)
# rear camera capture
#cap_rear = cv2.VideoCapture(1)

# for video Mode (for video mode recommended video width and height setting deactivate...)
cap_front = cv2.VideoCapture("D:\ProjectAsurada\ProjectAsurada\VideoSample\FrontCamera_Odeon.mp4")
#cap_front = cv2.VideoCapture("D:\ProjectAsurada\ProjectAsurada\VideoSample\Ffront_driving_plate_hwy_880_H264HD1080.mp4")
cap_rear = cv2.VideoCapture("D:\ProjectAsurada\ProjectAsurada\VideoSample\MulticamTestRearnn.mp4")

frame_size_front = (int(cap_front.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_front.get(cv2.CAP_PROP_FRAME_HEIGHT)))
frame_size_rear = (int(cap_rear.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_rear.get(cv2.CAP_PROP_FRAME_HEIGHT)))

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

#--------------------------------------Warpping (Bird Eye View)-------------------------------
def wrapping(image):
    (h, w) = (image.shape[0], image.shape[1])
    
    #source = np.float32([[w // 2 - 30, h * 0.53], [w // 2 + 60, h * 0.53], [w * 0.3, h], [w, h]])
    #destination = np.float32([[0, 0], [w-350, 0], [400, h], [w-150, h]])

    #one lane version
    source_point_left_upper = [w * 4.8 // 10, h * 0.62]
    source_point_right_upper = [w * 6.1 // 10, h * 0.62]
    source_point_left_bottom = [w * 0.20, h]
    source_point_right_bottom = [w* 0.85, h]

    destination_point_left_upper = [200, 0]
    destination_point_right_upper = [w-380, 0]
    destination_point_left_bottom = [400, h]
    destination_point_right_bottom = [w-150, h]

    source = np.float32([source_point_left_upper, source_point_right_upper, source_point_left_bottom, source_point_right_bottom])
    destination = np.float32([destination_point_left_upper, destination_point_right_upper, destination_point_left_bottom, destination_point_right_bottom])

    #getPerspectiveTransformation? the properties that it hold the property of linear, but not the property of parallelity
    #for example, train lanes are parallel but through the perspective transformation, it looks like they are meeing at the end of point
    #we need 4 point of input and moving point of output
    # for the transformation matrix we need, through the cv2.getPerspectiveTransform() function and adjust our transformation matrix to cv2.warpPerspective() function, we could have a final image
    # 
    transform_matrix = cv2.getPerspectiveTransform(source, destination)
    minv = cv2.getPerspectiveTransform(destination, source)
    _image = cv2.warpPerspective(image, transform_matrix, (w, h))

    return _image, minv
#---------------------------------------------------------------------------------------------

#-------------------------------------Color Filter (using HLS)--------------------------------------------------------
# HLS(Hue, Luminanse, Saturation) : 
#lower = ([minimum_blue, m_green, m_red])
#upper = ([Maximum_blue, M_green, M_red])
def color_filter(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    #White Filter
    #white_lower = np.array([20, 150, 20])
    #white_upper = np.array([255, 255, 255])
    # White-ish areas in image
    # H value can be arbitrary, thus within [0 ... 360] (OpenCV: [0 ... 180])
    # L value must be relatively high (we want high brightness), e.g. within [0.7 ... 1.0] (OpenCV: [0 ... 255])
    # S value must be relatively low (we want low saturation), e.g. within [0.0 ... 0.3] (OpenCV: [0 ... 255])
    white_lower = np.array([np.round(  0 / 2), np.round(0.55 * 255), np.round(0.00 * 255)])
    white_upper = np.array([np.round(360 / 2), np.round(1.00 * 255), np.round(0.20 * 255)])
    white_mask = cv2.inRange(hls, white_lower, white_upper)

    #Yellow Filter
    #yellow_lower = np.array([0, 85, 81])
    #yellow_upper = np.array([190, 255, 255])
    # Yellow-ish areas in image
    # H value must be appropriate (see HSL color space), e.g. within [40 ... 60]
    # L value can be arbitrary (we want everything between bright and dark yellow), e.g. within [0.0 ... 1.0]
    # S value must be above some threshold (we want at least some saturation), e.g. within [0.35 ... 1.0]
    yellow_lower = np.array([np.round( 40 / 2), np.round(0.00 * 255), np.round(0.35 * 255)])
    yellow_upper = np.array([np.round( 60 / 2), np.round(1.00 * 255), np.round(1.00 * 255)])
    yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)


    # Do filtering the each yellow lane and white lane,
    # Bitwise_or makes (yellow line and white line) combining -> mask
    # bitwise_and maeks (original image and mask) -> then left just masked part -> masked
    #yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)
    #white_mask = cv2.inRange(hls, white_lower, white_upper)
    mask = cv2.bitwise_or(yellow_mask, white_mask)
    masked = cv2.bitwise_and(image, image, mask = mask)

    return masked
#---------------------------------------------------------------------------------------------

#-------------------------------------ROI--------------------------------------------------------
def roi(image):
    x = int(image.shape[1])
    y = int(image.shape[0])
    # height, width, number of channels in image
    #height = img.shape[0]
    #width = img.shape[1]
    #channels = img.shape[2]
    #Height represents the number of pixel rows in the image or the number of pixels in each column of the image array.
    #Width represents the number of pixel columns in the image or the number of pixels in each row of the image array.
    #Number of Channels represents the number of components used to represent each pixel.
    #In the above example, Number of Channels = 4 represent Alpha, Red, Green and Blue channels.
    # *** here traffic sign on the street is deleted and ignored, if you don't wanna that, modify the ROI part.
    # 한 붓 그리기
    _shape = np.array([
        [int(0.19*x), int(y)],
          [int(0.19*x), int(0.1*y)],
            [int(0.45*x), int(0.1*y)],
              [int(0.5*x), int(y)],
                [int(0.6*x), int(y)],
                  [int(0.6*x), int(0.1*y)],
                  [int(0.82*x), int(0.1*y)],
                    [int(0.82*x), int(y)],
                      [int(0.3*x), int(y)]
                      ])

    mask = np.zeros_like(image)

    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, np.int32([_shape]), ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image
#---------------------------------------------------------------------------------------------

#-------------------------------------Histogram--------------------------------------------------------
# it is not histogram of opencv
# bitwise image has one channel and value between 0 ~ 255.
# if it is lane, they have a value near by 255, and if it isn't, then 0.
# it means for one column, when we add all row values, if there are lane, they has relative big value, if not, small value
# 1050 -> right lane, 350 -> left lane
def plothistogram(image):
    histogram = np.sum(image[image.shape[0]//2:, :], axis=0)
    midpoint = np.int64(histogram.shape[0]/2)
    leftbase = np.argmax(histogram[:midpoint])
    rightbase = np.argmax(histogram[midpoint:]) + midpoint
    
    return leftbase, rightbase
#---------------------------------------------------------------------------------------------

#-------------------------------------Window ROI--------------------------------------------------------
# why not cv2.HoughLines() and cv2.HoughLinesP()? -> these functions are heavy and detection is not exact for curve.
# left_current = a biggest index of image's left side
# good_left = save the part just in window
# next left_current of window is mean value of index, that good_left of nonzero_x have, if godd_left length is shorter than 50.
# np.concatenate : Array makes 1.Dimenstion array
# np.trunc : throw away a decimal part
def slide_window_search(binary_warped, left_current, right_current):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    nwindows = 4
    window_height = np.int64(binary_warped.shape[0] / nwindows)
    nonzero = binary_warped.nonzero()  # 선이 있는 부분의 인덱스만 저장 
    nonzero_y = np.array(nonzero[0])  # 선이 있는 부분 y의 인덱스 값
    nonzero_x = np.array(nonzero[1])  # 선이 있는 부분 x의 인덱스 값 
    margin = 100
    minpix = 50
    left_lane = []
    right_lane = []
    color = [0, 255, 0]
    thickness = 2

    for w in range(nwindows):
        win_y_low = binary_warped.shape[0] - (w + 1) * window_height  # window 윗부분
        win_y_high = binary_warped.shape[0] - w * window_height  # window 아랫 부분
        win_xleft_low = left_current - margin  # 왼쪽 window 왼쪽 위
        win_xleft_high = left_current + margin  # 왼쪽 window 오른쪽 아래
        win_xright_low = right_current - margin  # 오른쪽 window 왼쪽 위 
        win_xright_high = right_current + margin  # 오른쪽 window 오른쪽 아래

        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), color, thickness)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), color, thickness)
        # window 안에 있는 부분만을 저장
        good_left = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
        good_right = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]
        left_lane.append(good_left)
        right_lane.append(good_right)
        #cv2.imshow("oo", out_img)

        if len(good_left) > minpix:
            left_current = np.int64(np.mean(nonzero_x[good_left]))
        if len(good_right) > minpix:
            right_current = np.int64(np.mean(nonzero_x[good_right]))

    left_lane = np.concatenate(left_lane)  # np.concatenate() -> array를 1차원으로 합침 (array를 1차원 배열로 만들어줌)
    right_lane = np.concatenate(right_lane)

    leftx = nonzero_x[left_lane]
    lefty = nonzero_y[left_lane]
    rightx = nonzero_x[right_lane]
    righty = nonzero_y[right_lane]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    ltx = np.trunc(left_fitx)  # np.trunc() -> 소수점 부분을 버림
    rtx = np.trunc(right_fitx)

    out_img[nonzero_y[left_lane], nonzero_x[left_lane]] = [255, 0, 0]
    out_img[nonzero_y[right_lane], nonzero_x[right_lane]] = [0, 0, 255]

    #plt.imshow(out_img)
    #plt.plot(left_fitx, ploty, color = 'yellow')
    #plt.plot(right_fitx, ploty, color = 'yellow')
    #plt.xlim(0, 1280)
    #plt.ylim(720, 0)
    #plt.show()

    ret = {'left_fitx' : ltx, 'right_fitx': rtx, 'ploty': ploty}

    return ret
#---------------------------------------------------------------------------------------------

#-------------------------------------Draw Line--------------------------------------------------------
# with fillPoly function draw a polygon including left and right lane
# through the pts_mean, we could know the degree of curvature between the lane and lane
# from the function warpping, using the value minb, to the perspectived image, with addWeighted function, finish the work by combining the color of polygon lightly.
def draw_lane_lines(original_image, warped_image, Minv, draw_info):
    left_fitx = draw_info['left_fitx']
    right_fitx = draw_info['right_fitx']
    ploty = draw_info['ploty']

    warp_zero = np.zeros_like(warped_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    mean_x = np.mean((left_fitx, right_fitx), axis=0)
    pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])

    cv2.fillPoly(color_warp, np.int_([pts]), (216, 168, 74))
    cv2.fillPoly(color_warp, np.int_([pts_mean]), (216, 168, 74))

    newwarp = cv2.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0]))
    result = cv2.addWeighted(original_image, 1, newwarp, 0.4, 0)

    return pts_mean, result
#---------------------------------------------------------------------------------------------
#Implenting for detection and tracking



#---------------------------------------------------------------------------------------------

# Initialize count
count_f = 0
count_r = 0
#center_points_prev_frame = []
center_points_prev_frame_f = []
center_points_prev_frame_r = []

#tracking_objects = {}
#track_id = 0

tracking_objects_f = {}
track_id_f = 0

tracking_objects_r = {}
track_id_r = 0



while True:
    success, img_lane = cap_front.read()
    success, frame_front = cap_front.read()
    success, frame_rear = cap_rear.read()

    # ------Tracking Part_1------
    #count += 1
    count_f += 1
    count_r += 1
    # Point current frame
    #center_points_cur_frame = []
    center_points_cur_frame_f = []
    center_points_cur_frame_r = []
    # ------Tracking Part_1------


# Front Camera Lane Detection
    ## wrapped video show (bird eye-version)
    wrapped_img, minverse = wrapping(img_lane)
    #cv2.imshow('wrapped', wrapped_img)
    # color filter and mask (yellow and white lane)
    w_f_img = color_filter(wrapped_img)
    #cv2.imshow('w_f_img', w_f_img)
    ##ROI from color filtered image
    w_f_r_img = roi(w_f_img)
    #cv2.imshow('w_f_r_img', w_f_r_img)
    ## ROI and wrapped img threshold
    _gray = cv2.cvtColor(w_f_r_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(_gray, 140, 195, cv2.THRESH_BINARY)
    #cv2.imshow('threshold', thresh)
    ## 선 분포도 조사 histogram
    leftbase, rightbase = plothistogram(thresh)
    ## histogram 기반 window roi 영역
    draw_info = slide_window_search(thresh, leftbase, rightbase)
    ## 원본 이미지에 라인 넣기
    meanPts, frame_front = draw_lane_lines(frame_front, thresh, minverse, draw_info)


# Vehicle Detection with ML_Model with input video    
    results_front = model_n(frame_front, stream=True)
    results_rear = model_n(frame_rear, stream=True)

    
# Vehicle Detection for front camera
    for rf in results_front:
        boxesf = rf.boxes
        for boxf in boxesf:

            # Bounding Box
            x1,y1,x2,y2 = boxf.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(frame_front, (x1,y1), (x2,y2), (0, 0, 255, 127), -1)
            w, h = x2 - x1, y2 - y1
            # rt = -1 -> fullfilled rectangle, 0~ -> normal thickness
            cvzone.cornerRect(frame_front, (x1, y1, w, h), rt=1, colorR=(0, 0, 255))
            # Confidence
            confidence = math.ceil((boxf.conf[0] * 100)) / 100
            
            # Class Name
            cls = int(boxf.cls[0])

            #cvzone.putTextRect(frame_front, f'{classNames[cls]} {confidence}', (max(0, x1), max(35, y1))) # , scale = 3, thickness = 3

            # ------Tracking Part_2------
            # Assuming boxr.xyxy[0] gives you the coordinates in the form (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, boxf.xyxy[0])
            # Calculate the width and height from the coordinates
            w = x2 - x1
            h = y2 - y1

            # cv2.rectangle(frame_rear, (x1, y1), (x2, y2), (0, 0, 255), 1)

            # The center point of the box
            cx = x1 + w // 2
            cy = y1 + h // 2

            center_points_cur_frame_f.append((cx, cy))
            print("FRAME Numb.", count_f, " ", x1, y1, w, h)

            cv2.circle(frame_front, (cx, cy), 5, (0, 0, 255), -1)
            #cv2.rectangle(frame_front, (x1, y1), (x1 + w, y1 + h), (0, 255, 255), 2)

            # Only at the beginning we compare previous and current frame
            if count_f <= 2:
                for pt in center_points_cur_frame_f:
                    #    cv2.circle(frame, pt, 5, (0,0,255), -1)
                    for pt2 in center_points_prev_frame_f:
                        distance_f = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                        if distance_f < 30:
                            tracking_objects_f[track_id_f] = pt
                            track_id_f += 1
            else:

                tracking_objects_copy = tracking_objects_f.copy()
                center_points_cur_frame_copy = center_points_cur_frame_f.copy()

                for object_id, pt2 in tracking_objects_copy.items():
                    object_exists = False
                    for pt in center_points_cur_frame_copy:
                        distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                        # Update object(IDs) position
                        if distance < 30:
                            tracking_objects_f[object_id] = pt
                            object_exists = True
                            if pt in center_points_cur_frame_f:
                                center_points_cur_frame_f.remove(pt)
                                continue

                    # if object is disappear, then id is also should be disappear. Remove ID Lost
                    if not object_exists:
                        tracking_objects_f.pop(object_id)

                # Add new IDs found
                for pt in center_points_cur_frame_f:
                    tracking_objects_f[track_id_f] = pt
                    track_id_f += 1

            for object_id, pt in tracking_objects_f.items():
                cv2.circle(frame_front, pt, 5, (0, 0, 255), -1)
                cv2.putText(frame_front, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 1)

            #print("Tracking Object")
            #print(tracking_objects)

            #print("CUR FRAME")
            #print(center_points_cur_frame)

            #print("PREV FRAME")
            #print(center_points_prev_frame)

            #cv2.imshow("Frame", frame_front)

            # Make a copy of the points
            center_points_prev_frame = center_points_cur_frame_f.copy()


            cvzone.putTextRect(frame_front, f'{classNames[cls]} {confidence}',(max(0, x1), max(35, y1)), thickness=1, scale=1, offset=5)  # , scale = 3, thickness = 3


# Vehicle Detection for rear camera
    for rr in results_rear:
        boxesr = rr.boxes
        for boxr in boxesr:
            x3,y3,x4,y4 = boxr.xyxy[0]
            x3,y3,x4,y4 = int(x3), int(y3), int(x4), int(y4)
            # last parameter -1 -> fullfilled rectangle, 0~ -> normal thickness
            cv2.rectangle(frame_rear, (x3,y3), (x4,y4), (0, 0, 255), 1)
            # alpha = 0.4
            # opacity_rear = cv2.addWeighted(frame_rear, alpha, frame_rear, 1-alpha,0)

        # ------Tracking Part_2------
            # Assuming boxr.xyxy[0] gives you the coordinates in the form (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, boxr.xyxy[0])
            # Calculate the width and height from the coordinates
            w = x2 - x1
            h = y2 - y1

            #cv2.rectangle(frame_rear, (x1, y1), (x2, y2), (0, 0, 255), 1)

            # The center point of the box
            cx = x1 + w // 2
            cy = y1 + h // 2

            center_points_cur_frame_r.append((cx, cy))
            print("FRAME Numb.", count_r, " ", x1, y1, w, h)

            cv2.circle(frame_rear, (cx, cy), 5, (0, 0, 255), -1)
            #cv2.rectangle(frame_rear, (x1, y1), (x1 + w, y1 + h), (0, 255, 255), 2)

            # Only at the beginning we compare previous and current frame
            if count_r <= 2:
                for pt in center_points_cur_frame_r:
                    #    cv2.circle(frame, pt, 5, (0,0,255), -1)
                    for pt2 in center_points_prev_frame_r:
                        distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                        if distance < 100:
                            tracking_objects_r[track_id_r] = pt
                            track_id_r += 1
            else:

                tracking_objects_copy = tracking_objects_r.copy()
                center_points_cur_frame_copy = center_points_cur_frame_r.copy()

                for object_id, pt2 in tracking_objects_copy.items():
                    object_exists = False
                    for pt in center_points_cur_frame_copy:
                        distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                        # Update object(IDs) position
                        if distance < 100:
                            tracking_objects_r[object_id] = pt
                            object_exists = True
                            if pt in center_points_cur_frame_r:
                                center_points_cur_frame_r.remove(pt)
                                continue

                    # if object is disappear, then id is also should be disappear. Remove ID Lost
                    if not object_exists:
                        tracking_objects_r.pop(object_id)

                # Add new IDs found
                for pt in center_points_cur_frame_r:
                    tracking_objects_r[track_id_r] = pt
                    track_id_r += 1

            for object_id, pt in tracking_objects_r.items():
                cv2.circle(frame_rear, pt, 5, (0, 0, 255), -1)
                cv2.putText(frame_rear, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 1)

            #print("Tracking Object")
            #print(tracking_objects)

            #print("CUR FRAME")
            #print(center_points_cur_frame)

            #print("PREV FRAME")
            #print(center_points_prev_frame)

            #cv2.imshow("Frame", frame_front)

            # Make a copy of the points
            center_points_prev_frame = center_points_cur_frame_r.copy()




    cv2.imshow("frame_front",frame_front)
    cv2.imshow("frame_rear",frame_rear)
    #cv2.imshow("img_lane",img_lane)
    

    if cv2.waitKey(1) == ord('q'):
        break

cap_front.release()
cap_rear.release()
cv2.destroyAllWindows()