from ultralytics import YOLO
import cv2
#import cvzone
import math
import pandas
import numpy as np
import glob
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tracking_function import *

'''
August
#ToDo: 1) Risk judgement Algorithm_(X)( 4) using two camera), 2) Bounding Box Interface_(X), 3) curve and object acceleration control. (Steering) 19) Output Image Process Update
September_1
#ToDo: 5) Multiple Lane Detection 6) Traffic Signal 7) CUDA 8) M.A.P,(Canny Mask? or white and yellow color ) 
September_2
#ToDo: 9) Advanced Lane Detection
September_3
#ToDo: 10) Unity Simulation 11) ROS Simulation
September_4
#ToDo: 12) Unity Controllable Simulation
October_1
#ToDo: 13) Data Transfer with Unity 14) Data Transfer with ROS
November_3
#ToDo: 15) VR Environment Setting
November_4
#ToDo: 16) Unity Simulation with VR
December_1
#ToDo: 17) Jetson Environment Setting and Testing
December_2
#ToDo: 18) Making a Film
'''

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
model = YOLO('yolov8n.pt')  # load the official pretrained model (recommended for training)


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
    if event == cv2.EVENT_MOUSEMOVE:
        cursor_Coordinate = [x, y]
        print(cursor_Coordinate)

# Cursor Coordinate of front Camera
cv2.namedWindow('camera_front_input')                           # in ' ' should be filled by name of display window
cv2.setMouseCallback('camera_front_input', cursor_Coordinate)   # in ' ' should be filled by name of display window
# Cursor Coordinate of rear Camera
# cv2.namedWindow('camera_rear_input')                           # in ' ' should be filled by name of display window
# cv2.setMouseCallback('camera_rear_input', cursor_Coordinate)   # in ' ' should be filled by name of display window

#----------------------------------Object Detection----------------------------------#
# Based on (https://github.com/freedomwebtech/yolov8counting-trackingvehicles)
# Based on (https://github.com/murtazahassan)
# Make a Class Array
open_coco = open("coco.txt", "r")
read_coco = open_coco.read()
class_list = read_coco.split("\n")

## Object Tracking
# Initialize count frame
count_f = 0  # intialize for front camera video frame count
count_r = 0  # intialize for rear camera video frame count
# Call the class for tracking of front and rear video
tracker_f = Tracker()  # Tracking class call for front camera
tracker_r = Tracker()

#----------------------------------Lane Detection----------------------------------#
'''
Workflow of Lane Detection
distortion_factors[distorted Lane -> undistorted Lane]
-> Wrapping [Bird-Eye Effect]
-> Color Filter
->ROI because extracted Lane(perspetive view), but there are noise, (e.g. name, pointer...)
Based on (https://moon-coco.tistory.com/entry/OpenCV%EC%B0%A8%EC%84%A0-%EC%9D%B8%EC%8B%9D)
'''

### Step 1: Distortion or amera Calibration ###
# current cheap camera makes a distortion of images,
# main distortions are radial- / tangential - distortion.
def camera_calibration():
    # Prepare object points
    # From the provided calibration images, 9*6 corners are identified
    nx = 11  # 9 , 11, number of chessboard's horizontal pattern -1
    ny = 8  # 6 , 8, number of chessboard's vertical pattern -1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objpoints = []  # Object points are real world points, here a 3D coordinates matrix is generated
    imgpoints = []  # image points are xy plane points, here a 2D coordinates matrix is generated (z = 0, chessboard)
    objp = np.zeros((8 * 11, 3), np.float32)
    objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)

    # Make a list of calibration images
    os.listdir("camera_cal/")
    cal_img_list = os.listdir("camera_cal/")

    # Imagepoints are the coresspondant object points with their coordinates in the distorted image
    # They are found in the image using the Open CV 'findChessboardCorners' function
    for image_name in cal_img_list:
        import_from = 'camera_cal/' + image_name
        img = cv2.imread(import_from)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        # If found, draw corners
        if ret == True:
            # Draw and display the corners
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(img, (nx, ny), corners2, ret)  #
            imgpoints.append(corners2)
            objpoints.append(objp)
            print(objpoints)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    #Output Explain (https://foss4g.tistory.com/1665)

    ###################################
    ## checking the undistored image ##
    ###################################
    # for img_name in cal_img_list:
    #     import_from = 'camera_cal/' + img_name
    #     img = cv2.imread(import_from)
    #     undist = cv2.undistort(img, mtx, dist, None, mtx)
    #     export_to = 'camera_cal_undistorted/' + img_name
    #     #save the image in the destination folder#
    #     plt.imsave(export_to, undist)

    return mtx, dist

### STEP 2: Perspective Transform from Car Camera to Bird's Eye View ###
# Idea : View of from vertical side of street, instead of driver's view
# img_width = 1280
# img_heigt = 720

    ### Source Point ###
x_top_left_src = 480        #(x,y)##
y_top_left_src = 390        #

x_top_right_src = 565       ###(x,y)#
y_top_right_src = 390               #

x_bottom_left_src = 110     #
y_bottom_left_src = 690     #(x,y)###

x_bottom_right_src = 885            #
y_bottom_right_src = 690    ###(x,y)#

    ### Destination Point ###
x_top_left_dst = 55        #(x,y)##
y_top_left_dst = 0          #

x_top_right_dst = 1035       ###(x,y)#
y_top_right_dst = 0                 #

x_bottom_left_dst = 150     #
y_bottom_left_dst = 720     #(x,y)###

x_bottom_right_dst = 880            #
y_bottom_right_dst = 720    ###(x,y)#
def wrapping(image):
    if image is None:
        print('Image is None, skippung this iteration')
        return None, None

    h = image.shape[0]
    w = image.shape[1]
    img_size = (w, h)
    # print(img_size)
    offset = 150

    # source = np.float32([[w // 2 - 30, h * 0.53], [w // 2 + 60, h * 0.53], [w * 0.3, h], [w, h]])
    # destination = np.float32([[0, 0], [w-350, 0], [400, h], [w-150, h]])


    source = np.float32(
        [
            (x_bottom_left_src, y_bottom_left_src),     # bottom-left corner
            (x_top_left_src, y_top_left_src),           # top-left corner
            (x_top_right_src, y_top_right_src),         # top-right corner
            (x_bottom_right_src, y_bottom_right_src)    # bottom-right corner
        ])

    destination = np.float32(
        [
            (x_bottom_left_dst, y_bottom_left_dst),     # bottom-left corner
            (x_top_left_dst, y_top_left_dst),           # top-left corner
            (x_top_right_dst, y_top_right_dst),         # top-right corner
            (x_bottom_right_dst, y_bottom_right_dst)    # bottom-right corner
        ])

    # getPerspectiveTransformation? the properties that it hold the property of linear, but not the property of parallelity
    # for example, train lanes are parallel but through the perspective transformation, it looks like they are meeing at the end of point
    # we need 4 point of input and moving point of output
    # for the transformation matrix we need, through the cv2.getPerspectiveTransform() function and adjust our transformation matrix to cv2.warpPerspective() function, we could have a final image
    #
    transform_matrix = cv2.getPerspectiveTransform(source, destination)
    minv = cv2.getPerspectiveTransform(destination, source)
    _image = cv2.warpPerspective(image, transform_matrix, (image.shape[1], image.shape[0]))

    return _image, minv

# -------------------------------------Color Filter (using HLS)--------------------------------------------------------
# HLS(Hue, Luminanse, Saturation) :
# lower = ([minimum_blue, m_green, m_red])
# upper = ([Maximum_blue, M_green, M_red])
def color_filter(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    # https://stackoverflow.com/questions/55822409/hsl-range-for-yellow-lane-lines
    # White Filter
    # white_lower = np.array([20, 150, 20])
    # white_upper = np.array([255, 255, 255])
    # White-ish areas in image
    # H value can be arbitrary, thus within [0 ... 360] (OpenCV: [0 ... 180])
    # L value must be relatively high (we want high brightness), e.g. within [0.7 ... 1.0] (OpenCV: [0 ... 255])
    # S value must be relatively low (we want low saturation), e.g. within [0.0 ... 0.3] (OpenCV: [0 ... 255])
    white_lower = np.array([np.round(0 / 2), np.round(0.55 * 255), np.round(0.00 * 255)])
    white_upper = np.array([np.round(360 / 2), np.round(1.00 * 255), np.round(0.20 * 255)])
    white_mask = cv2.inRange(hls, white_lower, white_upper)

    # Yellow Filter
    # yellow_lower = np.array([0, 85, 81])
    # yellow_upper = np.array([190, 255, 255])
    # Yellow-ish areas in image
    # H value must be appropriate (see HSL color space), e.g. within [40 ... 60]
    # L value can be arbitrary (we want everything between bright and dark yellow), e.g. within [0.0 ... 1.0]
    # S value must be above some threshold (we want at least some saturation), e.g. within [0.35 ... 1.0]
    yellow_lower = np.array([np.round(40 / 2), np.round(0.00 * 255), np.round(0.35 * 255)])
    yellow_upper = np.array([np.round(60 / 2), np.round(1.00 * 255), np.round(1.00 * 255)])
    #yellow_lower = np.array([np.round(40 / 2), np.round(0.75 * 255), np.round(0.00 * 255)])
    #yellow_upper = np.array([np.round(60 / 2), np.round(1.00 * 255), np.round(0.30 * 255)])
    yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)

    # Do filtering the each yellow lane and white lane,
    # Bitwise_or makes (yellow line and white line) combining -> mask
    # bitwise_and maeks (original image and mask) -> then left just masked part -> masked
    # yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)
    # white_mask = cv2.inRange(hls, white_lower, white_upper)
    # mask = cv2.bitwise_or(yellow_mask, white_mask)
    # masked = cv2.bitwise_and(image, image, mask=mask)
    binary = cv2.bitwise_or(yellow_mask, white_mask)

    # return masked
    # for second lane detection method, mask(bitwise_or) is added or selected, original way is with masked!
    return binary


"""
def binary_thresholded(img):
    # Transform image to gray scale
    gray_img =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply sobel (derivative) in x direction, this is usefull to detect lines that tend to be vertical
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    # Scale result to 0-255
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    sx_binary = np.zeros_like(scaled_sobel)
    # Keep only derivative values that are in the margin of interest
    sx_binary[(scaled_sobel >= 30) & (scaled_sobel <= 255)] = 1

    # Detect pixels that are white in the grayscale image
    white_binary = np.zeros_like(gray_img)
    white_binary[(gray_img > 200) & (gray_img <= 255)] = 1  # 200,255

    # Convert image to HLS
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    H = hls[:, :, 0]
    S = hls[:, :, 2]
    sat_binary = np.zeros_like(S)
    # Detect pixels that have a high saturation value
    sat_binary[(S > 200) & (S <= 255)] = 1  # 90 , 255

    hue_binary = np.zeros_like(H)
    # Detect pixels that are yellow using the hue component
    hue_binary[(H > 15) & (H <= 25)] = 1  # 10, 25

    # Combine all pixels detected above
    binary_1 = cv2.bitwise_or(sx_binary, white_binary)
    binary_2 = cv2.bitwise_or(hue_binary, sat_binary)
    binary = cv2.bitwise_or(binary_1, binary_2)
    # plt.imshow(binary, cmap='gray')

    return binary
"""


# -------------------------------------ROI--------------------------------------------------------
# Another Idea is first ROI, in order to decrease the input area, then we don't need anymore, sky etc...
# but in this case we need to set a wrapping area, so it is not so effective, but in case SortDeep.. could be...

def roi(image):
    x = int(image.shape[1])
    y = int(image.shape[0])
    # height, width, number of channels in image
    # height = img.shape[0]
    # width = img.shape[1]
    # channels = img.shape[2]
    # Height represents the number of pixel rows in the image or the number of pixels in each column of the image array.
    # Width represents the number of pixel columns in the image or the number of pixels in each row of the image array.
    # Number of Channels represents the number of components used to represent each pixel.
    # In the above example, Number of Channels = 4 represent Alpha, Red, Green and Blue channels.
    # *** here traffic sign on the street is deleted and ignored, if you don't wanna that, modify the ROI part.
    # 한 붓 그리기

                    ### R.O.I Area ###

                    #2###3      6####7
                    ######      ######
                    ######      ######
                    ######      ######
                    ######      ######
                    ######      ######
                    ######      ######
                    #####4######5#####
                    #1#9#############8
    _shape = np.array([
        [int(0.05 * x), int(0.95 * y)],  #1
        [int(0.05 * x), int(0.01 * y)],  #2
        [int(0.45 * x), int(0.01 * y)],  #3
        [int(0.45 * x), int(0.94 * y)],  #4
        [int(0.60 * x), int(0.94 * y)],  #5
        [int(0.60 * x), int(0.01 * y)],  #6
        [int(0.95 * x), int(0.01 * y)],  #7
        [int(0.95 * x), int(0.95 * y)],  #8
        [int(0.11 * x), int(0.95 * y)]   #9
    ])

    mask = np.zeros_like(image)

    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, np.int32([_shape]), ignore_mask_color)
    #cv2.fillPoly(mask, np.float32([_shape]), ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image

# ---------------------------------------------------------------------------------------------


# -------------------------------------Window ROI--------------------------------------------------------
# why not cv2.HoughLines() and cv2.HoughLinesP()? -> these functions are heavy and detection is not exact for curve.
# left_current = a biggest index of image's left side (coordinate information)
# good_left = save the part just in window
# next left_current of window is mean value of index, that good_left of nonzero_x have, if godd_left length is shorter than 50.
# np.concatenate : Array makes 1.Dimenstion array
# np.trunc : throw away a decimal part

#def slide_window_search(binary_warped, left_current, right_current):
def slide_window_search(binary_warped):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img) #need check for usage.

    #
    ret, thresh = cv2.threshold(binary_warped , 140, 195, cv2.THRESH_BINARY)

    # ---Histogram---
    # it is not histogram of opencv
    # bitwise image has one channel and value between 0 ~ 255.
    # if it is lane, they have a value near by 255, and if it isn't, then 0.
    # it means for one column, when we add all row values, if there are lane, they has relative big value, if not, small value
    # 1050 -> right lane, 350 -> left lane

    # Take a histogram of the bottom half of the image
    histogram = np.sum(thresh[thresh.shape[0] // 2:, :], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int64(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # case for nwindows = 9
    # nwindows = 9


    # case for nwindows = 4
    nwindows = 4

    # Set the width of the windows +/- margin
    # margin = int(1080 * (50 / 1920))
    # margin = int(1080 * 0.025)
    margin = 100  # or 125      # window width
    # Set minimum number of pixels found to recenter window
    # minpix = int(1080 * (20 / 1920))
    # minpix = int(1080 * 0.01)
    minpix = 50 # or 10
    # Set height of windows - based on nwindows above and image shape
    window_height = np.int64(binary_warped.shape[0] / nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()  # 선이 있는 부분의 인덱스만 저장
    nonzero_y = np.array(nonzero[0])  # 선이 있는 부분 y의 인덱스 값
    nonzero_x = np.array(nonzero[1])  # 선이 있는 부분 x의 인덱스 값

    # Current positions to be updated later for each window in nwindows
    left_current = leftx_base
    right_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane = []
    right_lane = []
    color = [0, 255, 0]
    thickness = 2

    # Step through the windows one by one
    for w in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (w + 1) * window_height  # window 윗부분
        win_y_high = binary_warped.shape[0] - w * window_height  # window 아랫 부분
        win_xleft_low = left_current - margin  # 왼쪽 window 왼쪽 위
        win_xleft_high = left_current + margin  # 왼쪽 window 오른쪽 아래
        win_xright_low = right_current - margin  # 오른쪽 window 왼쪽 위
        win_xright_high = right_current + margin  # 오른쪽 window 오른쪽 아래

        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), color, thickness)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), color, thickness)
        # window 안에 있는 부분만을 저장
        # Identify the nonzero pixels in x and y within the window #
        good_left = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) & (
                    nonzero_x < win_xleft_high)).nonzero()[0]
        good_right = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) & (
                    nonzero_x < win_xright_high)).nonzero()[0]
        left_lane.append(good_left)
        right_lane.append(good_right)
        # cv2.imshow("oo", out_img)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left) > minpix:
            left_current = np.int64(np.mean(nonzero_x[good_left]))
        if len(good_right) > minpix:
            right_current = np.int64(np.mean(nonzero_x[good_right]))

        # ## if scan windows added
        # cv2.rectangle(window_img,(win_xleft_high,win_y_high),(win_xleft_low,win_y_low),(255,255,255),3)
        # cv2.rectangle(window_img,(win_xright_high,win_y_high),(win_xright_low,win_y_low),(255,255,255),3)
        # plt.imshow(window_img)
        # plt.show

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    #left_lane = np.concatenate(left_lane)  # np.concatenate() -> array를 1차원으로 합침 (array를 1차원 배열로 만들어줌)
    #right_lane = np.concatenate(right_lane)
    try:
        # For subsequent processing, it's easier to work with a single 1D array of indices rather than a list of arrays. That's why np.concatenate is used.
        left_lane = np.concatenate(left_lane)  # np.concatenate() -> array를 1차원으로 합침 (array를 1차원 배열로 만들어줌)
        right_lane = np.concatenate(right_lane)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    """
    # Suppose the x-coordinates of all non-zero pixels are
    nonzero_x = np.array([10, 20, 30, 40, 50])
    # And we have identified that the 1st and 4th pixels belong to the left lane (0-based index)
    left_lane = np.array([0, 3])
    # Then, extracting the x-coordinates of these pixels would be
    leftx = nonzero_x[left_lane]  # This will give us array([10, 40])
    """
    # Extract left and right line pixel positions
    leftx = nonzero_x[left_lane]
    lefty = nonzero_y[left_lane]
    rightx = nonzero_x[right_lane]
    righty = nonzero_y[right_lane]

    out_img[nonzero_y[left_lane], nonzero_x[left_lane]] = [255, 0, 0]
    out_img[nonzero_y[right_lane], nonzero_x[right_lane]] = [0, 0, 255]

    # plt.imshow(out_img)
    # plt.plot(left_fitx, ploty, color = 'yellow')
    # plt.plot(right_fitx, ploty, color = 'yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    # plt.show()

    return leftx, lefty, rightx, righty


def fit_poly(binary_warped, leftx, lefty, rightx, righty):
    ### Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    #left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    #right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty
        # we have to discuss about that later, whether we should do deleting comma or not.
        # ltx = np.trunc(left_fitx)  # np.trunc() -> 소수점 부분을 버림
        # rtx = np.trunc(right_fitx)

    return left_fit, right_fit, left_fitx, right_fitx, ploty



# ---------------------------------------------------------------------------------------------

# -------------------------------------Draw Line--------------------------------------------------------
# with fillPoly function draw a polygon including left and right lane
# through the pts_mean, we could know the degree of curvature between the lane and lane
# from the function warpping, using the value minb, to the perspectived image, with addWeighted function, finish the work by combining the color of polygon lightly.
#def draw_lane_lines(original_image, warped_image, Minv, draw_info):
#def draw_lane_lines(original_image, warped_image, Minv, left_fitx, right_fitx, ploty):
def draw_lane_lines(binary_warped, left_fitx, right_fitx, ploty):
    #left_fitx = draw_info['left_fitx']
    #right_fitx = draw_info['right_fitx']
    #ploty = draw_info['ploty']

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)

    # margin = int(1080 * (100 / 1920))
    #margin = int(1080 * 0.01)
    margin = 50
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))

    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # The commented part below can be said to have been omitted compared to the code above.
    #pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    #pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    #pts = np.hstack((pts_left, pts_right))

    ## if there are obstacle, then here color change, and return value add
    #cv2.fillPoly(color_warp, np.int_([pts]), (216, 168, 74))
    #cv2.fillPoly(color_warp, np.int_([pts_mean]), (216, 168, 74))
    # actually below part should be positioned at the just before return, but below part is commented,
    # actually it is strong related with just above part, it located here, but if you want to use this, then should be move!
    #newwarp = cv2.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0]))
    #result = cv2.addWeighted(original_image, 1, newwarp, 0.4, 0)

    mean_x = np.mean((left_fitx, right_fitx), axis=0)
    # using pts_mean, we get a degree of curve for center value between the two lanes
    pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])

    # Center Line added ###
    center_line_pts = (left_line_pts + right_line_pts) / 2

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (100, 100, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (100, 100, 0))
    cv2.fillPoly(window_img, np.int_([center_line_pts]), (200, 100, 0))

    result = cv2.addWeighted(out_img, 1, window_img, 0.9, 0)  # (0.3)

    # Plot the polynomial lines onto the image
    # plt.plot(left_fitx, ploty, color='green')
    # plt.plot(right_fitx, ploty, color='blue')
    ## End visualization steps ##

    return pts_mean, result

### STEP 5: Detection of Lane Lines Based on Previous Step ###
def find_lane_pixels_using_prev_poly(binary_warped):
    # global prev_left_fit
    # global prev_right_fit

    # width of the margin around the previous polynomial to search
    #margin = int(1080 * 0.01)
    margin = 50
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    ### Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    left_lane_inds = ((nonzerox > (prev_left_fit[0] * (nonzeroy ** 2) + prev_left_fit[1] * nonzeroy +
                                   prev_left_fit[2] - margin)) & (nonzerox < (prev_left_fit[0] * (nonzeroy ** 2) +
                                                                              prev_left_fit[1] * nonzeroy +
                                                                              prev_left_fit[2] + margin))).nonzero()[0]
    right_lane_inds = ((nonzerox > (prev_right_fit[0] * (nonzeroy ** 2) + prev_right_fit[1] * nonzeroy +
                                    prev_right_fit[2] - margin)) & (nonzerox < (prev_right_fit[0] * (nonzeroy ** 2) +
                                                                                prev_right_fit[1] * nonzeroy +
                                                                                prev_right_fit[2] + margin))).nonzero()[
        0]
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty

### STEP 6: Calculate Vehicle Position and Curve Radius ###

def measure_curvature_meters(binary_warped, left_fitx, right_fitx, ploty):
    # Define conversions in x and y from pixels space to meters
    # ym_per_pix = 30/1080 # meters per pixel in y dimension
    # xm_per_pix = 3.7/1920 # meters per pixel in x dimension
    ym_per_pix = 30 / 1080 * (720 / 1080)  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 1920 * (1080 / 1920)  # meters per pixel in x dimension

    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    return left_curverad, right_curverad


def measure_position_meters(binary_warped, left_fit, right_fit):
    # Define conversion in x from pixels space to meters
    #xm_per_pix = 3.7 / 1920 * (1080 / 1920)  # meters per pixel in x dimension
    xm_per_pix = 3.7 / 1080 * (720 / 1080)  # meters per pixel in x dimension
    # Choose the y value corresponding to the bottom of the image
    y_max = binary_warped.shape[0]
    # Calculate left and right line positions at the bottom of the image
    left_x_pos = left_fit[0] * y_max ** 2 + left_fit[1] * y_max + left_fit[2]
    right_x_pos = right_fit[0] * y_max ** 2 + right_fit[1] * y_max + right_fit[2]
    # Calculate the x position of the center of the lane
    center_lanes_x_pos = (left_x_pos + right_x_pos) // 2
    # Calculate the deviation between the center of the lane and the center of the picture
    # The car is assumed to be placed in the center of the picture
    # If the deviation is negative, the car is on the felt hand side of the center of the lane
    veh_pos = ((binary_warped.shape[1] // 2) - center_lanes_x_pos) * xm_per_pix
    return veh_pos


### STEP 7: Project Lane Delimitations Back on Image Plane and Add Text for Lane Info ###

def project_lane_info(img, binary_warped, ploty, left_fitx, right_fitx, M_inv, left_curverad, right_curverad, veh_pos):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Center Line modified
    #margin = 400 * (1080 / 1920)
    # margin = 400 * 0.1
    margin = 30
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])

    pts_left_c = np.array([np.transpose(np.vstack([left_fitx + margin, ploty]))])
    pts_right_c = np.array([np.flipud(np.transpose(np.vstack([right_fitx - margin, ploty])))])
    pts = np.hstack((pts_left_c, pts_right_c))

    # margin value test...
    #pts_left_i = np.array([np.transpose(np.vstack([left_fitx + margin + 150, ploty]))])
    pts_left_i = np.array([np.transpose(np.vstack([left_fitx + margin + 10, ploty]))])
    #pts_right_i = np.array([np.flipud(np.transpose(np.vstack([right_fitx - margin - 150, ploty])))])
    pts_right_i = np.array([np.flipud(np.transpose(np.vstack([right_fitx - margin - 10, ploty])))])
    pts_i = np.hstack((pts_left_i, pts_right_i))

    # Draw the lane onto the warped blank image
    colorwarp_img = cv2.polylines(color_warp, np.int_([pts_left]), False, (0, 0, 255), 30)
    colorwarp_img = cv2.polylines(color_warp, np.int_([pts_right]), False, (0, 0, 255), 30)
    colorwarp_img = cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    # colorwarp_img=cv2.fillPoly(color_warp, np.int_([pts_i]), (0,0, 255))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, M_inv, (img.shape[1], img.shape[0]))

    # Combine the result with the original image
    out_img = cv2.addWeighted(img, 0.7, newwarp, 0.3, 0)

    cv2.putText(out_img, 'Curve Radius [m]: ' + str((left_curverad + right_curverad) / 2)[:7], (5, 80),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out_img, 'Center Offset [m]: ' + str(veh_pos)[:7], (5, 110), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                (255, 255, 255), 2, cv2.LINE_AA)

    return out_img, colorwarp_img, newwarp


### STEP 8: Lane Finding Pipeline on Video ###

#def lane_finding_pipeline(img, init, mts, dist):
def lane_finding_pipeline(img, init):
    global left_fit_hist
    global right_fit_hist
    global prev_left_fit
    global prev_right_fit

    if init:
        left_fit_hist = np.array([])
        right_fit_hist = np.array([])
        prev_left_fit = np.array([])
        prev_right_fit = np.array([])

    # first warpping? then binary color change? -> ii), first color change and then warp? -> i) check!
    # i)
    binary_thresh = color_filter(img)
    #if calibration is adjusted below, if without camera calibration, then below below
    #binary_warped, M_inv, _ = wrapping(binary_thresh, mts, dist)
    binary_warped, M_inv = wrapping(binary_thresh)
    # ii)
    #binary_warped, M_inv = wrapping(img)
    #binary_thresh = color_filter(binary_warped)

    ## checking ###
    binary_thresh_s = np.dstack((binary_thresh, binary_thresh, binary_thresh)) * 255
    binary_warped_s = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    if (len(left_fit_hist) == 0):
        # here thresh is added because of previous method, if it is not necessary, it could be deleted
        leftx, lefty, rightx, righty = slide_window_search(binary_warped)
        left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(binary_warped, leftx, lefty, rightx, righty)
        # Store fit in history
        left_fit_hist = np.array(left_fit)
        new_left_fit = np.array(left_fit)
        left_fit_hist = np.vstack([left_fit_hist, new_left_fit])
        right_fit_hist = np.array(right_fit)
        new_right_fit = np.array(right_fit)
        right_fit_hist = np.vstack([right_fit_hist, new_right_fit])

    else:
        prev_left_fit = [np.mean(left_fit_hist[:, 0]), np.mean(left_fit_hist[:, 1]), np.mean(left_fit_hist[:, 2])]
        prev_right_fit = [np.mean(right_fit_hist[:, 0]), np.mean(right_fit_hist[:, 1]), np.mean(right_fit_hist[:, 2])]
        leftx, lefty, rightx, righty = find_lane_pixels_using_prev_poly(binary_warped)
        if (len(lefty) == 0 or len(righty) == 0):
            leftx, lefty, rightx, righty = slide_window_search(binary_warped)
        left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(binary_warped, leftx, lefty, rightx, righty)

        # Add new values to history
        new_left_fit = np.array(left_fit)
        left_fit_hist = np.vstack([left_fit_hist, new_left_fit])
        new_right_fit = np.array(right_fit)
        right_fit_hist = np.vstack([right_fit_hist, new_right_fit])

        # Remove old values from history
        if (len(left_fit_hist) > 5):  # 10
            left_fit_hist = np.delete(left_fit_hist, 0, 0)
            right_fit_hist = np.delete(right_fit_hist, 0, 0)

    ### chekcing ###
    # draw_poly_img : to draw the detected lane lines on a "bird's-eye view" image., pts_mean : it is for location of curve, but consider deleting
    pts_mean, draw_poly_img = draw_lane_lines(binary_warped, left_fitx, right_fitx, ploty)

    left_curverad, right_curverad = measure_curvature_meters(binary_warped, left_fitx, right_fitx, ploty)
    # measure_curvature_meters(binary_warped, left_fitx, right_fitx, ploty)
    veh_pos = measure_position_meters(binary_warped, left_fit, right_fit)
    out_img, colorwarp_img, newwarp = project_lane_info(img, binary_warped, ploty, left_fitx, right_fitx, M_inv,
                                                        left_curverad, right_curverad, veh_pos)
    return out_img, veh_pos, colorwarp_img, draw_poly_img

# ---------------------------------------------------------------------------------------------





















# Video Settings
fps = 24.0  # Viedo Frame
frame_size = (1080, 720)  # Video Size
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec using VideoWriter_fourcc
Font = cv2.FONT_HERSHEY_COMPLEX
FontSize = 0.4

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
video_front_raw_input = cv2.VideoCapture("D:\ProjectAsurada\ProjectAsurada\VideoSample\Front_driving_vm.mp4")
# Video Frame Size(Front)
frame_size_raw_front = (
int(video_front_raw_input.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_front_raw_input.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# fps_video_front_raw_input = video_front_raw_input.get(cv2.CAP_PROP_FPS)    # Input video FPS Check

# rear camera capture
video_rear_raw_input = cv2.VideoCapture("D:\ProjectAsurada\ProjectAsurada\VideoSample\Rear-driving.mp4")
# Video Frame Size(Rear)
frame_size_rear = (
int(video_rear_raw_input.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_rear_raw_input.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# fps_video_rear_raw_input = video_rear_raw_input.get(cv2.CAP_PROP_FPS)
####################################################

## Video Output Part
### Webcam Mode

### Video Mode
output_front = cv2.VideoWriter('video_front_output.mp4', fourcc, fps, frame_size)  # video export
output_rear = cv2.VideoWriter('video_rear_output.mp4', fourcc, fps, frame_size)
# output_front_test = cv2.VideoWriter('output_video_front_test.mp4', fourcc, fps, frame_size)
# output_rear_test = cv2.VideoWriter('output_video_rear_test.mp4', fourcc, fps, frame_size)

init = True
#mtx, dist = camera_calibration()

while True:
    # Video Input Read (Front)
    ret_front, frame_front = video_front_raw_input.read()  # ret = return, if ret false, loop will be closing
    # Video Input Read (Rear)
    ret_rear, frame_rear = video_rear_raw_input.read()
    #############################if...else#############################
    # if ret_front and ret_rear:   # Check if frames were read correctly
    #    ...
    #    if cv2.waitKey(1) == ord('q'):
    #        break
    # else:
    #    break
    ###################################################################
    # Video Frame Resize (Standardization of various video inputs)
    video_front_resize_input = cv2.resize(frame_front, frame_size)
    video_rear_resize_input = cv2.resize(frame_rear, frame_size)
    # ...

    # -----------------Object Detection and Tracking Part-----------------
    # One Frame Count Up
    count_f += 1
    count_r += 1
    ####################################################################
    # performance optimization using data reduction
    # if count_f % 3 != 0:
    #    continue
    ####################################################################

    ##########################Front Camera_Start########################
    # result_f = model.predict(video_front_resize_input, stream=True)
    # Object Detect using predict from YOLO and Input Video
    result_f = model.predict(video_front_resize_input)
    resbb_f = result_f[0].boxes.boxes
    px_f = pandas.DataFrame(resbb_f).astype("float")  # all detected vehicles's list in px

    list_f = []  # in List, save the each frame information of detected object's x1,x2,y1,y2 value

    # index is in each frame, and indexing of detected object
    # in row value or above px value,
    # 0~3 values are about coordinate of detected rectangle box, x1,y1 should be left upper, and x2,y2 should be right lower of display(1080*720)
    # 4th value is about ...
    # 5th value is class_list's id, if it is 2, it means a car, it related with our coco.txt

    for index, row in px_f.iterrows():
        x1 = int(row[0])  # left upper's x coordinate
        y1 = int(row[1])  # left upper's y coordinate
        x2 = int(row[2])  # right bottom's x coordinate
        y2 = int(row[3])  # right bottom's y coordinate
        d = int(row[5])
        c = class_list[d]  # print("class_list", c) -> if car(if d = 1), then show car(confirmed!)
        # if it is vehicles, then appended in list
        if 'car' or 'motorcycle' or 'bus' or 'truck' in c:
            list_f.append([x1, y1, x2, y2])
    bbox_id_f = tracker_f.update(list_f)
    for bbox_f in bbox_id_f:
        x3, y3, x4, y4, sd, dd, id, nr = bbox_f  # appended 4/4
        # x3,y3 :  left upper's coordinate, x4,y4 :  right bottom's coordinate
        # sd : space difference             if sd minus  then further, if sd plus closer
        # dd : distance difference          if dd minus then closer, if dd plus further
        # id : class_id
        # nr : vehicle identification nr(unsupported)
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2

        distance = float(format(math.sqrt(math.pow((540 - cx), 2) + math.pow((720 - cy), 2)), ".3f"))
        if dd ==0:
            collision_time = 2
        else:
            collision_time = distance / (abs(dd)*fps)

        #print("x3",x3,"y3",y3,"x4",x4,"y4",y4,"sd",sd,"dd",dd,"id",id,"nr",nr,"distance",distance,"collision_time",collision_time, "fps", fps)


        ###########################
        #  Type : Car (Vehicle ID)#
        #  distance(closer)       #
        #  Risk : dangerous/safe  #
        #          #####          #
        #      ####     ####      #
        #      #############      #
        #      ##         ##      #
        #                         #
        ###########################

        """
        # sd : space difference             if sd minus  then further, if sd plus closer
        # dd : distance difference          if dd minus then closer, if dd plus further
        
        # estimated collision time
        # t = D/relative speed = distance / delta dd * T = distance / delta_dd * (1/fps) = distance / (delta_dd * fps) < 1 s
        
        # low 	 Risk : sd is minus or dd is plus, then the object is moving away or is stationary.
        # medium Risk : sd is plus and dd is plus, then the object is getting closer but not directly towards your vehicle.
        # high 	 Risk : sd is plus and dd is minus, then the object is getting closer and is near the line of motion of your vehicle.
        # dangerous   : sd is plus and dd is minus and t < 1, then the object is getting closer and is near the line of motion of your vehicle 
                        and also having a dangerous potential in 1 second approachable.
        """

        if sd < 0 and dd > 0:
            # low Risk : sd is minus or dd is plus, then the object is moving away or is stationary.
            cv2.putText(video_front_resize_input, f'{"Type:"}{"not supported"}{"(Vehicle_ID)"}', (x3 + 5, y3 + 10), Font, FontSize, (0, 255, 255), 1)
            cv2.putText(video_front_resize_input, f'{distance}{"further"}', (x3 + 5, y3 + 25), Font, FontSize, (0, 255, 255), 1)
            cv2.putText(video_front_resize_input, f'{"low risk"}', (x3+5, y3 + 40), Font, FontSize, (0, 255, 255), 1)
            cv2.rectangle(video_front_resize_input, (x3, y3), (x4, y4), (0, 0, 255), 1)
            #cv2.line(video_front_resize_input, (cx, cy), (540, 719), (0, 0, 255), 2)
        elif sd >= 0 and dd > 0:
            # medium Risk : sd is plus and dd is plus, then the object is getting closer but not directly towards your vehicle.
            cv2.putText(video_front_resize_input, f'{"Type:"}{"not supported"}{"(Vehicle_ID)"}', (x3 + 5, y3 + 10), Font, FontSize, (0, 255, 255), 1)
            cv2.putText(video_front_resize_input, f'{distance}{"closer"}', (x3 + 5, y3 + 25), Font, FontSize, (0, 255, 255), 1)
            cv2.putText(video_front_resize_input, f'{"medium risk"}', (x3 + 5, y3 + 40), Font, FontSize, (0, 255, 255), 1)
            cv2.rectangle(video_front_resize_input, (x3, y3), (x4, y4), (0, 0, 255), 1)
            #cv2.line(video_front_resize_input, (cx, cy), (540, 719), (0, 0, 255), 2)
        elif sd >= 0 and dd <= 0:
            if collision_time >= 1.5:
                # high 	 Risk : sd is plus and dd is minus, then the object is getting closer and is near the line of motion of your vehicle.
                cv2.putText(video_front_resize_input, f'{"Type:"}{"not supported"}{"(Vehicle_ID)"}', (x3 + 5, y3 + 10), Font, FontSize, (0, 255, 255), 1)
                cv2.putText(video_front_resize_input, f'{distance}{"closer"}', (x3 + 5, y3 + 25), Font, FontSize, (0, 255, 255), 1)
                cv2.putText(video_front_resize_input, f'{"medium risk"}', (x3 + 5, y3 + 40), Font, FontSize, (0, 255, 255), 1)
                cv2.rectangle(video_front_resize_input, (x3, y3), (x4, y4), (0, 0, 255), 1)
                #cv2.line(video_front_resize_input, (cx, cy), (540, 719), (0, 0, 255), 2)
            else:
                # dangerous   : sd is plus and dd is minus and t < 1, then the object is getting closer and is near the line of motion of your vehicle
                # and also having a dangerous potential in 1 second approachable.
                cv2.putText(video_front_resize_input, f'{"Type:"}{"not supported"}{"(Vehicle_ID)"}', (x3 + 5, y3 + 10), Font, FontSize, (0, 255, 255), 1)
                cv2.putText(video_front_resize_input, f'{distance}{"closer"}', (x3 + 5, y3 + 25), Font, FontSize, (0, 255, 255), 1)
                cv2.putText(video_front_resize_input, f'{"high risk"}', (x3 + 5, y3 + 40), Font, FontSize, (0, 255, 255), 1)
                cv2.rectangle(video_front_resize_input, (x3, y3), (x4, y4), (0, 0, 255), 1)
                #cv2.line(video_front_resize_input, (cx, cy), (540, 719), (0, 0, 255), 2)
        else:
            # unknown ( #if sd < 0 and dd < 0: )
            cv2.putText(video_front_resize_input, f'{"Type:"}{"not supported"}{"(Vehicle_ID)"}', (x3 + 5, y3 + 10), Font, FontSize, (0, 255, 255), 1)
            cv2.putText(video_front_resize_input, f'{distance}{"calculating..."}', (x3 + 5, y3 + 25), Font, FontSize, (0, 255, 255), 1)
            cv2.putText(video_front_resize_input, f'{"unknown"}', (x3 + 5, y3 + 40), Font, FontSize, (0, 255, 255), 1)
            cv2.rectangle(video_front_resize_input, (x3, y3), (x4, y4), (0, 0, 255), 1)
            #cv2.line(video_front_resize_input, (cx, cy), (540, 719), (0, 0, 255), 2)

    ##########################Front Camera_End##########################

    ##########################Rear Camera_Start#########################
    # result_r = model.predict(video_rear_resize_input, stream=True)
    # Object Detect using predict from YOLO and Input Video
    result_r = model.predict(video_rear_resize_input)
    resbb_r = result_r[0].boxes.boxes
    px_r = pandas.DataFrame(resbb_r).astype("float")  # all detected vehicles's list in px

    list_r = []  # in List, save the each frame information of detected object's x1,x2,y1,y2 value

    # index is in each frame, and indexing of detected object
    # in row value or above px value,
    # 0~3 values are about coordinate of detected rectangle box, x1,y1 should be left upper, and x2,y2 should be right lower of display(1080*720)
    # 4th value is about ...
    # 5th value is class_list's id, if it is 2, it means a car, it related with our coco.txt
    for index, row in px_r.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]  # print("class_list", c) -> if car(if d = 1), then show car(confirmed!)
        # if it is vehicles, then appended in list
        if 'car' or 'motorcycle' or 'bus' or 'truck' in c:
            list_r.append([x1, y1, x2, y2])
    bbox_id_r = tracker_r.update(list_r)
    for bbox_r in bbox_id_r:
        x3, y3, x4, y4, sd, dd, id, nr = bbox_r  # appended 4/4
        # sd : space difference
        # dd : distance difference
        # id : class_id
        # nr : vehicle identification nr(unsupported)
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        #cv2.circle(video_front_resize_input, (455, 400), 4, (0, 0, 255), -1)
        #cv2.circle(video_front_resize_input, (580, 400), 4, (0, 0, 255), -1)
        #cv2.circle(video_front_resize_input, (280, 530), 4, (0, 0, 255), -1)
        #cv2.circle(video_front_resize_input, (740, 530), 4, (0, 0, 255), -1)
        # distance between object's center and my car (middle bottom -> 540,720)

        distance = float(format(math.sqrt(math.pow((540 - cx), 2) + math.pow((720 - cy), 2)), ".3f"))
        rs = float(format(sd * fps, ".3f"))
        roa = float(format((rs / distance), ".3f"))
        cv2.putText(video_rear_resize_input, f'{id}', (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
        cv2.rectangle(video_rear_resize_input, (x3, y3), (x4, y4), (0, 0, 255), 1)

        """
        distance = float(format(math.sqrt(math.pow((540 - cx), 2) + math.pow((720 - cy), 2)), ".3f"))

        # Danger Judgment Part
        # real world speed, not necessary, main thema is how dangerous -> box size chnaging / delta time(1/frame) -> rate of changing
        # relative_speed = space_difference / delta_t(if 30fps->0.03333... = 0.033)
        # if pos. -> closer, if neg. it further
        # relative speed (relative speed = space difference / delta time(=1/fps) = space difference * fps)
        rs = float(format(sd * fps, ".3f"))
        # rate_of_approach (rate of approach = rate of changing / distance)
        roa = float(format(rs / distance, ".3f"))
        # rate of approach( =1/approaching time)
        # if time is shorter than human react(1s) -> dangerous
        # research
        # 1. Driver Reaction Time to Tactile and Auditory Rear-End Collision Warnings While Talking on a Cell Phone
        # 2. Driver’s reaction time in a simulated, complex road incident
        # 3. Drivers ’reaction time research in the conditions in the real traffic
        # 4. How Does the Driver’s Perception Reaction Time Affect the Performances of Crash Surrogate Measures?
        if roa == 0:
            at=2
        else:
            at = float(format((1 / roa), ".3f"))
        # if closer
        if (rs < 0):
            if (at < 1):
                if cx > 540:
                    cv2.putText(video_rear_resize_input, f'{id}{nr}{rs}{distance}{"closerr"}', (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                    # cv2.putText(video_front_resize_input, str(id), (max(0, cx), max(35, cy)), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.rectangle(video_rear_resize_input, (x3, y3), (x4, y4), (0, 0, 255), 1)
                else:
                    cv2.putText(video_rear_resize_input, f'{id}{nr}{rs}{distance}{"closerr"}', (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.rectangle(video_rear_resize_input, (x3, y3), (x4, y4), (0, 0, 255), 1)
            else:
                if cx > 540:
                    cv2.putText(video_rear_resize_input, f'{id}{nr}{rs}{distance}{"closer"}', (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.rectangle(video_rear_resize_input, (x3, y3), (x4, y4), (0, 255, 255), 1)
                else:
                    cv2.putText(video_rear_resize_input, f'{id}{nr}{rs}{distance}{"closer"}', (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.rectangle(video_rear_resize_input, (x3, y3), (x4, y4), (0, 255, 255), 1)
        # if further
        else:
            if cx > 540:
                cv2.putText(video_rear_resize_input, f'{id}{nr}{rs}{distance}{"further"}', (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                cv2.rectangle(video_rear_resize_input, (x3, y3), (x4, y4), (0, 255, 0), 1)
            else:
                cv2.putText(video_rear_resize_input, f'{id}{nr}{rs}{distance}{"further"}', (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                cv2.rectangle(video_rear_resize_input, (x3, y3), (x4, y4), (0, 255, 0), 1)
        """
    ##########################Rear Camera_End###########################

    ##########################Lane_Finding_Start###########################
    # Wrapping (Bird Eye View)
    wrapped_img, minverse = wrapping(video_front_resize_input)

    cv2.circle(video_front_resize_input, (x_bottom_left_src, y_bottom_left_src), 4, (0, 0, 255), -1)
    cv2.circle(video_front_resize_input, (x_top_left_src, y_top_left_src), 4, (0, 0, 255), -1)
    cv2.circle(video_front_resize_input, (x_top_right_src, y_top_right_src), 4, (0, 0, 255), -1)
    cv2.circle(video_front_resize_input, (x_bottom_right_src, y_bottom_right_src), 4, (0, 0, 255), -1)

    cv2.circle(wrapped_img, (x_bottom_left_dst, y_bottom_left_dst), 4, (0, 255, 255), -1)
    cv2.circle(wrapped_img, (x_top_left_dst, y_top_left_dst), 4, (0, 255, 255), -1)
    cv2.circle(wrapped_img, (x_top_right_dst, y_top_right_dst), 4, (0, 255, 255), -1)
    cv2.circle(wrapped_img, (x_bottom_right_dst, y_bottom_right_dst), 4, (0, 255, 255), -1)

    x = 720
    y = 1080

    #cv2.line(video_front_resize_input, (x_bottom_left_src, y_bottom_left_src), (x_top_left_src, y_top_left_src), (0, 0, 255), 2)
    #cv2.line(video_front_resize_input, (x_top_right_src, y_top_right_src), (x_bottom_right_src, y_bottom_right_src), (0, 0, 255), 2)

    #cv2.line(wrapped_img, (x_bottom_left_dst, y_bottom_left_dst), (x_top_left_dst, y_top_left_dst), (0,255,255), 2)
    #cv2.line(wrapped_img, (x_top_right_dst, y_top_right_dst), (x_bottom_right_dst, y_bottom_right_dst), (0, 255, 255), 2)

    """
    #
    w_f_img = color_filter(wrapped_img)
    roi_result = roi(w_f_img)
    #print(roi_result.shape)
    #cv2.circle(roi_result, (int(0.05 * x), int(0.95 * y)), 4, (0, 0, 255), -1)
    #cv2.circle(roi_result, (int(0.05 * x), int(0.01 * y)), 4, (0, 0, 255), -1)
    #cv2.circle(roi_result, (int(0.45 * x), int(0.01 * y)), 4, (0, 0, 255), -1)
    #cv2.circle(roi_result, (int(0.45 * x), int(0.94 * y)), 4, (0, 0, 255), -1)
    #cv2.circle(roi_result, (int(0.60 * x), int(0.94 * y)), 4, (0, 0, 255), -1)
    #cv2.circle(roi_result, (int(0.60 * x), int(0.01 * y)), 4, (0, 0, 255), -1)
    #cv2.circle(roi_result, (int(1.10 * x), int(0.01 * y)), 4, (0, 0, 255), -1)
    #cv2.circle(roi_result, (int(0.95 * x), int(0.95 * y)), 4, (0, 0, 255), -1)
    #cv2.circle(roi_result, (int(0.11 * x), int(0.95 * y)), 4, (0, 0, 255), -1)
    ##ROI from color filtered image

    #_gray = cv2.cvtColor(roi_result, cv2.COLOR_BGR2GRAY)
    _gray = cv2.cvtColor(w_f_img, cv2.COLOR_BGR2GRAY)
    leftx, lefty, rightx, righty, thresh = slide_window_search(_gray)

    #draw_info, thresh = fit_poly(_gray,leftx, lefty, rightx, righty)
    left_fit, right_fit, left_fitx, right_fitx, ploty, thresh = fit_poly(_gray,leftx, lefty, rightx, righty, thresh)

    ## 원본 이미지에 라인 넣기
    meanPts, video_front_resize_input = draw_lane_lines(video_front_resize_input, thresh, minverse, left_fitx, right_fitx, ploty)
    """
    img_out, angle, colorwarp, draw_poly_img = lane_finding_pipeline(video_front_resize_input, init)

    if angle > 1.5 or angle < -1.5:
        init = True
    else:
        init = False

    # Original Image Transfer and processing and add to detected Vehicles Image, not directly using detected Vehicles Images
    cv2.namedWindow('camera_front_input', cv2.WINDOW_NORMAL)
    cv2.imshow('camera_front_input', img_out)
    cv2.namedWindow('colorwarp', cv2.WINDOW_NORMAL)
    cv2.imshow('colorwarp', colorwarp)
    cv2.namedWindow('draw_poly', cv2.WINDOW_NORMAL)
    # to draw the detected lane lines on a "bird's-eye view" image.
    cv2.imshow('draw_poly', draw_poly_img)

    ##########################Lane_Finding_End###########################




    # Output Video
    ## Original Video(just resize)
    #cv2.imshow("camera_front_input", video_front_resize_input)
    #cv2.imshow("wrapped_img", wrapped_img)
    #cv2.imshow("w_f_img", w_f_img)
    #cv2.imshow('_gray', _gray)
    #cv2.imshow('threshold', thresh)
    #print("leftbase", leftbase)
    #print("rightbase", rightbase)
    #cv2.imshow("camera_rear_input", video_rear_resize_input)
    # cv2.imshow("camera_front_resize_input_test", camera_front_resize_input_test)
    # cv2.imshow("camera_rear_resize_input_test", camera_rear_resize_input_test)

    ## Video Export
    output_front.write(video_front_resize_input)
    output_rear.write(video_rear_resize_input)
    # output_front.write(camera_front_resize_input)  # Test Video Front
    # output_rear.write(camera_rear_resize_input)    # Test Video Rear

    if cv2.waitKey(1) == ord('q'):
        break

video_front_resize_input.release()
video_rear_resize_input.release()
output_front.release()
output_rear.release()

cv2.destroyAllWindows()

# Function to calculate the steering angle based on lane information
def calculate_steering_angle(lane_info):
    left_x1, left_y1, left_x2, left_y2 = lane_info['left_lane']
    right_x1, right_y1, right_x2, right_y2 = lane_info['right_lane']
    
    # Calculate slopes of the left and right lanes
    slope_left = (left_y2 - left_y1) / (left_x2 - left_x1)
    slope_right = (right_y2 - right_y1) / (right_x2 - right_x1)
    
    # Calculate the average slope
    slope_avg = (slope_left + slope_right) / 2
    
    # Convert the slope to angle in degrees
    angle = math.atan(slope_avg) * (180 / math.pi)
    return angle

# Function to get lane information based on the coordinates of the left and right lanes
def get_lane_info(left_lane, right_lane):
    return {'left_lane': left_lane, 'right_lane': right_lane}

# Function to rotate the steering wheel image based on the calculated steering angle
def rotate_steering_wheel(image, angle):
    # Get image dimensions
    (h, w) = image.shape[:2]
    
    # Calculate the center of the image
    center = (w // 2, h // 2)
    
    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    
    return rotated

# Example usage
# lane_info = get_lane_info((left_x1, left_y1, left_x2, left_y2), (right_x1, right_y1, right_x2, right_y2))
# steering_angle = calculate_steering_angle(lane_info)
# print(f"Steering Angle: {steering_angle} degrees")

# Uncomment below lines after loading your steering wheel image
# rotated_image = rotate_steering_wheel(your_steering_wheel_image, steering_angle)
# cv2.imshow('Rotated Steering Wheel', rotated_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
