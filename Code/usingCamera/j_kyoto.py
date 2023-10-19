from ultralytics import YOLO
import cv2
import pandas
import numpy as np
import os
import math
import glob
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tracking_function import *


'''
#ToDo: 1) Traffic Signal 2) M.A.P,(Canny Mask? or white and yellow color) 3) Camera Calibration of Lane Finding
#ToDo: 4) Advanced Lane Detection 5)Futuristic Rectangles
#ToDo: 6) Unity Simulation 7) ROS Simulation
#ToDo: 10) Data Transfer with Unity 11) Data Transfer with ROS
#ToDo: 12) Unity Simulation with VR
#ToDo: 13) ROS Simulation with VR
#ToDo: 14) Jetson Environment Setting and Testing
#ToDo: 15) Making a Film
'''


################################################################################################
# ------------------------------------------Settings------------------------------------------ #
################################################################################################
# Check and change Working Directory
# print("Current Working Directory:", os.getcwd())
os.chdir("d:\\ProjectAsurada\\ProjectAsurada\\Code\\usingCamera")
# --------------------------------------Object Detection-------------------------------------- #
# A model in machine learning refers to the mathematical and computational framework that is used to make predictions or decisions based on input data. 
# It consists of an architecture (the structure and type of the neural network) and weights (the parameters learned during training).
# The purpose of model is used to predict outputs (e.g., object classes and bounding box coordinates in object detection) given new, unseen inputs.
# The model is a convolutional neural network (CNN) that has been trained to detect objects in images. 
# YOLO is known for its ability to detect objects in real-time due to its efficient architecture and computation.
# YOLO (https://github.com/alanzhichen/yolo8-ultralytics)
model = YOLO('yolov8n.pt')      # Nano is the fastest and smallest
# model = YOLO('yolov8s.pt')    #               ...
# model = YOLO('yolov8m.pt')    #               ...
# model = YOLO('yolov8l.pt')    #               ...
# model = YOLO('yolov8x.pt')    # the most accurate yet the slowest among them
# The coco.txt file is used to map the numerical predictions of the model to human-readable class names, which can be displayed in the output, e.g., in bounding box labels.
# open_coco = open("D:\ProjectAsurada\ProjectAsurada\Code\usingCamera\coco.txt", "r")
open_coco = open("coco.txt", "r")
read_coco = open_coco.read()
class_list = read_coco.split("\n")

################################################################################################
# ------------------------------------------Modules------------------------------------------- #
# ============================================================================================ #
# ---------------------------------------Test Functions--------------------------------------- #
# ============================================================================================ #

# Mouse Cursor Coordinate
# :show the current mouse cursor coordinate in the window.
#  normally it should be deactive, but in the test or must be checked the coordinate of window
#  , then should be active to below cv2.nameWindow and cv2.setMouseCallback

"""
def cursor_Coordinate(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        cursor_Coordinate = [x, y]
        print(cursor_Coordinate)

cv2.namedWindow('camera_front_input')      # in ' ' should be filled by name of display window
cv2.setMouseCallback('camera_front_input', cursor_Coordinate)
"""

# ============================================================================================ #
# ---------------------------------------Lane Detection--------------------------------------- #
# ============================================================================================ #
# https://docs.opencv.org/3.4.3/dc/dbb/tutorial_py_calibration.html
# https://foss4g.tistory.com/1665

# Step 1: Distortion and Camera Calibration
def camera_calibration():
    nx = 11         # 9 , 11, number of chessboard's horizontal pattern -1
    ny = 8          # 6 , 8, number of chessboard's vertical pattern -1

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objpoints = []  # Object points are real world points, here a 3D coordinates matrix is generated
    imgpoints = []  # image points are xy plane points, here a 2D coordinates matrix is generated (z = 0, chessboard)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,9,0)
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)

    # Make a list of calibration images
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

    # cv.calibrateCamera() which returns the camera matrix, distortion coefficients, rotation and translation vectors etc.
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

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


# STEP 2_A: Perspective transform from driver's view to bird's eye view _ for main Lane
# input_img_width = 1280
# input_img_heigt = 720

# Source point of front camera (for main Lane)
# (1)                                  # top_left(1)                         # top_right(2)
x_top_left_src_f        = 480               #(x,y)###########################(x,y)#
y_top_left_src_f        = 390                #                                   #
# (2)                                         #                                 #
x_top_right_src_f       = 565                  #                               #
y_top_right_src_f       = 390                   #                             #
# (3)                                            #                           #
x_bottom_left_src_f     = 110                     #(x,y)###############(x,y)#
y_bottom_left_src_f     = 690             # bottom_left(3)          # bottom_right(4)
# (4)
x_bottom_right_src_f    = 885
y_bottom_right_src_f    = 690

# Destination point of front camera (for main Lane)
# (1)                                 # top_left(1)                         # top_right(2)
x_top_left_dst_f        = 55    #10        #(x,y)###########################(x,y)#
y_top_left_dst_f        = 0                #                                     #
# (2)                                      #                                     #
x_top_right_dst_f       = 1035   #1070     #                                     #
y_top_right_dst_f       = 0                #                                     #
# (3)                                      #                                     #
x_bottom_left_dst_f     = 150   #480       #(x,y)###########################(x,y)#
y_bottom_left_dst_f     = 720      # bottom_left(3)                   # bottom_right(4)
# (4)
x_bottom_right_dst_f    = 880    #600
y_bottom_right_dst_f    = 720


def wrapping_f(img):
    if img is None:
        print('Image is None, skippung this iteration')
        return None, None

    h = img.shape[0]              # height of input image
    w = img.shape[1]              # width of input image

    # Set the wrapping area

    # Method A: Set the image size, proportional to input image size
    # source = np.float32([[w // 2 - 30, h * 0.53], [w // 2 + 60, h * 0.53], [w * 0.3, h], [w, h]])
    # destination = np.float32([[0, 0], [w-350, 0], [400, h], [w-150, h]])

    # Method B: Set image size to set value
    source_f = np.float32(
        [
            (x_bottom_left_src_f, y_bottom_left_src_f),     # bottom-left corner
            (x_top_left_src_f, y_top_left_src_f),           # top-left corner
            (x_top_right_src_f, y_top_right_src_f),         # top-right corner
            (x_bottom_right_src_f, y_bottom_right_src_f)    # bottom-right corner
        ])

    destination_f = np.float32(
        [
            (x_bottom_left_dst_f, y_bottom_left_dst_f),     # bottom-left corner
            (x_top_left_dst_f, y_top_left_dst_f),           # top-left corner
            (x_top_right_dst_f, y_top_right_dst_f),         # top-right corner
            (x_bottom_right_dst_f, y_bottom_right_dst_f)    # bottom-right corner
        ])

    # getPerspectiveTransformation? the properties that it hold the property of linear, but not the property of parallelity
    # for example, train lanes are parallel but through the perspective transformation, it looks like they are meeting at the end of point
    # we need 4 point of input and moving point of output
    transform_matrix = cv2.getPerspectiveTransform(source_f, destination_f)
    # minv: In the final process, we need to use the inverse function again to change from bird's eye view to driver view. (matrix inverse)
    minv = cv2.getPerspectiveTransform(destination_f, source_f)
    # _img: wr
    _img = cv2.warpPerspective(img, transform_matrix, (img.shape[1], img.shape[0]))

    return _img, minv


# STEP 2.B: Perspective transform from driver's view to bird's eye view _ for left/right side Lane
# Source point of front camera (for left side Lane)
# (1)
x_top_left_src_left     = 420
y_top_left_src_left     = 395
# (2)
x_top_right_src_left    = 500
y_top_right_src_left    = 395
# (3)
x_bottom_left_src_left  = 30
y_bottom_left_src_left  = 510
# (4)
x_bottom_right_src_left = 420
y_bottom_right_src_left = 510

# Destination point of front camera (for left side Lane)
# (1)
x_top_left_dst_left     = 55
y_top_left_dst_left     = 0
# (2)
x_top_right_dst_left    = 1035
y_top_right_dst_left    = 0
# (3)
x_bottom_left_dst_left  = 150
y_bottom_left_dst_left  = 720
# (4)
x_bottom_right_dst_left = 880
y_bottom_right_dst_left = 720

# Source point of front camera (for right side Lane)
# (1)
x_top_left_src_right     = 550
y_top_left_src_right     = 410
# (2)
x_top_right_src_right    = 700
y_top_right_src_right    = 410
# (3)
x_bottom_left_src_right  = 640
y_bottom_left_src_right  = 510
# (4)
x_bottom_right_src_right = 1060
y_bottom_right_src_right = 510

# Destination point of front camera (for right side Lane)
# (1)
x_top_left_dst_right     = 55
y_top_left_dst_right     = 0
# (2)
x_top_right_dst_right    = 1035
y_top_right_dst_right    = 0
# (3)
x_bottom_left_dst_right  = 150
y_bottom_left_dst_right  = 720
# (4)
x_bottom_right_dst_right = 880
y_bottom_right_dst_right = 720


# Warping Left/Right Lane for lane checking(if you want to check "left lane", scanning state should be "left")
def wrapping_line_lr(img, scanning_state):
    if img is None:
        print('Image is None, skippung this iteration')
        return None, None

    if scanning_state == 'left':
        source_lr = np.float32(
            [
                (x_bottom_left_src_left, y_bottom_left_src_left),   # bottom-left corner
                (x_top_left_src_left, y_top_left_src_left),         # top-left corner
                (x_top_right_src_left, y_top_right_src_left),       # top-right corner
                (x_bottom_right_src_left, y_bottom_right_src_left)  # bottom-right corner
            ])

        destination_lr = np.float32(
            [
                (x_bottom_left_dst_left, y_bottom_left_dst_left),   # bottom-left corner
                (x_top_left_dst_left, y_top_left_dst_left),         # top-left corner
                (x_top_right_dst_left, y_top_right_dst_left),       # top-right corner
                (x_bottom_right_dst_left, y_bottom_right_dst_left)  # bottom-right corner
            ])
    elif scanning_state == 'right':
        source_lr = np.float32(
            [
                (x_bottom_left_src_right, y_bottom_left_src_right),   # bottom-left corner
                (x_top_left_src_right, y_top_left_src_right),         # top-left corner
                (x_top_right_src_right, y_top_right_src_right),       # top-right corner
                (x_bottom_right_src_right, y_bottom_right_src_right)  # bottom-right corner
            ])
        destination_lr = np.float32(
            [
                (x_bottom_left_dst_right, y_bottom_left_dst_right),   # bottom-left corner
                (x_top_left_dst_right, y_top_left_dst_right),         # top-left corner
                (x_top_right_dst_right, y_top_right_dst_right),       # top-right corner
                (x_bottom_right_dst_right, y_bottom_right_dst_right)  # bottom-right corner
            ])

    transform_matrix = cv2.getPerspectiveTransform(source_lr, destination_lr)
    minv = cv2.getPerspectiveTransform(destination_lr, source_lr)
    _img = cv2.warpPerspective(img, transform_matrix, (img.shape[1], img.shape[0]))

    return _img, minv

# STEP 3
# Method A: Filtering the warped image for white and yellow lane to check (using HLS)
# https://stackoverflow.com/questions/55822409/hsl-range-for-yellow-lane-lines
# HLS(Hue, Luminanse, Saturation) :
# lower = ([minimum_blue, m_green, m_red])
# upper = ([Maximum_blue, M_green, M_red])

def color_filter(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
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
    # yellow_lower = np.array([np.round(40 / 2), np.round(0.75 * 255), np.round(0.00 * 255)])
    # yellow_upper = np.array([np.round(60 / 2), np.round(1.00 * 255), np.round(0.30 * 255)])
    yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)

    # Do filtering the each yellow lane and white lane,
    # Bitwise_or makes (yellow line and white line) combining -> mask
    # bitwise_and maeks (original image and mask) -> then left just masked part -> masked

    mask = cv2.bitwise_or(yellow_mask, white_mask)
    # masked = cv2.bitwise_and(img, img, mask=mask)
    return mask

# Method B: Using Sobel, detecting the white and yellow lane
def binary_thresholded(img):
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

# STEP 4
# Method A: Region of Interest, from the bird eye's view, decreasing the area, focus more on the lane.
# Method B: Another Idea is first ROI, in order to decrease the input area, then we don't need anymore, sky etc...
#           but in this case we need to set a wrapping area, so it is not so effective, but in case SortDeep.. could be...

def roi(img):
    x = int(img.shape[1])       # height = img.shape[0]
    y = int(img.shape[0])       # width = img.shape[1]
                                # channels = img.shape[2]
    # Height represents the number of pixel rows in the image or the number of pixels in each column of the image array.
    # Width represents the number of pixel columns in the image or the number of pixels in each row of the image array.
    # Number of Channels represents the number of components used to represent each pixel.
    # In the above example, Number of Channels = 4 represent Alpha, Red, Green and Blue channels.
    # *** here traffic sign on the street is deleted and ignored, if you don't wanna that, modify the ROI part.

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
        [int(0.05 * x), int(0.95 * y)],  # 1
        [int(0.05 * x), int(0.01 * y)],  # 2
        [int(0.45 * x), int(0.01 * y)],  # 3
        [int(0.45 * x), int(0.94 * y)],  # 4
        [int(0.60 * x), int(0.94 * y)],  # 5
        [int(0.60 * x), int(0.01 * y)],  # 6
        [int(0.95 * x), int(0.01 * y)],  # 7
        [int(0.95 * x), int(0.95 * y)],  # 8
        [int(0.11 * x), int(0.95 * y)]  # 9
    ])

    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, np.int32([_shape]), ignore_mask_color)
    # cv2.fillPoly(mask, np.float32([_shape]), ignore_mask_color)
    masked_img = cv2.bitwise_and(img, mask)

    return masked_img

# STEP 5
# finding a lane using sliding window
def slide_window_search(binary_wrapped):
    # creating a 3-channel image(out_img) from the single-channel(binary_wrapped)
    # multiply by 255, convert from the binary image to 8-bit image
    out_img = np.dstack((binary_wrapped, binary_wrapped, binary_wrapped)) * 255
    window_img = np.zeros_like(out_img)  # need check for usage.

    # using threshold filtering out noise
    ret, thresh = cv2.threshold(binary_wrapped, 140, 195, cv2.THRESH_BINARY)

    # ---Histogram---
    # it is not histogram of opencv
    # bitwise image has one channel and value between 0 ~ 255.
    # if it is lane, they have a value near by 255, and if it isn't, then 0.
    # it means for one column, when we add all row values, if there are lane, they has relative big value, if not, small value

    # Take a histogram of the bottom half of the image
    histogram = np.sum(thresh[thresh.shape[0] // 2:, :], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int64(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # set the nwindows
    nwindows = 9
    
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Set height of windows - based on nwindows above and image shape
    window_height = np.int64(binary_wrapped.shape[0] / nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_wrapped.nonzero()  # save the index, where a lane exists.
    nonzero_y = np.array(nonzero[0])  # index value of y, where line exists.
    nonzero_x = np.array(nonzero[1])  # index value of x, where line exists.

    # Current positions to be updated later for each window in nwindows
    # left_current = a biggest index of image's left side (coordinate information)
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
        win_y_low = binary_wrapped.shape[0] - (w + 1) * window_height  # window top
        win_y_high = binary_wrapped.shape[0] - w * window_height  # window bottom
        win_xleft_low = left_current - margin  # left window's left top
        win_xleft_high = left_current + margin  # left window's right bottom
        win_xright_low = right_current - margin  # right window's left top
        win_xright_high = right_current + margin  # right window's right bottom

        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), color, thickness)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), color, thickness)
        # Save only the part inside the window
        # Identify the nonzero pixels in x and y within the window #
        # good_left = save the part just in window
        good_left = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
        good_right = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]
        left_lane.append(good_left)
        right_lane.append(good_right)
        # cv2.imshow("oo", out_img)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left) > minpix:
            left_current = np.int64(np.mean(nonzero_x[good_left]))
        if len(good_right) > minpix:
            right_current = np.int64(np.mean(nonzero_x[good_right]))

    try:
        # For subsequent processing, it's easier to work with a single 1D array of indices rather than a list of arrays. That's why np.concatenate is used.
        left_lane = np.concatenate(left_lane)  # np.concatenate() -> Array makes 1.Dimension array
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


def fit_poly(binary_wrapped, leftx, lefty, rightx, righty):
    # Fit a second order polynomial to the detected left and right lane pixels. 
    # The resulting coefficients describe the curvature of the lanes.
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_wrapped.shape[0] - 1, binary_wrapped.shape[0])
    
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

    # left_fit describes the mathematical representation (coefficients) of the left lane's shape.
    # left_fitx provides Given the polynomial coefficients in left_fit and a set of y values (ploty), provides the corresponding x values of the polynomial.
    # ploty provides the vertical positions in the image where the polynomial (lane) is evaluated.(as an array of y-values)
    return left_fit, right_fit, left_fitx, right_fitx, ploty

# List to keep track of lines' y-positions
dynamic_lines_y = []
# dynamic lane scanning for left and right side
dynamic_lines_y_left = []
dynamic_lines_y_right = []

# Step 6.
# with fillPoly function draw a polygon including left and right lane
# through the pts_mean, we could know the degree of curvature between the lane and lane
# from the function warpping, using the value minv, to the perspectived image, with addWeighted function, finish the work by combining the color of polygon lightly.
def draw_lane_lines(binary_wrapped, left_fitx, right_fitx, ploty):
    global dynamic_lines_y  # Declare dynamic_lines_y as global within the function

    # First: Draw Lane
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_wrapped, binary_wrapped, binary_wrapped)) * 255
    window_img = np.zeros_like(out_img)

    # margin = int(1080 * (100 / 1920))
    margin = 50  # Margin for the lane lines

    # Generate polygons to illustrate the search window area for left and right lanes
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))

    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lanes on the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (100, 100, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (100, 100, 0))

    result = cv2.addWeighted(out_img, 1, window_img, 0.9, 0)  # (0.3)

    # Second: Draw Dynamic Lane
    # Initialize dynamic_lines_y if it's the first frame or update it otherwise
    if len(dynamic_lines_y) == 0:
        dynamic_lines_y = list(range(int(min(ploty)), int(max(ploty)), 50))
    else:
        if scanning_state == ['right', 'left']:
            speed = 20  # Change this to make lines move faster or slower
        else:
            speed = 5  # Change this to make lines move faster or slower
        dynamic_lines_y = [y - speed for y in dynamic_lines_y if y >= min(ploty)]

    # Add new lines if needed
    if len(dynamic_lines_y) < 10:  # 10 is an example; use the number that works best for you.
        dynamic_lines_y.append(max(ploty))

    # Draw dynamic lines
    for y in dynamic_lines_y:
        if np.any(ploty == y):
            cv2.line(result, (int(left_fitx[np.where(ploty == y)][0]), int(y)),
                     (int(right_fitx[np.where(ploty == y)][0]), int(y)),
                     (0, 255, 255), 5)

    mean_x = np.mean((left_fitx, right_fitx), axis=0)
    pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])

    return result


# STEP 7: Detection of Lane Lines Based on Previous Step(Efficency and Stability Method for detection of lane)
# the find_lane_pixels_using_prev_poly method identifies lane pixels in a given binary image based on a previously detected polynomial. 
# Instead of re-computing lane detections from scratch, it leverages the previous detection to simplify and expedite the process.
def find_lane_pixels_using_prev_poly(binary_wrapped):

    # width of the margin around the previous polynomial to search
    margin = 50
    # Grab activated pixels
    nonzero = binary_wrapped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Set the area of search based on activated x-values
    # within the +/- margin of our polynomial function
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


# STEP 8: Calculate Vehicle Position and Curve Radius ###
# Step 8.A
def measure_curvature_meters(binary_wrapped, left_fitx, right_fitx, ploty):
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

# Step 8.B
def measure_position_meters(binary_wrapped, left_fit, right_fit):
    # Define conversion in x from pixels space to meters
    # xm_per_pix = 3.7 / 1920 * (1080 / 1920)  # meters per pixel in x dimension
    xm_per_pix = 3.7 / 1080 * (720 / 1080)  # meters per pixel in x dimension
    # Choose the y value corresponding to the bottom of the image
    y_max = binary_wrapped.shape[0]
    # Calculate left and right line positions at the bottom of the image
    left_x_pos = left_fit[0] * y_max ** 2 + left_fit[1] * y_max + left_fit[2]
    right_x_pos = right_fit[0] * y_max ** 2 + right_fit[1] * y_max + right_fit[2]
    # Calculate the x position of the center of the lane
    center_lanes_x_pos = (left_x_pos + right_x_pos) // 2
    # Calculate the deviation between the center of the lane and the center of the picture
    # The car is assumed to be placed in the center of the picture
    # If the deviation is negative, the car is on the felt hand side of the center of the lane
    veh_pos = ((binary_wrapped.shape[1] // 2) - center_lanes_x_pos) * xm_per_pix
    return veh_pos


# To store the previous N steering angles
previous_angles = []

# Step 9. Caculate and visualization Steering Angle
# Step 9.A.
# This function now keeps track of the last num_prev_angles (default is 5) steering angles and averages them to determine the current steering angle.
# It also divides the calculated angle by a sensitivity factor (default is 4) to make it less sensitive.
# You can adjust num_prev_angles and sensitivity as needed to get the desired performance.
def calculate_steering_angle(left_lane_points, right_lane_points, num_prev_angles=5, sensitivity=20):
    global previous_angles

    # Calculate midpoints of the first and last coordinates of the lanes
    mid_start   = [(left_lane_points[0][0] + right_lane_points[0][0]) // 2, (left_lane_points[0][1] + right_lane_points[0][1]) // 2]
    mid_end     = [(left_lane_points[1][0] + right_lane_points[1][0]) // 2, (left_lane_points[1][1] + right_lane_points[1][1]) // 2]

    # Calculate angle
    angle = math.atan2(mid_end[1] - mid_start[1], mid_end[0] - mid_start[0])
    angle = math.degrees(angle)

    # Normalize the angle by dividing it by a sensitivity factor
    normalized_angle = angle / sensitivity

    # Add to the list of previous angles
    previous_angles.append(normalized_angle)

    # Only keep the last N angles
    if len(previous_angles) > num_prev_angles:
        previous_angles.pop(0)

    # Average the last N angles for smoother transitions
    average_angle = sum(previous_angles) / len(previous_angles)

    return average_angle


# Function to get lane information based on the coordinates of the left and right lanes
def get_lane_info(left_lane, right_lane):
    return {'left_lane': left_lane, 'right_lane': right_lane}


steering_wheel_image = cv2.imread('D:\ProjectAsurada\ProjectAsurada\pic\SteerWheel.png')

# Step 9.B.
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


# STEP 10: Project Lane Delimitations Back on Image Plane and Add Text for Lane Info
def project_lane_info(img, binary_wrapped, ploty, left_fitx, right_fitx, M_inv, left_curverad, right_curverad, veh_pos):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_wrapped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Center Line modified
    # margin = 400 * (1080 / 1920)
    # margin = 400 * 0.1
    margin = 30
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])

    pts_left_c = np.array([np.transpose(np.vstack([left_fitx + margin, ploty]))])
    pts_right_c = np.array([np.flipud(np.transpose(np.vstack([right_fitx - margin, ploty])))])
    pts = np.hstack((pts_left_c, pts_right_c))

    # margin value test...
    # pts_left_i = np.array([np.transpose(np.vstack([left_fitx + margin + 150, ploty]))])
    pts_left_i = np.array([np.transpose(np.vstack([left_fitx + margin + 10, ploty]))])
    # pts_right_i = np.array([np.flipud(np.transpose(np.vstack([right_fitx - margin - 150, ploty])))])
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

    #cv2.putText(out_img, 'Curve Radius [m]: ' + str((left_curverad + right_curverad) / 2)[:7], (5, 80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    #cv2.putText(out_img, 'Center Offset [m]: ' + str(veh_pos)[:7], (5, 110), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

    return out_img, colorwarp_img, newwarp


# STEP 11: Lane Finding Pipeline on Video
# main idea of lane_finding_process is that, define a lane at the wrapped image(birdeye view) -> inverse wrapping -> overlaying on the image
def lane_finding_pipeline(img_lane, img_veh, camera_type, init):
    global left_fit_hist
    global right_fit_hist
    global prev_left_fit
    global prev_right_fit

    if init:
        left_fit_hist = np.array([])
        right_fit_hist = np.array([])
        prev_left_fit = np.array([])
        prev_right_fit = np.array([])

    # i) color convert color->binary
    binary_thresh = color_filter(img_lane)
    
    if camera_type == "front":
        # ii) wrapping(birdeye view)
        binary_wrapped, M_inv = wrapping_f(binary_thresh)
    else:
        print("give the line (front/left/right)")

    if (len(left_fit_hist) == 0):
        # here thresh is added because of previous method, if it is not necessary, it could be deleted
        leftx, lefty, rightx, righty = slide_window_search(binary_wrapped)
        left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(binary_wrapped, leftx, lefty, rightx, righty)
        # Store fit in history
        left_fit_hist = np.array(left_fit)
        new_left_fit = np.array(left_fit)
        left_fit_hist = np.vstack([left_fit_hist, new_left_fit])
        right_fit_hist = np.array(right_fit)
        new_right_fit = np.array(right_fit)
        right_fit_hist = np.vstack([right_fit_hist, new_right_fit])

        left_lane_points = [(left_fitx[0], ploty[0]), (left_fitx[-1], ploty[-1])]
        right_lane_points = [(right_fitx[0], ploty[0]), (right_fitx[-1], ploty[-1])]
        steering_angle = calculate_steering_angle(left_lane_points, right_lane_points, num_prev_angles=5,sensitivity=20)
        # print(f"Steering Angle: {steering_angle} degrees")
        rotated_image = rotate_steering_wheel(steering_wheel_image, steering_angle)

    else:
        prev_left_fit = [np.mean(left_fit_hist[:, 0]), np.mean(left_fit_hist[:, 1]), np.mean(left_fit_hist[:, 2])]
        prev_right_fit = [np.mean(right_fit_hist[:, 0]), np.mean(right_fit_hist[:, 1]), np.mean(right_fit_hist[:, 2])]
        leftx, lefty, rightx, righty = find_lane_pixels_using_prev_poly(binary_wrapped)
        if (len(lefty) == 0 or len(righty) == 0):
            leftx, lefty, rightx, righty = slide_window_search(binary_wrapped)
        left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(binary_wrapped, leftx, lefty, rightx, righty)

        # Add new values to history
        new_left_fit = np.array(left_fit)
        left_fit_hist = np.vstack([left_fit_hist, new_left_fit])
        new_right_fit = np.array(right_fit)
        right_fit_hist = np.vstack([right_fit_hist, new_right_fit])

        left_lane_points = [(left_fitx[0], ploty[0]), (left_fitx[-1], ploty[-1])]
        right_lane_points = [(right_fitx[0], ploty[0]), (right_fitx[-1], ploty[-1])]
        steering_angle = calculate_steering_angle(left_lane_points, right_lane_points, num_prev_angles=5,
                                                  sensitivity=20)
        # print(f"Steering Angle: {steering_angle} degrees")
        rotated_image = rotate_steering_wheel(steering_wheel_image, steering_angle)

        # Remove old values from history
        if (len(left_fit_hist) > 5):  # 10
            left_fit_hist = np.delete(left_fit_hist, 0, 0)
            right_fit_hist = np.delete(right_fit_hist, 0, 0)

    # Drawing the Lane Polynomials:
    draw_poly_img = draw_lane_lines(binary_wrapped, left_fitx, right_fitx, ploty)
    # Unwarping the Image(Lane Polynomials):
    draw_poly_img_unwarped = cv2.warpPerspective(draw_poly_img, M_inv, (video_front_resize_input.shape[1], video_front_resize_input.shape[0]))
    # Overlaying the Lanes onto the Original Frame:
    front_out_combined = cv2.addWeighted(video_front_resize_input, 1, draw_poly_img_unwarped, 0.7, 0)

    left_curverad, right_curverad = measure_curvature_meters(binary_wrapped, left_fitx, right_fitx, ploty)
    veh_pos = measure_position_meters(binary_wrapped, left_fit, right_fit)
    out_img, colorwarp_img, newwarp = project_lane_info(front_out_combined, binary_wrapped, ploty, left_fitx, right_fitx, M_inv, left_curverad, right_curverad, veh_pos)

    margin = 50  # You can adjust this value
    # Define the polygon for the left lane area
    left_lane_poly = [(left_fitx[i] - margin, ploty[i]) for i in range(len(ploty))] + \
                     [(left_fitx[i] + margin, ploty[i]) for i in reversed(range(len(ploty)))]

    # Define the polygon for the right lane area
    right_lane_poly = [(right_fitx[i] - margin, ploty[i]) for i in range(len(ploty))] + \
                      [(right_fitx[i] + margin, ploty[i]) for i in reversed(range(len(ploty)))]

    return out_img, veh_pos, colorwarp_img, draw_poly_img, rotated_image


# def lane_finding_pipeline(img, init, mts, dist):
def lane_finding_pipeline_suggest(img_lane, img_veh, camera_type, init):
    global left_fit_hist
    global right_fit_hist
    global prev_left_fit
    global prev_right_fit

    if init:
        left_fit_hist = np.array([])
        right_fit_hist = np.array([])
        prev_left_fit = np.array([])
        prev_right_fit = np.array([])

    # i) color conver color->binary
    binary_thresh = color_filter(img_lane)
    
    if camera_type == "front":
        # ii) wrapping(birdeye view)
        binary_wrapped, M_inv = wrapping_f(binary_thresh)
    else:
        print("give the line (front/left/right)")
        
    if (len(left_fit_hist) == 0):
        # here thresh is added because of previous method, if it is not necessary, it could be deleted
        leftx, lefty, rightx, righty = slide_window_search(binary_wrapped)
        left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(binary_wrapped, leftx, lefty, rightx, righty)
        # Store fit in history
        left_fit_hist = np.array(left_fit)
        new_left_fit = np.array(left_fit)
        left_fit_hist = np.vstack([left_fit_hist, new_left_fit])
        right_fit_hist = np.array(right_fit)
        new_right_fit = np.array(right_fit)
        right_fit_hist = np.vstack([right_fit_hist, new_right_fit])

        left_lane_points = [(left_fitx[0], ploty[0]), (left_fitx[-1], ploty[-1])]
        right_lane_points = [(right_fitx[0], ploty[0]), (right_fitx[-1], ploty[-1])]
        steering_angle = calculate_steering_angle(left_lane_points, right_lane_points, num_prev_angles=5, sensitivity=20)
        # print(f"Steering Angle: {steering_angle} degrees")
        rotated_image = rotate_steering_wheel(steering_wheel_image, steering_angle)

    else:
        prev_left_fit = [np.mean(left_fit_hist[:, 0]), np.mean(left_fit_hist[:, 1]), np.mean(left_fit_hist[:, 2])]
        prev_right_fit = [np.mean(right_fit_hist[:, 0]), np.mean(right_fit_hist[:, 1]), np.mean(right_fit_hist[:, 2])]
        leftx, lefty, rightx, righty = find_lane_pixels_using_prev_poly(binary_wrapped)
        if (len(lefty) == 0 or len(righty) == 0):
            leftx, lefty, rightx, righty = slide_window_search(binary_wrapped)
        left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(binary_wrapped, leftx, lefty, rightx, righty)

        # Add new values to history
        new_left_fit = np.array(left_fit)
        left_fit_hist = np.vstack([left_fit_hist, new_left_fit])
        new_right_fit = np.array(right_fit)
        right_fit_hist = np.vstack([right_fit_hist, new_right_fit])

        left_lane_points = [(left_fitx[0], ploty[0]), (left_fitx[-1], ploty[-1])]
        right_lane_points = [(right_fitx[0], ploty[0]), (right_fitx[-1], ploty[-1])]
        steering_angle = calculate_steering_angle(left_lane_points, right_lane_points, num_prev_angles=5, sensitivity=20)
        # print(f"Steering Angle: {steering_angle} degrees")
        rotated_image = rotate_steering_wheel(steering_wheel_image, steering_angle)

        # Remove old values from history
        if (len(left_fit_hist) > 5):  # 10
            left_fit_hist = np.delete(left_fit_hist, 0, 0)
            right_fit_hist = np.delete(right_fit_hist, 0, 0)

    margin = 50  # You can adjust this value
    # Define the polygon for the left lane area
    left_lane_poly = [(left_fitx[i] - margin, ploty[i]) for i in range(len(ploty))] + \
                     [(left_fitx[i] + margin, ploty[i]) for i in reversed(range(len(ploty)))]

    # Define the polygon for the right lane area
    right_lane_poly = [(right_fitx[i] - margin, ploty[i]) for i in range(len(ploty))] + \
                      [(right_fitx[i] + margin, ploty[i]) for i in reversed(range(len(ploty)))]

    return left_lane_poly, right_lane_poly

# important argument scanning_state, according to scanning_state the searching lane will be decided.
scanning_state = 'none'

def lane_finding_pipeline_lr(img_lane, img_veh, scanning_state, init):
    global left_fit_hist_lr
    global right_fit_hist_lr
    global prev_left_fit_lr
    global prev_right_fit_lr

    if init:
        left_fit_hist_lr = np.array([])
        right_fit_hist_lr = np.array([])
        prev_left_fit_lr = np.array([])
        prev_right_fit_lr = np.array([])

    # i) color conver color->binary
    binary_thresh = color_filter(img_lane)

    if scanning_state == 'left':
        binary_wrapped, M_inv = wrapping_line_lr(binary_thresh, scanning_state)
    elif scanning_state == 'right':
        binary_wrapped, M_inv = wrapping_line_lr(binary_thresh, scanning_state)
    else:
        print("not selected...")

    if (len(left_fit_hist_lr) == 0):
        # here thresh is added because of previous method, if it is not necessary, it could be deleted
        leftx, lefty, rightx, righty = slide_window_search(binary_wrapped)
        left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(binary_wrapped, leftx, lefty, rightx, righty)
        # Store fit in history
        left_fit_hist_lr = np.array(left_fit)
        new_left_fit_lr = np.array(left_fit)
        left_fit_hist_lr = np.vstack([left_fit_hist_lr, new_left_fit_lr])
        right_fit_hist_lr = np.array(right_fit)
        new_right_fit_lr = np.array(right_fit)
        right_fit_hist_lr = np.vstack([right_fit_hist_lr, new_right_fit_lr])

        left_lane_points = [(left_fitx[0], ploty[0]), (left_fitx[-1], ploty[-1])]
        right_lane_points = [(right_fitx[0], ploty[0]), (right_fitx[-1], ploty[-1])]
        steering_angle = calculate_steering_angle(left_lane_points, right_lane_points, num_prev_angles=5,sensitivity=20)
        # print(f"Steering Angle: {steering_angle} degrees")
        rotated_image = rotate_steering_wheel(steering_wheel_image, steering_angle)

    else:
        prev_left_fit_lr = [np.mean(left_fit_hist_lr[:, 0]), np.mean(left_fit_hist_lr[:, 1]), np.mean(left_fit_hist_lr[:, 2])]
        prev_right_fit_lr = [np.mean(right_fit_hist_lr[:, 0]), np.mean(right_fit_hist_lr[:, 1]), np.mean(right_fit_hist_lr[:, 2])]
        leftx, lefty, rightx, righty = find_lane_pixels_using_prev_poly(binary_wrapped)
        if (len(lefty) == 0 or len(righty) == 0):
            leftx, lefty, rightx, righty = slide_window_search(binary_wrapped)
        left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(binary_wrapped, leftx, lefty, rightx, righty)

        # Add new values to history
        new_left_fit_lr = np.array(left_fit)
        left_fit_hist_lr = np.vstack([left_fit_hist_lr, new_left_fit_lr])
        new_right_fit_lr = np.array(right_fit)
        right_fit_hist_lr = np.vstack([right_fit_hist_lr, new_right_fit_lr])

        left_lane_points = [(left_fitx[0], ploty[0]), (left_fitx[-1], ploty[-1])]
        right_lane_points = [(right_fitx[0], ploty[0]), (right_fitx[-1], ploty[-1])]
        steering_angle = calculate_steering_angle(left_lane_points, right_lane_points, num_prev_angles=5,
                                                  sensitivity=20)
        # print(f"Steering Angle: {steering_angle} degrees")
        rotated_image = rotate_steering_wheel(steering_wheel_image, steering_angle)

        # Remove old values from history
        if (len(left_fit_hist_lr) > 5):  # 10
            left_fit_hist = np.delete(left_fit_hist_lr, 0, 0)
            right_fit_hist = np.delete(right_fit_hist_lr, 0, 0)

    draw_poly_img_lr = draw_lane_lines(binary_wrapped, left_fitx, right_fitx, ploty)
    draw_poly_img_unwarped = cv2.warpPerspective(draw_poly_img_lr, M_inv, (video_front_resize_input.shape[1], video_front_resize_input.shape[0]))
    front_out_combined_lr = cv2.addWeighted(img_veh, 1, draw_poly_img_unwarped, 0.7, 0)        # dynamic scanning

    left_curverad, right_curverad = measure_curvature_meters(binary_wrapped, left_fitx, right_fitx, ploty)
    veh_pos = measure_position_meters(binary_wrapped, left_fit, right_fit)
    out_img, colorwarp_img, newwarp = project_lane_info(img_veh, binary_wrapped, ploty, left_fitx, right_fitx, M_inv, left_curverad, right_curverad, veh_pos)

    return front_out_combined_lr, veh_pos, colorwarp_img, draw_poly_img_lr, rotated_image

def point_inside_polygon(x, y, poly):
    """
    Check if a point (x, y) is inside a polygon defined by a list of vertex coordinates [(x1, y1), (x2, y2), ...]
    :param x: x-coordinate of the point
    :param y: y-coordinate of the point
    :param poly: list of coordinates [(x1, y1), (x2, y2), ...]
    :return: True if the point is inside the polygon, False otherwise
    """
    n = len(poly)
    inside = False

    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

################################################################################################


################################################################################################
# --------------------------------------- --Script-------------------------------------------- #
# ============================================================================================ #
# -----------------------------------------Settings------------------------------------------- #
# ============================================================================================ #

# Initialize count frame
count_f = 0  # intialize for front camera video frame count
count_r = 0  # intialize for rear camera video frame count

# Call the class for tracking of front and rear video
tracker_f = Tracker()  # Tracking class call for front camera
tracker_r = Tracker()  # Tracking class call for rear camera


# Video Settings
fps = 24.0                                  # Viedo Frame
frame_size = (1080, 720)                    # Video Size
fourcc = cv2.VideoWriter_fourcc(*'mp4v')    # Define the codec using VideoWriter_fourcc
Font = cv2.FONT_HERSHEY_COMPLEX
FontSize = 0.3

start_time = time.time()

## Video Input Part
# Webcam Mode
####################################################
# front camera capture
# video_front_raw_input = cv2.VideoCapture(0)
# Video Frame Size(Front)
# frame_size_raw_front = (int(video_front_raw_input.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_front_raw_input.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# fps_video_front_raw_input = video_front_raw_input.get(cv2.CAP_PROP_FPS)    # Input video FPS Check

# rear camera capture
# video_rear_raw_input = cv2.VideoCapture(1)
# Video Frame Size(Rear)
# frame_size_rear = (int(video_rear_raw_input.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_rear_raw_input.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# fps_video_rear_raw_input = video_rear_raw_input.get(cv2.CAP_PROP_FPS)
####################################################

# Video Mode
####################################################
# front camera capture
video_front_raw_input = cv2.VideoCapture("D:\ProjectAsurada\ProjectAsurada\VideoSample\TestFront.mp4")
# Video Frame Size(Front)
frame_size_raw_front = (int(video_front_raw_input.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_front_raw_input.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# fps_video_front_raw_input = video_front_raw_input.get(cv2.CAP_PROP_FPS)    # Input video FPS Check

# rear camera capture
video_rear_raw_input = cv2.VideoCapture("D:\ProjectAsurada\ProjectAsurada\VideoSample\TestRear.mp4")
# Video Frame Size(Rear)
frame_size_rear = (int(video_rear_raw_input.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_rear_raw_input.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# fps_video_rear_raw_input = video_rear_raw_input.get(cv2.CAP_PROP_FPS)
####################################################

## Video Output Part
# Webcam Mode
output_front = cv2.VideoWriter('video_front_output.mp4', fourcc, fps, frame_size)
output_rear = cv2.VideoWriter('video_rear_output.mp4', fourcc, fps, frame_size)

### Video Mode
output_front = cv2.VideoWriter('video_front_output.mp4', fourcc, fps, frame_size)  # video export
output_rear = cv2.VideoWriter('video_rear_output.mp4', fourcc, fps, frame_size)  # Video 1/3

init_f = True
init_lr = True

any_vehicle_in_left_lane = False
any_vehicle_in_right_lane = False

# mtx, dist = camera_calibration()

while True:
    # Video Input Read (Front)
    ret_front, frame_front = video_front_raw_input.read()  # ret = return, if ret false, loop will be closing
    # Video Input Read (Rear)
    ret_rear, frame_rear = video_rear_raw_input.read()
    
    # Video Frame Resize (Standardization of various video inputs)
    video_front_resize_input = cv2.resize(frame_front, frame_size)
    video_front_resize_input_l = cv2.resize(frame_front, frame_size)
    video_rear_resize_input = cv2.resize(frame_rear, frame_size)
    video_rear_resize_input_l = cv2.resize(frame_rear, frame_size)

    # One Frame Count Up
    count_f += 1
    count_r += 1
    ####################################################################
    # performance optimization using data reduction
    # if count_f % 3 != 0:
    #    continue
    ####################################################################

    # cv2.waitKey's purposes: it allows for real-time processing and provides a mechanism for user interaction during the display of images or videos. 
    #                       If this line is removed from a loop displaying video content, the window might become unresponsive, or the video might not display correctly.
    key = cv2.waitKey(1)            # wait for 1 millisecond(for real-time processing) for any keyboard event, if no key, return -1
    if key != -1:
        if key == ord('a'):         # Left arrow key
            scanning_state = 'left'
        elif key == ord('d'):       # Right arrow key
            scanning_state = 'right'
        #print(f"scanning_state: {scanning_state}")

    # ============================================================================================ #
    # -------------------------------------Vehicles Tracking-------------------------------------- #
    # ============================================================================================ #
    # ----------------------------------------Front Camera---------------------------------------- #
    # ============================================================================================ #
    
    # Object Detect using predict from YOLO and Input Video
    result_f = model.predict(video_front_resize_input)                          # Object Detection using YOLO
    resbb_f = result_f[0].boxes.boxes                                           # Bounding Box
    px_f = pandas.DataFrame(resbb_f.cpu().detach().numpy()).astype("float")     # all detected vehicles's list in px
    # px_f = pandas.DataFrame(resbb_f).astype("float")                          #
    # resbb_f.cpu(): If resbb_f is a tensor (likely from a deep learning framework like PyTorch), this transfers the tensor from GPU memory to CPU memory.
    # detach(): Detaches the tensor from the computation graph. This means that no gradient will be backpropagated along this tensor. 
    #           It's often used when you want to take a tensor out of a computation graph but keep its current value.
    # numpy(): Converts the tensor into a numpy array.
    # Use Cases: It typically used in deep learning workflows where you're working with GPU tensors in PyTorch and want to analyze or store results using pandas.


    list_f = []  # in List, save the each frame information of detected object's x1,x2,y1,y2 value
    
    left_lane_poly, right_lane_poly = lane_finding_pipeline_suggest(video_front_resize_input_l, video_front_resize_input, "front", init_f)

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
    bbox_id_f = tracker_f.update_f(list_f)
    for bbox_f in bbox_id_f:
        x3, y3, x4, y4, sd, dd, id, nr = bbox_f  # appended 4/4
        # x3,y3 :  left upper's coordinate, x4,y4 :  right bottom's coordinate
        # sd : space difference             if sd minus  then further, if sd plus closer
        # dd : distance difference          if dd minus then closer, if dd plus further
        # id : class_id
        # nr : vehicle identification nr(unsupported)
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2

        # Check, whether in the left or right lane, vehicles exist.
        if point_inside_polygon(cx, cy, left_lane_poly):
            any_vehicle_in_left_lane = True
        if point_inside_polygon(cx, cy, right_lane_poly):
            any_vehicle_in_right_lane = True

        # Test for where is check area for lane changing
        """
        cv2.circle(video_front_resize_input, (right_lane_poly[0]), 4, (0, 255, 255), -1)
        cv2.circle(video_front_resize_input, (right_lane_poly[1]), 4, (0, 255, 255), -1)
        cv2.circle(video_front_resize_input, (right_lane_poly[2]), 4, (0, 255, 255), -1)
        cv2.circle(video_front_resize_input, (right_lane_poly[3]), 4, (0, 255, 255), -1)

        cv2.circle(video_front_resize_input, (left_lane_poly[0]), 4, (0, 255, 255), -1)
        cv2.circle(video_front_resize_input, (left_lane_poly[1]), 4, (0, 255, 255), -1)
        cv2.circle(video_front_resize_input, (left_lane_poly[2]), 4, (0, 255, 255), -1)
        cv2.circle(video_front_resize_input, (left_lane_poly[3]), 4, (0, 255, 255), -1)

        cv2.imshow('poly test', video_front_resize_input)
        """

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
        # red (0,0,255), yellow (0,255,255), green (0,128,0), mintcream(250, 255, 245)
        """

        distance_ = float(format(math.sqrt(math.pow((540 - cx), 2) + math.pow((720 - cy), 2)), ".3f"))
        distance = float(format(720 - distance_,".3f"))
        if dd == 0:
            collision_time = 2
        else:
            collision_time = distance / (abs(dd) * fps)      

        if sd < 0 and dd > 0 or distance > 350:
            # low Risk : sd is minus or dd is plus, then the object is moving away or is stationary.
            cv2.putText(video_front_resize_input, f'{"Type:"}{"not supported"}{"(Vehicle_ID)"}', (x3 + 5, y3 + 10),
                        Font, FontSize, (0, 255, 255), 1)
            cv2.putText(video_front_resize_input, f'{distance}{"further"}', (x3 + 5, y3 + 25), Font, FontSize,
                        (0, 255, 255), 1)
            cv2.putText(video_front_resize_input, f'{"low risk"}', (x3 + 5, y3 + 40), Font, FontSize, (0, 255, 255), 1)
            cv2.rectangle(video_front_resize_input, (x3, y3), (x4, y4), (0, 128, 0), 1)
            # cv2.line(video_front_resize_input, (cx, cy), (540, 719), (0, 0, 255), 2)
        elif sd >= 0 and dd > 0:
            # medium Risk : sd is plus and dd is plus, then the object is getting closer but not directly towards your vehicle.
            cv2.putText(video_front_resize_input, f'{"Type:"}{"not supported"}{"(Vehicle_ID)"}', (x3 + 5, y3 + 10),
                        Font, FontSize, (0, 255, 255), 1)
            cv2.putText(video_front_resize_input, f'{distance}{"closer"}', (x3 + 5, y3 + 25), Font, FontSize,
                        (0, 255, 255), 1)
            cv2.putText(video_front_resize_input, f'{"medium risk"}', (x3 + 5, y3 + 40), Font, FontSize, (0, 255, 255),
                        1)
            cv2.rectangle(video_front_resize_input, (x3, y3), (x4, y4), (0, 0, 255), 1)
            # cv2.line(video_front_resize_input, (cx, cy), (540, 719), (0, 0, 255), 2)
        elif sd >= 0 and dd <= 0:
            if collision_time >= 1.5:
                # high 	 Risk : sd is plus and dd is minus, then the object is getting closer and is near the line of motion of your vehicle.
                cv2.putText(video_front_resize_input, f'{"Type:"}{"not supported"}{"(Vehicle_ID)"}', (x3 + 5, y3 + 10),
                            Font, FontSize, (0, 255, 255), 1)
                cv2.putText(video_front_resize_input, f'{distance}{"closer"}', (x3 + 5, y3 + 25), Font, FontSize,
                            (0, 255, 255), 1)
                cv2.putText(video_front_resize_input, f'{"medium risk"}', (x3 + 5, y3 + 40), Font, FontSize,
                            (0, 255, 255), 1)
                cv2.rectangle(video_front_resize_input, (x3, y3), (x4, y4), (0, 0, 255), 1)
                # cv2.line(video_front_resize_input, (cx, cy), (540, 719), (0, 0, 255), 2)
            else:
                # dangerous   : sd is plus and dd is minus and t < 1, then the object is getting closer and is near the line of motion of your vehicle
                # and also having a dangerous potential in 1 second approachable.
                cv2.putText(video_front_resize_input, f'{"Type:"}{"not supported"}{"(Vehicle_ID)"}', (x3 + 5, y3 + 10),
                            Font, FontSize, (0, 255, 255), 1)
                cv2.putText(video_front_resize_input, f'{distance}{"closer"}', (x3 + 5, y3 + 25), Font, FontSize,
                            (0, 255, 255), 1)
                cv2.putText(video_front_resize_input, f'{"high risk"}', (x3 + 5, y3 + 40), Font, FontSize,
                            (0, 255, 255), 1)
                cv2.rectangle(video_front_resize_input, (x3, y3), (x4, y4), (0, 0, 255), 1)
                # cv2.line(video_front_resize_input, (cx, cy), (540, 719), (0, 0, 255), 2)
        else:
            # unknown ( #if sd < 0 and dd < 0: )
            cv2.putText(video_front_resize_input, f'{"Type:"}{"not supported"}{"(Vehicle_ID)"}', (x3 + 5, y3 + 10),
                        Font, FontSize, (0, 255, 255), 1)
            cv2.putText(video_front_resize_input, f'{distance}{"calculating..."}', (x3 + 5, y3 + 25), Font, FontSize,
                        (0, 255, 255), 1)
            cv2.putText(video_front_resize_input, f'{"unknown"}', (x3 + 5, y3 + 40), Font, FontSize, (0, 255, 255), 1)
            cv2.rectangle(video_front_resize_input, (x3, y3), (x4, y4), (0, 0, 255), 1)
            # cv2.line(video_front_resize_input, (cx, cy), (540, 719), (0, 0, 255), 2)

    # ============================================================================================ #
    # -----------------------------------------Rear Camera---------------------------------------- #
    # ============================================================================================ #

    # Object Detect using predict from YOLO and Input Video
    result_r = model.predict(video_rear_resize_input)                           # Object Detection using YOLO
    resbb_r = result_r[0].boxes.boxes                                           # Bounding Box
    px_r = pandas.DataFrame(resbb_r.cpu().detach().numpy()).astype("float")     # all detected vehicle's list in px
    # px_f = pandas.DataFrame(resbb_r).astype("float")  # all detected vehicles's list in px

    list_r = []  # in List, save the each frame information of detected object's x1,x2,y1,y2 value

    for index, row in px_r.iterrows():
        x1_r = int(row[0])  # left upper's x coordinate
        y1_r = int(row[1])  # left upper's y coordinate
        x2_r = int(row[2])  # right bottom's x coordinate
        y2_r = int(row[3])  # right bottom's y coordinate
        d_r = int(row[5])
        c_r = class_list[d_r]  # print("class_list", c) -> if car(if d = 1), then show car(confirmed!)
        # if it is vehicles, then appended in list
        if 'car' or 'motorcycle' or 'bus' or 'truck' in c:
            list_r.append([x1_r, y1_r, x2_r, y2_r])
    bbox_id_r = tracker_r.update_r(list_r)
    for bbox_r in bbox_id_r:
        x3_r, y3_r, x4_r, y4_r, sd_r, dd_r, id_r, nr_r = bbox_r  # appended 4/4
        # x3,y3 :  left upper's coordinate, x4,y4 :  right bottom's coordinate
        # sd : space difference             if sd minus  then further, if sd plus closer
        # dd : distance difference          if dd minus then closer, if dd plus further
        # id : class_id
        # nr : vehicle identification nr(unsupported)
        cx_r = int(x3_r + x4_r) // 2
        cy_r = int(y3_r + y4_r) // 2

        distance_r = float(format(math.sqrt(math.pow((540 - cx_r), 2) + math.pow((720 - cy_r), 2)), ".3f"))
        if dd_r == 0:
            collision_time = 2
        else:
            collision_time = distance_r / (abs(dd_r) * fps)

        if sd_r < 0 and dd_r > 0:
            # low Risk : sd is minus or dd is plus, then the object is moving away or is stationary.
            cv2.putText(video_rear_resize_input, f'{"Type:"}{"not supported"}{id_r}', (x3_r + 5, y3_r + 10),
                        Font, FontSize, (0, 255, 255), 1)
            cv2.putText(video_rear_resize_input, f'{distance_r}{"further"}', (x3_r + 5, y3_r + 25), Font, FontSize,
                        (0, 255, 255), 1)
            cv2.putText(video_rear_resize_input, f'{"low risk"}', (x3_r + 5, y3_r + 40), Font, FontSize, (0, 255, 255),
                        1)
            cv2.rectangle(video_rear_resize_input, (x3_r, y3_r), (x4_r, y4_r), (0, 0, 255), 1)
            # cv2.line(video_front_resize_input, (cx, cy), (540, 719), (0, 0, 255), 2)
        elif sd_r >= 0 and dd_r > 0:
            # medium Risk : sd is plus and dd is plus, then the object is getting closer but not directly towards your vehicle.
            cv2.putText(video_rear_resize_input, f'{"Type:"}{"not supported"}{id_r}', (x3_r + 5, y3_r + 10),
                        Font, FontSize, (0, 255, 255), 1)
            cv2.putText(video_rear_resize_input, f'{distance_r}{"closer"}', (x3_r + 5, y3_r + 25), Font, FontSize,
                        (0, 255, 255), 1)
            cv2.putText(video_rear_resize_input, f'{"medium risk"}', (x3_r + 5, y3_r + 40), Font, FontSize,
                        (0, 255, 255), 1)
            cv2.rectangle(video_rear_resize_input, (x3_r, y3_r), (x4_r, y4_r), (0, 0, 255), 1)
            # cv2.line(video_front_resize_input, (cx, cy), (540, 719), (0, 0, 255), 2)
        elif sd_r >= 0 and dd_r <= 0:
            if collision_time >= 1.5:
                # high 	 Risk : sd is plus and dd is minus, then the object is getting closer and is near the line of motion of your vehicle.
                cv2.putText(video_rear_resize_input, f'{"Type:"}{"not supported"}{id_r}',
                            (x3_r + 5, y3_r + 10), Font, FontSize, (0, 255, 255), 1)
                cv2.putText(video_rear_resize_input, f'{distance_r}{"closer"}', (x3_r + 5, y3_r + 25), Font, FontSize,
                            (0, 255, 255), 1)
                cv2.putText(video_rear_resize_input, f'{"medium risk"}', (x3_r + 5, y3_r + 40), Font, FontSize,
                            (0, 255, 255), 1)
                cv2.rectangle(video_rear_resize_input, (x3_r, y3_r), (x4_r, y4_r), (0, 0, 255), 1)
                # cv2.line(video_front_resize_input, (cx, cy), (540, 719), (0, 0, 255), 2)
            else:
                # dangerous   : sd is plus and dd is minus and t < 1, then the object is getting closer and is near the line of motion of your vehicle
                # and also having a dangerous potential in 1 second approachable.
                cv2.putText(video_rear_resize_input, f'{"Type:"}{"not supported"}{id_r}',
                            (x3_r + 5, y3_r + 10), Font, FontSize, (0, 255, 255), 1)
                cv2.putText(video_rear_resize_input, f'{distance_r}{"closer"}', (x3_r + 5, y3_r + 25), Font, FontSize,
                            (0, 255, 255), 1)
                cv2.putText(video_rear_resize_input, f'{"high risk"}', (x3_r + 5, y3_r + 40), Font, FontSize,
                            (0, 255, 255), 1)
                cv2.rectangle(video_rear_resize_input, (x3_r, y3_r), (x4_r, y4_r), (0, 0, 255), 1)
                # cv2.line(video_front_resize_input, (cx, cy), (540, 719), (0, 0, 255), 2)
        else:
            # unknown ( #if sd < 0 and dd < 0: )
            cv2.putText(video_rear_resize_input, f'{"Type:"}{"not supported"}{id_r}', (x3_r + 5, y3_r + 10),
                        Font, FontSize, (0, 255, 255), 1)
            cv2.putText(video_rear_resize_input, f'{distance_r}{"calculating..."}', (x3_r + 5, y3_r + 25), Font,
                        FontSize, (0, 255, 255), 1)
            cv2.putText(video_rear_resize_input, f'{"unknown"}', (x3_r + 5, y3_r + 40), Font, FontSize, (0, 255, 255),
                        1)
            cv2.rectangle(video_rear_resize_input, (x3_r, y3_r), (x4_r, y4_r), (0, 0, 255), 1)
            # cv2.line(video_front_resize_input, (cx, cy), (540, 719), (0, 0, 255), 2)

    # ============================================================================================ #
    # -------------------------------------Front_Lane_Finding------------------------------------- #
    # ============================================================================================ #
    """
    # Wrapping area Test (Bird Eye View)
    wrapped_test, minverse = wrapping_f(video_front_resize_input)

    cv2.circle(video_front_resize_input, (x_bottom_left_src_f, y_bottom_left_src_f), 4, (0, 0, 255), -1)
    cv2.circle(video_front_resize_input, (x_top_left_src_f, y_top_left_src_f), 4, (0, 0, 255), -1)
    cv2.circle(video_front_resize_input, (x_top_right_src_f, y_top_right_src_f), 4, (0, 0, 255), -1)
    cv2.circle(video_front_resize_input, (x_bottom_right_src_f, y_bottom_right_src_f), 4, (0, 0, 255), -1)

    cv2.circle(wrapped_test, (x_bottom_left_dst_f, y_bottom_left_dst_f), 4, (0, 255, 255), -1)
    cv2.circle(wrapped_test, (x_top_left_dst_f, y_top_left_dst_f), 4, (0, 255, 255), -1)
    cv2.circle(wrapped_test, (x_top_right_dst_f, y_top_right_dst_f), 4, (0, 255, 255), -1)
    cv2.circle(wrapped_test, (x_bottom_right_dst_f, y_bottom_right_dst_f), 4, (0, 255, 255), -1)

    cv2.imshow("wrapped_test_original", video_front_resize_input)
    cv2.imshow("wrapped_test_wrapped", wrapped_test)
    """

    """
    # ROI area Test (Bird Eye View)
    w_f_img = color_filter(wrapped_img)
    roi_result = roi(w_f_img)
    
    cv2.circle(roi_result, (int(0.05 * x), int(0.95 * y)), 4, (0, 0, 255), -1)
    cv2.circle(roi_result, (int(0.05 * x), int(0.01 * y)), 4, (0, 0, 255), -1)
    cv2.circle(roi_result, (int(0.45 * x), int(0.01 * y)), 4, (0, 0, 255), -1)
    cv2.circle(roi_result, (int(0.45 * x), int(0.94 * y)), 4, (0, 0, 255), -1)
    cv2.circle(roi_result, (int(0.60 * x), int(0.94 * y)), 4, (0, 0, 255), -1)
    cv2.circle(roi_result, (int(0.60 * x), int(0.01 * y)), 4, (0, 0, 255), -1)
    cv2.circle(roi_result, (int(1.10 * x), int(0.01 * y)), 4, (0, 0, 255), -1)
    cv2.circle(roi_result, (int(0.95 * x), int(0.95 * y)), 4, (0, 0, 255), -1)
    cv2.circle(roi_result, (int(0.11 * x), int(0.95 * y)), 4, (0, 0, 255), -1)
    """

    # Lane Check Part
    if scanning_state == 'left':
        front_out, angle_f, colorwarp_f, draw_poly_img_f, rotated_image_f = lane_finding_pipeline(video_front_resize_input_l, video_front_resize_input, "front", init_f)
        front_out, angle_f, colorwarp_f, draw_poly_img_f, rotated_image_f = lane_finding_pipeline_lr(video_front_resize_input_l, front_out, 'left', init_lr)
        #cv2.imshow('1', front_out)
    elif scanning_state == 'right':
        front_out, angle_f, colorwarp_f, draw_poly_img_f, rotated_image_f = lane_finding_pipeline(video_front_resize_input_l, video_front_resize_input, "front", init_f)
        front_out, angle_f, colorwarp_f, draw_poly_img_f, rotated_image_f = lane_finding_pipeline_lr(video_front_resize_input_l, front_out, 'right', init_lr)
        #cv2.imshow('2', front_out)
    else:
        front_out, angle_f, colorwarp_f, draw_poly_img_f, rotated_image_f = lane_finding_pipeline(video_front_resize_input_l, video_front_resize_input, "front", init_f)
        #cv2.imshow('3', front_out)

    #cv2.imshow('4', front_out)

    if angle_f > 1.5 or angle_f < -1.5:
        init_f = True
    else:
        init_f = False

    # Lane changeable check
    elapsed_time = time.time() - start_time

    if elapsed_time >= 5:  # 3 seconds
        if scanning_state == 'left':
            if any_vehicle_in_left_lane:            # if left lane would be occupied
                print("left lane change not suggestion")
                scanning_state = 'center'
                start_time = time.time()            # Reset the timer
                any_vehicle_in_left_lane = False    # Reset the left lane state
            else:
                print("left lane chage suggestion")
                scanning_state = 'center'
                start_time = time.time()            # Reset the timer
        elif scanning_state == 'right':
            if any_vehicle_in_right_lane:
                print("right lane change not suggestion")
                scanning_state = 'center'
                start_time = time.time()            # Reset the timer
                any_vehicle_in_right_lane = False
            else:
                print("right lane chage suggestion")
                scanning_state = 'center'
                start_time = time.time()            # Reset the timer

        else:
            scanning_state = 'center'
            start_time = time.time()                # Reset the timer



    # ============================================================================================ #
    # -----------------------------------------SteerWheel----------------------------------------- #
    # ============================================================================================ #
    # Get the shape of the original image and the steering wheel image
    height_f, width_f, _ = front_out.shape
    height_f_s, width_f_s, _ = rotated_image_f.shape
    scale_factor_f = 0.25                               # 50% of the original size

    # Calculate new dimensions
    new_width_f = int(width_f_s * scale_factor_f)
    new_height_f = int(height_f_s * scale_factor_f)

    # Resize the image
    rotated_image_resized_f = cv2.resize(rotated_image_f, (new_width_f, new_height_f))
    height_f_s, width_f_s, _ = rotated_image_resized_f.shape

    # Position to overlay the steering wheel image at the bottom-middle of the original image
    position_f = (width_f // 2 - width_f_s // 2, int(height_f * 0.75) - height_f_s // 2)

    # Create a region of interest in the original image where the steering wheel will be placed
    roi_f = front_out[position_f[1]:position_f[1] + height_f_s, position_f[0]:position_f[0] + width_f_s]

    # Resize the rotated_image to fit the ROI
    rotated_image_resized_f = cv2.resize(rotated_image_f, (roi_f.shape[1], roi_f.shape[0]))

    # Then blend the images
    blended_f = cv2.addWeighted(roi_f, 1, rotated_image_resized_f, 0.8, 0)

    # Place the blended image back into the original image
    front_out[position_f[1]:position_f[1] + height_f_s, position_f[0]:position_f[0] + width_f_s] = blended_f

    # Display the final image with the overlay
    #cv2.imshow('camera_front_input', front_out)

    # ============================================================================================ #
    # -------------------------------------------Output------------------------------------------- #
    # ============================================================================================ #
    # Output Video
    # for example... cv2.imshow("camera_front_input", video_front_resize_input)
    cv2.imshow('camera_front_input', front_out)
    
    # Video Export
    output_front.write(front_out)       # Video 2/3
    output_rear.write(video_rear_resize_input)

    if cv2.waitKey(1) == ord('q'):
        break

video_front_raw_input.release()
video_rear_raw_input.release()
output_front.release()
output_rear.release()                   # Video 3/3

cv2.destroyAllWindows()
################################################################################################