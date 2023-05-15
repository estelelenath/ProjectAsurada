# https://moon-coco.tistory.com/entry/OpenCV%EC%B0%A8%EC%84%A0-%EC%9D%B8%EC%8B%9D

import cv2
import youtube_dl
import pafy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
# pip install youtube_dl && pip install pafy

# Load video from youtube
#url = 'https://www.youtube.com/watch?v=ipyzW38sHg0'
#video = pafy.new(url)
#best = video.getbest(preftype = 'mp4')

cap = cv2.VideoCapture("D:\ProjectAsurada\ProjectAsurada\VideoSample\FrontCameraTestnnn.mp4")
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

#Video Record
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out1 = cv2.VideoWriter('D:\ProjectAsurada\ProjectAsurada\VideoSample\OutputSample.mp4', fourcc, 20.0, frame_size)


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

while True:
    retval, img = cap.read()
    if not retval:
         break
    
    ##ckeck the points through the circle for the original image
    #cv2.circle(img, center, radius, color, thickness=1, lineType=8, shift=0)
    (h, w) = (img.shape[0], img.shape[1])
    thickness = 3
    #cv2.circle(img, (int(w * 4.8 // 10), int(h * 0.62)), thickness, (255,0,0), -1)
    #cv2.circle(img, (int(w * 6.1 // 10), int(h * 0.62)), thickness, (255,0,0), -1)
    #cv2.circle(img, (int(w * 0.20), int(h)), thickness, (255,0,0), -1)
    #cv2.circle(img, (int(w* 0.85), int(h)), thickness, (255,0,0), -1)

    ## original video show
    cv2.imshow("video", img)
    
    ## wrapped video show (bird eye-version)
    wrapped_img, minverse = wrapping(img)
    
    #ckeck the points through the circle for the wrapped image  
    # cv2.circle(wrapped_img, (int(200), int(0)), thickness, (255,0,0), -1)
    # cv2.circle(wrapped_img, (int(w-380), int(0)), thickness, (255,0,0), -1)
    # cv2.circle(wrapped_img, (int(400), int(h)), thickness, (255,0,0), -1)
    # cv2.circle(wrapped_img, (int(w-150), int(h)), thickness, (255,0,0), -1)

    #cv2.imshow('wrapped', wrapped_img)

    # color filter and mask (yellow and white lane)
    w_f_img = color_filter(wrapped_img)
    #cv2.imshow('w_f_img', w_f_img)

    ##ROI from color filtered image
    w_f_r_img = roi(w_f_img)

    # cv2.circle(w_f_r_img, (int(0.23*w), int(h)), thickness, (255,0,255), -1)
    # cv2.circle(w_f_r_img, (int(0.23*w), int(0.1*h)), thickness, (255,0,255), -1)
    # cv2.circle(w_f_r_img, (int(0.45*w), int(0.1*h)), thickness, (255,0,0), -1)
    # cv2.circle(w_f_r_img, (int(0.5*w), int(h)), thickness, (255,0,255), -1)
    # cv2.circle(w_f_r_img, (int(0.6*w), int(h)), thickness, (255,0,0), -1)
    # cv2.circle(w_f_r_img, (int(0.6*w), int(0.1*h)), thickness, (255,0,0), -1)
    # cv2.circle(w_f_r_img, (int(0.75*w), int(0.1*h)), thickness, (255,0,0), -1)
    # cv2.circle(w_f_r_img, (int(0.75*w), int(h)), thickness, (255,0,0), -1)
    # cv2.circle(w_f_r_img, (int(0.3*w), int(h)), thickness, (255,255,0), -1)

    #cv2.imshow('w_f_r_img', w_f_r_img)

    ## ROI and wrapped img threshold
    _gray = cv2.cvtColor(w_f_r_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(_gray, 140, 195, cv2.THRESH_BINARY)
    #cv2.imshow('threshold', thresh)

    ## 선 분포도 조사 histogram
    leftbase, rightbase = plothistogram(thresh)
    #plt.plot(hist)
    #plt.show()

    ## histogram 기반 window roi 영역
    draw_info = slide_window_search(thresh, leftbase, rightbase)
    #plt.plot(left_fit)
    #plt.show()

    ## 원본 이미지에 라인 넣기
    meanPts, result = draw_lane_lines(img, thresh, minverse, draw_info)
    cv2.imshow("result", result)

    ## 동영상 녹화
    #out1.write(result)

    key = cv2.waitKey(25)
    if key == 27:
    	break

if cap.isOpened():
	cap.release()
cv2.destroyAllWindows()