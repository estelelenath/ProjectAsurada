# <p align="center">ProjectAsurada</p>

### <p align="center">Driving Assistence System supported by Embedded System 'Jetson'</p>
<br/>

<p align="center">
    <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/title_001.png?raw=true" width="490" height="490"></center>
</p>

<br/>
<br/>
<br/>
<br/>

## The goals / steps of this project are the following:
 - Step 1: Research and analysis the Optimal Algorithms for Vehicle and Lane Detection in Autonomous Driving and integrate Vehicle and Lane Detection in Live Streaming using two cameras and Update to the Latest 2023 Version.
   - Vehicle Detection
   - Lane Detection
 - Step 2: Advanced Driving Assistance Systems with Advanced Features and Functionality
   - Vehicle Tracking
   - Speed / Distnace Estimation
   - Risk judgment Algorithm (Pop up Rear Camera screen)
   - Steering(a left-right distance adjustment function) and Speed control from front vehicle(a front-rear distance adjustment function)
   - Safe Lane Suggestion (incl. Multiple Lanes)
   - Auto Driving support system using two cameras
 - Step 3: Advanced Driving Assistance Systems with Extended Features in Simulation and Virtual Reality
   - VID(Vehicle Identification Number) and Traffic Signal recognition
   - Current traffic situation(bird eye) -> M.A.P
   - Advanced Lane Detection based on Deep Learning
   - Hardware Develop Environment
     - Virtual Reality
       - Visualisation
       - Real-time Data Transfer(implemented Driving Assistance System)
     - CUDA
   - Simulation
     - A.I Simulation (Unity / ROS)
     - Real-time Data Transfer(implemented Driving Assistance System)
 - Step 4: System optimization (including Jetson embedded systems) and additional sensor testing and final evaluation
   - System Optimization
     - Jetson Embedded System
     - etc.
   - Final Evaluation and Conclusion
 
<br/>



<p align="center">
    <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/plan_010.png?raw=true" width="957" height="538"></center>
</p>
<p align="center">Workflow Version 2.</p>
<br/>


<br/>


<br/>

<figure>
    <p align="center">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/titleGif.gif?raw=true" width="400" height="225"></center>
    </p>
    <figcaption align="center">Asurada at the Future GPX Cyber Formula, 1996.</figcaption>
</figure>

<br/>

## Step 1. Basic Functions - Warming up
***
Goal : This step delves into the critical components of autonomous driving: object detection and lane detection. 
A comprehensive examination of algorithms is embarked, evaluating their strengths and weaknesses, to discern those best suited for applications in autonomous driving.
By focusing on the most recent advancements, It is ensured that the analysis remains grounded in the state-of-the-art methodologies and datasets pertinent to 2023.

<br/>

## Vehicle Detection
Objective : In the sphere of autonomous driving, object recognition—with a particular emphasis on vehicle detection—stands as a foundational function. It is not hyperbolic to assert that the genesis of all developmental strides in this implementation can be traced back to this fundamental capability. The preliminary phase of implementation is anchored in vehicle detection, and the results derived from this process serve as the bedrock for subsequent tasks, such as vehicle tracking and the measurement of speed and distance.
Therefore, in the process of implementing vehicle detection, it is necessary to think about the important considerations.
Primarily, Firstly, there is an need for the ability of the precise real-time detection and classification of vehicles. 
Concurrently, the system must be adept at processing video feeds expeditiously, ensuring instantaneous detection for prompt decision-making. 
Therefore, how accurate object recognition and how fast it can be done will be the most important objective and the dual objectives of accuracy in object recognition and its expeditious execution time emerge as cardinal benchmarks.
It is also imperative to note that our testing paradigm is rooted in real-world conditions, eschewing controlled laboratory environments. This places an added onus on the system's robustness. 
Even in the capricious nature of external factors such as lighting, weather, and traffic infrastructure environment, we need a certain level of reliability. 
### Algorithm
As mentioned above, the comparison focused on the accuracy and speed of vehicle detection, and the robustness was developed by adjusting the threshold value within the implementation.
Here are the most frequently used algorithms for vehicle detection, along with their pros and cons.

 - Haar Cascade Classifier : The Haar Cascade Classifier is predominantly recognized for its prowess in face detection, though it can be trained to detect other objects. It boasts of speed, being lightweight, and the availability of pre-trained models, especially for face detection in OpenCV. However, its default classifiers are tailored for frontal faces, making the detection of faces in varied orientations challenging
 - HOG(Histogram of Oriented Gradients) : HOG is a feature descriptor predominantly used for object detection. Its claim to fame has been its effectiveness in pedestrian detection.
 - R-CNN(Regions with CNN) : R-CNN, which stands for Regions with Convolutional Neural Networks, is an object detection algorithm that has high accuracy. One of its primary strengths is its use of selective search to generate region proposals, ensuring precision in object detection. However, this meticulous approach comes with its drawbacks. R-CNN is notably slow, making it unsuitable for real-time processing. Additionally, it demands a significant amount of memory, which can be a limitation in certain applications.
 - Fast R-CNN : As an evolution of R-CNN, Fast R-CNN offers improved speed. It achieves this by employing ROI (Region of Interest) pooling for extracting fixed-size feature maps. Despite its name suggesting speed, Fast R-CNN still leans on selective search for region proposals, which can be a bottleneck in terms of processing speed.
 - Faster R-CNN : Faster R-CNN takes the advancements of Fast R-CNN a step further. It incorporates a Region Proposal Network (RPN) for generating region proposals, making it notably faster than its predecessor. However, this speed comes at the cost of a more intricate architecture, which necessitates extended training periods.
 - YOLO(You Only Look Once) : YOLO revolutionizes object detection with its blazing speed. Designed for real-time processing, YOLO can detect objects in a single forward pass through the network. While its speed is commendable, it does compromise on accuracy, particularly when detecting smaller objects.
 - SSD(Single Shot MultiBox Detector) : SSD strikes a balance between the speed of YOLO and the accuracy of Faster R-CNN. It amalgamates the strengths of both, offering a harmonious blend of speed and precision. However, similar to YOLO, SSD can sometimes falter when detecting smaller objects.
 - RetinaNet : RetinaNet stands out for its use of Focal Loss, a technique specifically designed to address class imbalance issues. This ensures high accuracy in object detection. However, its sophisticated architecture demands more computational resources, which might be a constraint in some scenarios.

For the dynamic and unpredictable realm of autonomous driving, real-time processing is important, assuming that it has a certain level of reliability.
Also, ultimately, at the end of our testing, we aim to optimize in embedded systems. Therefore, it is necessary to select a model that is lightweight, fast and accurate.
There is direct comparison data for Faster R-CNN and YOLO, so I quote it.
<figure>
    <p align="center">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/yolo_benchmark.gif?raw=true" width="400" height="225">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/fasterrcnnyolo_benchmark.gif?raw=true" width="400" height="225">
    </p>
    <figcaption align="center"> Comparison of YOLO and Faster R-CNN ( left: YOLO / right: Faster R-CNN )</figcaption>
    <figcaption align="center"> source: https://github.com/alen-smajic/Real-time-Object-Detection-for-Autonomous-Driving-using-Deep-Learning</figcaption>
</figure>

<br/>
 

<figure>
    <p align="center">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/yoloFasterRCnn2.jpg?raw=true" width="400" height="225">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/yolo-comparison-plots.png?raw=true" width="400" height="225">
    </p>
    <figcaption align="center"> Left: Comparison of YOLO and Faster R-CNN (mAP%*: Mean Average Precision. FPS: Frames per second.) </figcaption>
    <figcaption align="center"> Right: benchmark of YOLO performance by version</figcaption>
    <figcaption align="center"> source: https://github.com/ultralytics/ultralytics</figcaption>
</figure>

<br/>

<div align="center">

~~~
Performance comparison by YOLO mode
~~~
</div>


<div align="center">

   Model | Size(pixels) | mAP^val 50-95 | Speed CPU ONNX(ms) | FLOPs     
  :------------: | :-------------: | :-------------: | :-------------: | :-------------:
   YOLO8n | 640 | 37.3 | 80.4 | 8.7
   YOLO8s | 640 | 44.9 | 128.4 | 28.6
   YOLO8m | 640 | 50.2 | 234.7 | 78.9
   YOLO8l | 640 | 52.9 | 375.2 | 165.2
   YOLO8x | 640 | 53.9 | 479.1 | 257.8
</div>


In the table above, you can consider mAP % as accuracy and FPS as processing speed.
As can be seen from the table, in terms of accuracy, Faster R-CNN has better values and YOLO processes data faster.
Therefore, algorithms such as YOLO and SSD, which focus on speed, is considered as front runners.
In addition, in the case of YOLO, it has been updated to the current (summer 2023) YOLOv8 model, and as a result of continuous upgrades, it is superior to other models in terms of accuracy and processing speed is faster, so we decided to use the latest model, YOLOv8 in this test.
<br/>

### Implementation
```python
model = YOLO('yolov8n.pt')

# Object Detect using predict from YOLO and Input Video
    result_f = model.predict(video_front_resize_input)                          # Object Detection using YOLO
    resbb_f = result_f[0].boxes.boxes                                           # Bounding Box
```
### Evaluation
<figure>
    <p align="center">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/01_video_rear_output_recctangleBox.gif?raw=true" width="400" height="225">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/02_video_front_output_recctangleBox.gif?raw=true" width="400" height="225">
    </p>
    <figcaption align="center"> left: Video for front camera / right: Video for rear camera</figcaption>
</figure>


<br/>

## Lane Detection
Objective : Similar to previously implemented vehicle detection algorithm models, lane detection in the autonomous driving is implemented with particular emphasis on accuracy and fast processing speed. 
Likewise, a certain level of reliability must be secured in various weather environments and external factors.
### Algorithm

The lane recognition algorithm can be divided into a traditional method and a method that utilizes Deep Learning. 
The traditional Hough Transform or Sobel Operator have heavy functions, lacks curve recognition, or is vulnerable to external change factors.
However, since the use of CUDA using a graphics card was strongly recommended in the lane recognition model using deep learning, the lane recognition model using histogram was implemented by itself as a basic method in this chapter by separating the white and yellow lanes with a filter, and later In the expanded function, we will implement an advanced lane recognition model based on deep learning.

The overall flow first pre-processes the distortion of the image, prepares for lane recognition by wrapping, also known as the bird's eye effect, and converts it to a binary image using a color filter. Lanes are recognized using a sliding window in a binary image, and reverse wrapping is used to recognize lanes in the original image.

#### Step 1: Distortion and Camera Calibration

The first lane we see (perceived by the vehicle's camera) is affected by the perspective. Therefore, distortion and perspective are corrected through camera correction (camera calibration) and wrapping (perspective correction or bird's eye view).

Current cheap camera makes a distortion of images, main distortions are radial- / tangential - distortion.
However such a distortion of images could be significantly  corrected by basic function, camera calibration of opencv.
Wrapping is controlled by adjusting the configuration of preset points.
This currently requires configuration each time, but will require automation in the future.

<figure>
    <p align="center">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/00_originalFrontView.gif?raw=true" width="400" height="225">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/03_Birdeyeview.gif?raw=true" width="400" height="225">
    </p>
    <figcaption align="center"> left: Original Image / right: Birdeye view </figcaption>
</figure>

The corrected image uses a color filter to detect white and yellow lanes, internationally accepted lane standards. In this case, HLS was used. Other colors are excluded as they are not necessary for lane detection.

#### Step 2: Color Filter

<figure>
    <p align="center">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/00_originalFrontView.gif?raw=true" width="400" height="225">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/04_colorfilter.gif?raw=true" width="400" height="225">
    </p>
    <figcaption align="center"> left: Original Image / right: White/Yellow Filter Image </figcaption>
</figure>

The detected image redefines the part to be observed through ROI once again.
The reason for this process is that the road infrastructure may not always be in the best condition (eg cracked asphalt roads, all things we are not interested).
However, in this implementation, the road infra condition was nice, so the use of ROI had no effect and was omitted. However, the code has been implemented in the implementation part, so it can be refered to it.

#### Step 3: Sliding Window

As mentioned earlier, the general Hough Transform or Sobel Operator is heavy and vulnerable to curve detection, so lane detection was performed using a histogram.
Search using a sliding window, and the histogram of the searched points appears as shown below. This means that when you add up all the row values(y-values) for one column(x-values), it will have a relatively large value if there exists a lane, and a small value otherwise.
Since a lane must have continuous row values (height or y value), in histogram classification, the sum is relatively 255 consecutive sums, that is, a very large value.
Conversely, if no lanes are detected, this value approaches 0.

<figure>
    <p align="center">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/05_SlidingWindow.gif?raw=true" width="400" height="225">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/06_histogram.png?raw=true" width="400" height="225">
    </p>
    <figcaption align="center"> left: Sliding Window / right: Histogram detection value </figcaption>
</figure>

### Implementation
```python
configuration of preset points of wrapping image

# Source point of front camera (for main Lane)
# (1)                                  # top_left(1)                         # top_right(2)
x_top_left_src_f        = 480               #(x_1,y_1)###################(x_2,y_2)#
y_top_left_src_f        = 390                #                                   #
# (2)                                         #                                 #
x_top_right_src_f       = 565                  #                               #
y_top_right_src_f       = 390                   #                             #
# (3)                                            #                           #
x_bottom_left_src_f     = 110                     #(x_3,y_3)#######(x_4,y_4)#
y_bottom_left_src_f     = 690             # bottom_left(3)          # bottom_right(4)
# (4)
x_bottom_right_src_f    = 885
y_bottom_right_src_f    = 690

# Destination point of front camera (for main Lane)
# (1)                                 # top_left(1)                         # top_right(2)
x_top_left_dst_f        = 55    #10        #(x_1,y_1)###################(x_2,y_2)#
y_top_left_dst_f        = 0                #                                     #
# (2)                                      #                                     #
x_top_right_dst_f       = 1035   #1070     #                                     #
y_top_right_dst_f       = 0                #                                     #
# (3)                                      #                                     #
x_bottom_left_dst_f     = 150   #480       #(x_3,y_3)###################(x_4,y_4)#
y_bottom_left_dst_f     = 720      # bottom_left(3)                   # bottom_right(4)
# (4)
x_bottom_right_dst_f    = 880    #600
y_bottom_right_dst_f    = 720
```
### Evaluation

<figure>
    <p align="center">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/07_laneDetectionEvaluation.gif?raw=true" width="400" height="225">
    </p>
    <figcaption align="center"> Video for front camera</figcaption>
</figure>

<br/>

## Step 2. Advanced Features - First Half
***
Goal : 
We are now implementing detailed and advanced functions for autonomous driving.
Continuous vehicle tracking is implemented using vehicle detection in previous chapter.
Vehicle tracking is an essential function for speed measurement, distance measurement, and autonomous driving that will be implemented in this and future chapter.
It will be also implemented a risk judgment system that determines the risk of other vehicles by measuring speed and distance.
Since every moment on the road can lead to a dangerous moment, one of the most important factors is to determine the level of risk and warn the driver to ensure the safety of the driver.
Lane recognition and surrounding vehicle recognition provide steering assistance and the already famous driving assistance system called cruise driving.
Lastly, in addition to determining the level of risk, warning, or assisting driving, we use the front and rear cameras to suggest lane changes and provide support throughout the entire driving.
This comprehensive assistance is the reason it is named this project 'Asurada'.

## Vehicle Tracking 
Objective : Vehicle detection and vehicle tracking are problems of completely different scales.
However, for a system that is updated every moment, it is a very difficult problem to determine whether the vehicle observed in this moment is “the car” from the previous.
Therefore, vehicle tracking can be said to determine whether the vehicle we see now is the vehicle we saw a moment ago.
### Algorithm
In order to implement what we said in the objective, we first need to define 'moment' as a 'frame'.
So, based on the center point of the detected vehicle in each frame, if the distance in the next frame is less than to the distance we specify, we consider it to be the same vehicle.
This distance is therefore linked to the accuracy of vehicle tracking directly.
If this distance is too long, we will judge that most vehicles are the same vehicle. Conversely, if this distance is too strictly short, most vehicles will be recognized as the different vehicle, even it is same.
### Implementation

```python
for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 35:
                    self.center_points[id] = (cx, cy)
                    self.box_size[id] = w * h

                    self.distance_from_other[id] = math.dist((cx, cy), self.user_vehicle_point)

                    objects_bbs_ids.append([x1, y1, x2, y2, self.space_difference, self.distance_difference_from_other_vehicle, id, "id_nr"])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                self.box_size[self.id_count] = w * h
                self.distance_from_other[self.id_count] = math.dist((cx, cy), self.user_vehicle_point)
                objects_bbs_ids.append([x1, y1, x2, y2, 0, 0, self.id_count, "id_nr"])
                self.id_count += 1
```
### Evaluation

<figure>
    <p align="center">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/08_trackingFront.gif?raw=true" width="400" height="225">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/09_trackingRear.gif?raw=true" width="400" height="225">
    </p>
    <figcaption align="center"> left: Front Camera / right: Rear Camera </figcaption>
</figure>

<br/>

## Speed and Distance Estimation 
Objective : 
The next objective is to estimate the speed and distance.
First, we concluded that the accuracy of a single front camera alone was insufficient to estimate the vehicle's speed. Of course, there is also speed estimation using deep learning, which it can be refered to below. (article: https://arxiv.org/pdf/1907.06989v1.pdf, distribution: https://github.com/joseph-zhong/KITTI-devkit)
The reason we estimate speed is to determine how fast something can approach to driver and how dangerous it is rather than absolute speed itself.
Therefore, relative speed was set as the objective.
Distance can be obtained relatively easily from the scene recognized by the camera.

### Algorithm
To find the relative speed, the rate of change of the bounding box and the rate of change of the distance were used.
As it gets closer to driver, the bounding box will get bigger.
Conversely, the farther away the vehicle is, the smaller the object appears due to perspective, and thus the size of the bounding box becomes smaller.
Therefore, the larger the bounding box and the greater the amount of change, the faster it approaches the driver at a high speed.
However, in object recognition, the change in size of the bounding box was not large, so the change in distance was also used to compensate for this.
The distance from the driver's vehicle (center bottom) to the other vehicle is observed in real time every frame, and this change is more intuitive to the distance between the driver and the other driver.
However, this method of measuring the rate of change in distance also has the disadvantage of lowering reliability if the other person moves sideways or moves away.
Therefore, the rate of change in area of the bounding box and the rate of change in distance are used to measure speed by complementing each other.
### Implementation
```python
self.space_difference = (w * h - self.box_size[id]) / w * h
self.space_difference = round(self.space_difference, 2)
self.box_size[id] = w * h

self.distance_difference_from_other_vehicle = math.dist((cx, cy), self.user_vehicle_point) - self.distance_from_other[id]
self.distance_difference_from_other_vehicle = round(self.distance_difference_from_other_vehicle, 2)
self.distance_from_other[id] = math.dist((cx, cy), self.user_vehicle_point)
```
### Evaluation

<figure>
    <p align="center">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/10_distanceEstiFront.gif?raw=true" width="400" height="225">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/11_distanceEstiRear.gif?raw=true" width="400" height="225">
    </p>
    <figcaption align="center"> left: Distance estimation in real-time for Front Camera / right: for Rear Camera </figcaption>
</figure>

<br/>

## Risk Judgment 
Objective : Previous process, estimating speed, estimating distance, and calculating the rate of change can be said to be a process for determining risk grad.
Therefore, the objective this chapter is to determine how dangerous the other vehicle is to the driver by combining various information, including relative speed and distance, and issue a warning.
### Algorithm

<figure>
    <p align="center">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/Graph2.png?raw=true" width="500" height="500">
    </p>
    <figcaption align="center"> Risk Judgment Algorithm Tree </figcaption>
</figure>

The first factor to be considered in the algorithm for determining risk is distance.
In general, it was judged that the risk was significantly low if the other vehicle was at a certain distance. Based on the collision time, if a vehicle traveling at 100 km/h is further away than the time it takes to recognize and react (2 seconds), the risk was judged to be low.
Also, based on the rate of change of the bounding box and distance measured in the previous chapter, if both values are moving away, low risk. 
if one of the two values is positive, i.e. approaching to driver, the risk grad was judged as caution. 
and if both values are positive, it was judged as dangerous. Based on the assumption that the last distance and frame were based on 24 fps, if the impact time was less than 2 seconds, which is the time required to recognize and react, it was immediately judged to be dangerous.
The rate of change in speed, rate of change in bounding box area, or other variables are considered to require continuous updates and improvements in the future.
Additionally, if the vehicle approaches at high speed from the rear or is too close, information is provided to the driver through a pop-up window.
### Implementation

```python
if sd < 0 and dd > 0 or distance > 350:
            # low Risk : sd is minus or dd is plus, then the object is moving away or is stationary.
        elif sd >= 0 and dd > 0:
            # medium Risk : sd is plus and dd is plus, then the object is getting closer but not directly towards your vehicle.
        elif sd >= 0 and dd <= 0:
            if collision_time >= 1.5:
                # high 	 Risk : sd is plus and dd is minus, then the object is getting closer and is near the line of motion of your vehicle.
            else:
                # dangerous   : sd is plus and dd is minus and t < 1, then the object is getting closer and is near the line of motion of your vehicle
                # and also having a dangerous potential in 1 second approachable.
        else:
            # unknown ( #if sd < 0 and dd < 0: )
```

### Evaluation

<figure>
    <p align="center">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/12_RiskJudgement.gif?raw=true" width="400" height="225">
    </p>
    <figcaption align="center"> Risk Judgment </figcaption>
</figure>

<br/>

## Steering and Speed Control for autonomous Driving 
Objective : Implements automatic control of the steering system through lane recognition and speed control is also implemented, also known as ‘cruise mode,’ which automatically adjusts speed by taking into account the surrounding environment, including vehicles ahead.

### Algorithm
The automatic steering control system is based on the lane detected by lane recognition.
It recognizes left and right lanes in real time and maintains the center point in the middle.
Through the variable location of the center point and multiple center points, it can be anticipated sudden path changes (e.g. curved sections) in advance and improve performance.

The speed can be also controled by checking to see if there is a car ahead in the current lane and, if so, maintaining that distance. Speed control has not been implemented yet, but since the implementation of whether there is a car ahead is already implemented in the next part, 'Safe Lane Suggestion', if it based on that, measuring distance and speed control are not difficult tasks.

### Implementation

```python
    # Calculate left and right line positions at the bottom of the input
    left_x_pos = left_fit[0] * y_max ** 2 + left_fit[1] * y_max + left_fit[2]
    right_x_pos = right_fit[0] * y_max ** 2 + right_fit[1] * y_max + right_fit[2]
    # Calculate the x position of the center of the lane
    center_lanes_x_pos = (left_x_pos + right_x_pos) // 2
```

### Evaluation

<figure>
    <p align="center">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/13_steering.gif?raw=true" width="400" height="225">
    </p>
    <figcaption align="center"> Automatic Steering Control System </figcaption>
</figure>

<br/>

## Safe Lane Suggestion
Objective : Active support is possible beyond passive support based on lane recognition, automatic steering, vehicle detection, and risk judgment.
A representative method is 'Safe Lane Suggestion'.
Driver is trying to change lanes and the system suggests whether this decision is right or not.

### Algorithm
<figure>
    <p align="center">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/14_conceptSafeLane.jpg?raw=true" width="400" height="225">
    </p>
    <figcaption align="center"> Base concept of safe lane suggestion </figcaption>
</figure>

If the driver tries to change to the left or right lane (virtual implementation using the direction keys), it searches the lane and first checks to see if there is a vehicle. If not, it suggests a lane change and warns if there are any obstacles.
The rearview camera can be also used  to monitor oncoming cars in the lane and through the configuration of the other enviroment variables, performance could be improved.

### Implementation
```python
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
```


### Evaluation

<figure>
    <p align="center">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/15_safeLanecheckLeft.gif?raw=true" width="400" height="225">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/16_safeLanecheckRight.gif?raw=true" width="400" height="225">
    </p>
    <figcaption align="center"> Safe lane Suggestion (Left Lane / Right Lane) </figcaption></figure>

<br/>

## Auto Driving support system using two camera 
There are many difficulties in supporting and applying autonomous driving with only a camera, but the limitations of the camera can be overcome by using two or multiple cameras.
Additionally, although there are many existing GitHub materials and experiments using a single camera, there is a lack of information about multiple cameras.
By using multiple cameras, it is possible to figure out the current traffic situation to be implemented in the future through recognition of surrounding objects. 
As already implemented, when assessing risk or changing lanes, vehicles approaching from the front and rear can be recognized in advance and the information can be combined to judge, predict, and make decisions.
With multiple cameras, many things can be implemented and applied.

<br/>
<br/>

## Step 3. Extended Features in Simulation and Virtual Reality - Second Half 
***
Goal : Beyond 'software' aspects of autonomous driving, it can be approached as ultimate project goal by adding remote access using simulation and virtual reality, improving physical hardware performance using CUDA, and various expanded software functions.

## Vehicle Identification Number and Traffic Signal recognition (In Planning)
Objective : The amount of data recorded by cameras during autonomous driving is quite large, and there are various ways to use this data.
In particular, edge computing, a next-generation technology, enables more sharing of data.
license plate recognition is used not only to track vehicles but also to search vehicles in times of emergency. However, it may become an element of Big Brother or personal information infringement, so measures are needed to deal with this.

<br/>

## Current traffic situation (Bird eye view): M.A.P(Monitoring Aerial Perspective) (In Development)
Objective : Being able to see the current state of the road(traffic) from a bird's eye perspective can be more helpful to drivers in understanding the situation.
It is already supported by several car manufacturers, including Tesla, but we will simply implement it with two cameras.
### Algorithm
A bird's eye view is implemented already using front and rear cameras in previous course.
Using  image processing can be used to implement the tentative name M.A.P (Monitoring Aerial Perspective).
Here, there are many contour detection methods such as Sobel and Prewitt, and OpenSV also supports them as a 'cv2.findContours'.

### Implementation
```python
images, contours, hierachy = cv2.findContours(image, mode, method)

```

### Evaluation
<figure>
    <p align="center">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/17_mapConcept.png?raw=true" width="400" height="225">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/loading.gif?raw=true" width="400" height="225">
    </p>
    <figcaption align="center"> left: Concept of M.A.P / right: Contours image processing </figcaption>
</figure>

<br/>


## Advanced Lane Detection based on Deep Learning (In Planning)
Objective : Lane detection is basic, but as a core technology in autonomous driving, stability and reliablility must always be guaranteed.
Additionally, in order to accurately identify lanes with only one camera, lane detection using deep learning will be implemented in the future.

<br/>

## Hardware Development Environment for Virtual Reality (In Development)
Objective : It is focused on diversifying interfaces using virtual reality.
And it is attempting remote data exchange as an opportunity to collect data for next project, remote control using virtual reality.
### Algorithm
### Implementation
### Evaluation

<br/>

## Hardware Development Environment for CUDA
Objective : Such autonomous driving requires processing a vast amount of data, and due to its real-time property, the load on data processing further increases.
General CPU-based data processing has limitations, and data processing using GPU is proposed. In particular, efficiency can be dramatically increased by using CUDA, a tool that makes parallel processing possible.
<figure>
    <p align="center">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/18_CPU.gif?raw=true" width="400" height="225">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/19_GPU.gif?raw=true" width="400" height="225">
    </p>
    <figcaption align="center"> left: CPU / right: GPU </figcaption>
    <figcaption align="center"> source: https://www.youtube.com/watch?v=-P28LKWTzrI </figcaption>
</figure>

### Implementation

<div align="center">

~~~
CUDA Install and Configuration at Desktop
~~~
</div>

<div align="center">

   
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  +----------------------------------+
  |     Model     |      Version     |
  +===============+==================+
  |  GPU          |   GeFroce 4070 Ti|
  |  CUDA cores   |   7680           |
  |  CUDA         |   11.8           |
  |  cudnn        |   8.9.0          |
  |  torch        |   2.0.1+cu118    |
  |  torchaudio   |   2.0.2+cu118    |
  |  torchvision  |   0.15.2+cu118   |
  |  OS           |   Windows 11     |
  +----------------------------------+
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

</div>
* Don't forget a enviroment variables
  
<div align="center">

~~~
CUDA Install and Configuration at Jetson Embedded Board (Jetson Nano)
~~~
</div>

<div align="center">

   
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  +----------------------------------+
  |     Model     |      Version     |
  +===============+==================+
  |  GPU          |     Maxwell      |
  |  Jetpack Ver. |      4.6.1       |
  |  CUDA cores   |      128         |
  |  CUDA         |      10.2        |
  |  cudnn        |      8.2.1       |
  |  torch        |      v1.10.0     |
  |  torchaudio   |      v0.10.0     |
  |  torchvision  |      v0.11.0     |
  |  TensorFlow   |      2.7.0       |
  |  OS           |   Ubuntu 18.04   |
  +----------------------------------+
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   

</div>

<div align="center">

~~~
CUDA Install and Configuration at Jetson Embedded Board (Jetson Xavier)
~~~
</div>

<div align="center">

   
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  +----------------------------------+
  |     Model     |      Version     |
  +===============+==================+
  |  GPU          |      Volta       |
  |  Jetpack Ver. |      4.6.1       |
  |  CUDA cores   |      512         |
  |  CUDA         |      10.2        |
  |  cudnn        |      8.2.1       |
  |  torch        |      v1.10.0     |
  |  torchaudio   |      v0.10.0     |
  |  torchvision  |      v0.11.0     |
  |  TensorFlow   |      2.7.0       |
  |  OS           |   Ubuntu 18.04   |
  +----------------------------------+
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   

</div>

<div align="center">

~~~
CUDA Install and Configuration at Jetson Embedded Board (Jetson Orin Nano)
~~~
</div>

<div align="center">

   
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  +----------------------------------+
  |     Model     |      Version     |
  +===============+==================+
  |  GPU          |      Ampere      |
  |  Jetpack Ver. |      5.1.0       |
  |  CUDA cores   |      512/1024    |
  |  CUDA         |      11.4.19     |
  |  cudnn        |      8.6.0       |
  |  torch        |      v2.0.0      |
  |  torchaudio   |      v0.13.1     |
  |  torchvision  |      v0.14.1     |
  |  TensorFlow   |      2.12.0      |
  |  OS           |   Ubuntu 20.04   |
  +----------------------------------+
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   

</div>

<div align="center">

~~~
CUDA Install and Configuration at Jetson Embedded Board (Jetson Orin AGX)
~~~
</div>

<div align="center">

   
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  +----------------------------------+
  |     Model     |      Version     |
  +===============+==================+
  |  GPU          |      Ampere      |
  |  Jetpack Ver. |      5.1.0       |
  |  CUDA cores   |      1792/2048   |
  |  CUDA         |      11.4.19     |
  |  cudnn        |      8.6.0       |
  |  torch        |      v2.0.0      |
  |  torchaudio   |      v0.13.1     |
  |  torchvision  |      v0.14.1     |
  |  TensorFlow   |      2.12.0      |
  |  OS           |   Ubuntu 20.04   |
  +----------------------------------+
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   
</div>

*Issue: The error message TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. 
-> Solution: suggests that you are trying to convert a PyTorch tensor that is on the GPU (CUDA) to a NumPy array directly. NumPy operates on CPU, so you'll need to move the tensor to CPU before converting it to a NumPy array.
```python
px_r = pandas.DataFrame(resbb_r).astype("float")  # all detected vehicles's list in px
px_r = pandas.DataFrame(resbb_r.cpu().numpy()).astype("float")  # all detected vehicles's list in px
```
### Evaluation

<figure>
    <p align="center">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/21_withoutCuda.gif?raw=true" width="400" height="225">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/20_withCuda.gif?raw=true" width="400" height="225">
    </p>
    <figcaption align="center"> left: Data Processing without CUDA / right: Data Processing with CUDA </figcaption>
</figure>

<br/>

## Simulation for Unity
Objective : It can be simulated using the game engine Unity.
One of the reasons for doing simulation is that it costs a lot to actually test on the road, but simulation can reduce this.
Therefore, various simulation scenarios can be configured.
Software for developed autonomous driving assistance system during the simulation will also be evaluated.
Additionally, implementation in virtual reality through simulation is one of the goals of this project.
Unity has good support for head-mounted devices(HMD) for virtual reality.
Thus it should be evaluated in simulations and attempt to interface with virtual reality HMD.
### Evaluation

<figure>
    <p align="center">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/loading.gif?raw=true" width="400" height="225">
    </p>
    <figcaption align="center"> Simulation in Unity Environment </figcaption>
</figure>

<br/>

## Simulation for ROS (Robot Operating System) (In Development)
Objective : Another method for simulation is ROS.
ROS stands for Robot Operating System and is specialized in robot simulation and control.
Therefore, it will be also proceeded with simulation and verification of the developed program in ROS.

### Evaluation

<figure>
    <p align="center">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/loading.gif?raw=true" width="400" height="225">
    </p>
    <figcaption align="center"> Simulation in ROS Environment </figcaption>
</figure>

<br/>

## Simulation for Real-time Data transfer
Objective : This is the finish line of this project.
We developed an autonomous driving assistant system and developed project was simulated in Unity and ROS.
A future project, remote robot control, is our final destination.
For this purpose, simulations are run in ROS for control aspects, and head-mounted devices are run in Unity for remote access and virtual reality implementation.
Remote access and virtual reality implementation using head-mounted devices will be postponed to the next project. In this project, YOLOV8 will be tested in a ROS environment and data communication between PCs and embedded systems will be implemented and evaluated.

```bash
Project Root
├── /control
│ ├── /launch
│  └── yolobot_control.launch.py
│ ├── /scripts
│  └── robot_control.py
│ ├── CMakeLists.txt
│ └── package.xml
├── /description
│ ├── /launch
│  └── spawn_yolobot.py
│  └── spawn_yolobot_launch.launch.py
│ ├── /robot
│  └── yolobot.urdf
│ ├── CMakeLists.txt
│ └── package.xml
├── /gazebo
│ ├── /launch
│  └── yolobot_launch.py
│  └── start_world_launch.py
│ ├── /world
│  └── yolo_test.world
│ ├── CMakeLists.txt
│ └── package.xml
├── recognition
│ ├── /launch
│  └── launch_yolov8.launch.py
│ ├── /scripts
│  └── yolov8n.pt
│  └── yolov8_ros2_pt.py
│  └── yolov8_ros2_subscriber.py
│  └── .dockerignore
│  └── .gitattributes
│  └── .gitignore
│ ├── CMakeLists.txt
│ └── package.xml
├── msg
│ ├── /msg
│  └── InferenceResult.msg
│  └── Yolov8Inference.msg
│ ├── CMakeLists.txt
│ └── package.xml
└── requirements.txt
```



### Evaluation

<figure>
    <p align="center">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/loading.gif?raw=true" width="400" height="225">
    </p>
    <figcaption align="center"> Simulation in ROS Environment </figcaption>
</figure>

<br/>

## Step 4. System Optimization - Overtime 
***
Goal : System optimization and testing on embedded platforms ensure that the system is suitable for real-world applications and can be deployed in actual vehicles.
## Jetson Embedded System
Objective : The compact size, power efficiency, ruggedness, real-time processing capabilities, AI and machine learning support, connectivity options, customization, and strong ecosystem make NVIDIA Jetson embedded boards an attractive option for industrial applications.
<br/>

## Final Evaluation
<figure>
    <p align="center">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/finalEval.gif?raw=true" width="400" height="225">
    </p>
    <figcaption align="center"> Final Evaluation </figcaption>
</figure>
<br/>


## Conclusion
In this project, it was started the work by focusing on the implementation of the overall driving assistance system and especially the implementation and optimization of embedded systems.
At the beginning of the project, object recognition models were compared, and vehicle recognition was implemented using YOLOv8, one of the most recent object recognition models as of 2023. In addition, lane recognition was implemented.

When it is started the project in earnest, we continuously tracked the vehicle and calculated the speed and distance of other vehicles. Based on this, we built an algorithmic model that determines how much each vehicle poses a risk to the user's vehicle (for example, calculating the possible collision time). Additionally, a driving assistance system called cruise control has been implemented to prevent drivers from leaving their lane. It is also implemented searching for each sideline on the right and left to calculate whether it is recommended when the user changes lanes. Lastly, we visually implemented a risk judgment model algorithm based on front and rear cameras, right and left sideline information, and status information (speed, distance) of other nearby vehicles.

It is implemented so-called Monitoring Aerial Perspective(M.A.P.) through a bird's eye view using the front and rear cameras. This is the same as the navigation and map apps we are familiar with, but ours was a surrounding map system based on real-time data using our own camera. Lastly, it is implemented a head-mounted system for manipulation in a virtual environment, a robot operating system(ROS), and an embedded system using CUDA, especially the Jetson series. Although system optimization between hardware is not yet 100% finished, successful testing of each module is meaningful and has the potential to be further developed through modularization in the next project.
<br/>

## Outlining Future Work
This project focuses on software, in particular, implementation of assistance for autonomous driving systems. Now, in the next project, we will use the implemented software program to implement the robot in hardware and apply various sensors, access and control the system using virtual reality that was not completed in this project, optimize data processing in the embedded system, and Tasks related to the collection and processing of various real-time information can be implemented.
<br/>


<div align="center">

~~~
-Fin.-
~~~
</div>

