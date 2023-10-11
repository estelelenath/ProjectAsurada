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
   - Danger Risk judgment Algorithm (Pop up Rear Camera screen)
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
   - Sensor Test
   - Final Evaluation
 
<br/>
<p align="center">Workflow Version 2.</p>
<br/>

<p align="center">
    <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/plan_008.png?raw=true" width="957" height="538"></center>
</p>

<br/>

Overall, ...

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
    <figcaption align="center"> Comparison of YOLO and Faster R-CNN left: YOLO / right: Faster R-CNN</figcaption>
    <figcaption align="center"> source) https://github.com/alen-smajic/Real-time-Object-Detection-for-Autonomous-Driving-using-Deep-Learning</figcaption>
</figure>

<br/>
 

<figure>
    <p align="center">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/yoloFasterRCnn.jpg?raw=true" width="400" height="225">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/yolo-comparison-plots.png?raw=true" width="400" height="225">
    </p>
    <figcaption align="center"> Comparison of YOLO and Faster R-CNN (mAP%: Mean Average Precision. FPS: Frames per second.) </figcaption>
    <figcaption align="center"> benchmark of YOLO performance by version) (https://github.com/ultralytics/ultralytics</figcaption>)
</figure>

<br/>

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
YOLO (https://github.com/alanzhichen/yolo8-ultralytics)
  +---------------------------------------------------------------------------------------+
  | Model   |   Size(pixels)    |   mAP^val 50-95  |   Speed CPU ONNX(ms)  |    FLOPs     |
  +=========+===================+==================+=======================+==============+
  | YOLO8n  |   640             |   37.3           |   80.4                |    8.7       |
  | YOLO8s  |   640             |   44.9           |   128.4               |    28.6      |
  | YOLO8m  |   640             |   50.2           |   234.7               |    78.9      |
  | YOLO8l  |   640             |   52.9           |   375.2               |    165.2     |
  | YOLO8x  |   640             |   53.9           |   479.1               |    257.8     |
  +---------------------------------------------------------------------------------------+
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the table above, you can consider mAP % as accuracy and FPS as processing speed.
As can be seen from the table, in terms of accuracy, Faster R-CNN has better values and YOLO processes data faster.
Therefore, algorithms such as YOLO and SSD, which focus on speed, is considered as front runners.
In addition, in the case of YOLO, it has been updated to the current (summer 2023) YOLOv8 model, and as a result of continuous upgrades, it is superior to other models in terms of accuracy and processing speed is faster, so we decided to use the YOLO model in this test.
<br/>

### Implementation
```
Python
Code
```
### Evaluation
<figure>
    <p align="center">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/loading.gif?raw=true" width="400" height="225">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/loading.gif?raw=true" width="400" height="225">
    </p>
    <figcaption align="center"> left: Video for front camera / right: Video for rear camera</figcaption>
</figure>


<br/>

## Lane Detection
Objective : Similar to previously implemented vehicle detection algorithm models, lane detection in the autonomous driving is implemented with particular emphasis on accuracy and fast processing speed. 
Likewise, a certain level of reliability must be secured in various weather environments and external factors.
### Algorithm

Workflow of Lane Detection
Preprocessing(distortion) -> Wrapping [Bird-Eye Effect] -> Filter -> Search the Lane-> Re-Wrapping 

Step 1: Distortion and Camera Calibration

current cheap camera makes a distortion of images,
main distortions are radial- / tangential - distortion.

The lane recognition algorithm can be divided into a traditional method and a method that utilizes Deep Learning. 
The traditional Hough Transform or Sobel Operator have heavy functions, lacks curve recognition, or is vulnerable to external change factors.
However, since the use of CUDA using a graphics card was strongly recommended in the lane recognition model using deep learning, the lane recognition model using histogram was implemented by itself as a basic method in this chapter by separating the white and yellow lanes with a filter, and later In the expanded function, we will implement an advanced lane recognition model based on deep learning.

First, the overall workflow of the basic method is as follows.
The first lane we see (perceived by the vehicle's camera) is affected by the perspective. Therefore, distortion and perspective are corrected through camera correction (camera calibration) and wrapping (perspective correction or bird's eye view).

<figure>
    <p align="center">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/loading.gif?raw=true" width="400" height="225">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/loading.gif?raw=true" width="400" height="225">
    </p>
    <figcaption align="center"> left: Original video input / right: Bird eye view </figcaption>
</figure>

The corrected image uses a color filter to detect white and yellow lanes, internationally accepted lane standards. In this case, HLS was used.

<figure>
    <p align="center">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/loading.gif?raw=true" width="400" height="225">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/loading.gif?raw=true" width="400" height="225">
    </p>
    <figcaption align="center"> left: Bird eye view / right: White/Yellow Lane Detection </figcaption>
</figure>

The detected image redefines the part to be observed through ROI once again.
The reason for this process is that the road infrastructure may not always be in the best condition (eg cracked asphalt roads, all things we are not interested).

<figure>
    <p align="center">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/loading.gif?raw=true" width="400" height="225">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/loading.gif?raw=true" width="400" height="225">
    </p>
    <figcaption align="center"> left: White/Yellow Lane Detection / right: ROI view </figcaption>
</figure>

The following is the process of checking whether the value we detected is suboptimal or not.
The detected lane should have a continuous value in the vertical value (height or y value), so it has a value close to 255 on the histogram classification. 
Conversely, if lanes are not detected, this value will be close to 0.

<figure>
    <p align="center">
        <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/loading.gif?raw=true" width="400" height="225">
    </p>
    <figcaption align="center"> Histogram detection value </figcaption>
</figure>

### Implementation
```
Python
Code
```
### Evaluation

<br/>

## Step 2. Advanced Features - First Half
***
Goal : 
## Vehicle Tracking 
Objective : 
### Algorithm
### Implementation
```
Python
Code
```
### Evaluation
<br/>

## Speed Estimation 
Objective : 
### Algorithm
### Implementation
```
Python
Code
```
### Evaluation
<br/>

## Distance Estimation 
Objective : 
### Algorithm
### Implementation
```
Python
Code
```
### Evaluation
<br/>

## Danger Risk Judgment 
Objective : 
### Algorithm
Pop up rear camera on the screen
### Implementation
```
Python
Code
```
### Evaluation
<br/>

## Steering and Speed Control for autonomous Driving 
Objective : 
### Algorithm
### Implementation
### Evaluation
<br/>

## Safe Lane Suggestion (In planning...) 
Objective : 
### Algorithm
### Implementation
### Evaluation
<br/>

## Auto Driving support system using two camera 
Objective : 
### Algorithm
### Implementation
### Evaluation
<br/>
<br/>

## Step 3. Extended Features in Simulation and Virtual Reality - Second Half 
***
Goal :
## Vehicle Identification Number and Traffic Signal recognition
Objective : 
### Algorithm
### Implementation
### Evaluation

<br/>

## Current traffic situation (Bird eye view) -> M.A.P(Monitoring Aerial Perspective)
Objective : 
### Algorithm
### Implementation
### Evaluation

<br/>


## Advanced Lane Detection based on Deep Learning
Objective : 
### Algorithm
### Implementation
### Evaluation

<br/>

## Hardware Development Environment for Virtual Reality
Objective : Visualisation / Real-time Data transfer (implemented Driving Assistance System)
### Algorithm
### Implementation
### Evaluation

<br/>

## Hardware Development Environment for CUDA
Objective : Visualisation / Real-time Data transfer (implemented Driving Assistance System)
### Algorithm
### Implementation
GeForce 4070 Ti
CUDA 11.8
cudnn 8.9.0
torch              2.0.1+cu118
torchaudio         2.0.2+cu118
torchvision        0.15.2+cu118

The error message TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. 
suggests that you are trying to convert a PyTorch tensor that is on the GPU (CUDA) to a NumPy array directly. 
NumPy operates on CPU, so you'll need to move the tensor to CPU before converting it to a NumPy array.

px_r = pandas.DataFrame(resbb_r).astype("float")  # all detected vehicles's list in px
px_r = pandas.DataFrame(resbb_r.cpu().numpy()).astype("float")  # all detected vehicles's list in px
### Evaluation

<br/>

## Simulation for Unity 
Objective : A.I
### Algorithm
### Implementation
### Evaluation

<br/>

## Simulation for ROS (Robot Operating System) 
Objective : A.I
### Algorithm
### Implementation
### Evaluation

<br/>

## Simulation for Real-time Data transfer (implemented Driving Assistance System)
Objective : 
### Algorithm
### Implementation
### Evaluation

<br/>

## Step 4. System Optimization - Overtime 
***
Goal : System optimization and testing on embedded platforms ensure that the system is suitable for real-world applications and can be deployed in actual vehicles.
## Jetson Embedded System
Objective :
### Algorithm
### Implementation
### Evaluation
<br/>

## Sensor Test
Objective : 
### Algorithm
### Implementation
### Evaluation
<br/>

## Final Evaluation

<br/>


## Conclusion

<br/>

## Outlining Future Work

<br/>

-Fin.-
