# <p align="center">ProjectAsurada</p>

### <p align="center">Driving Assistence System supported by Jetson</p>
<br/>

<p align="center">
    <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/title_001.png?raw=true" width="490" height="490"></center>
</p>

<br/>
<br/>
<br/>
<br/>

## The goals / steps of this project are the following:
 - Step 1: Identify the Optimal Algorithms for Vehicle and Lane Detection in Autonomous Driving
 - Step 2: Integrate Vehicle and Lane Detection in Live Streaming and Update to the Latest 2023 Version
 - Step 3: Enhance Driving Assistance Systems with Advanced Features and Functionality
 - Step 4: Provide Enhanced Driving Assistance Experiences through Real-Time Data Accumulation and Augmented Reality, and Outline Our Future Work
 
<br/>
<p align="center">Workflow Version 0.</p>
<br/>

<p align="center">
    <img src="https://github.com/estelelenath/ProjectAsurada/blob/main/pic/plan_003.jpg?raw=true" width="432" height="741"></center>
</p>

<br/>

## Basic Functions
### Algorithm
<Conduct a literature review of existing algorithms for vehicle and lane detection in autonomous driving.>
<Analyze the strengths and weaknesses of each algorithm to the latest 2023 version>
Vehicle recognition is a task in the field of computer vision that involves detecting and identifying vehicles in digital images or videos. 
Here are some of the most frequently used algorithms for vehicle recognition and their pros and cons:

 - Haar Cascade Classifier: This is a machine learning-based object detection method that uses Haar features and a cascade of simple classifiers to detect objects in images. Pros: It is computationally efficient, easy to implement, and can be trained on a variety of vehicle images. Cons: It may not perform well on small or partially occluded vehicles, and it may require a large number of positive and negative samples for training.
 - HOG (Histogram of Oriented Gradients): This is a feature descriptor that characterizes the shape and structure of an object by computing the gradient orientation histograms in a dense grid of cells. Pros: It is computationally efficient, can handle partial occlusions, and is relatively robust to viewpoint changes. Cons: It may not be able to handle significant shape variations, and it may not perform well on highly reflective or textured vehicles.
 - YOLO (You Only Look Once): This is a real-time object detection system that uses a single convolutional neural network to perform both object detection and classification. Pros: It is computationally efficient and can perform well on a variety of object categories, including vehicles. Cons: It may not be able to handle partial occlusions or fine-grained object variations.
 - Faster R-CNN: This is a region-based object detection system that uses a convolutional neural network to generate region proposals and perform object classification. Pros: It can handle partial occlusions and fine-grained object variations, and it can achieve high detection accuracy. Cons: It may not be computationally efficient for real-time applications, and it may require a large amount of labeled data for training.
In conclusion, the choice of algorithm for vehicle recognition will depend on the specific requirements of the task, including the processing speed, accuracy, and the complexity of the environment.


Specially the choice between Histogram of Oriented Gradients (HOG) and You Only Look Once (YOLO) for real-time vehicle recognition and autonomous driving largely depends on the specific requirements and constraints of the task.
HOG is a feature-based object detection method that is computationally efficient and can handle partial occlusions, making it well-suited for real-time applications. It has been widely used for object detection tasks, including vehicle recognition, and has shown good results in terms of accuracy and processing speed.
On the other hand, YOLO is a real-time object detection system that uses a single convolutional neural network (CNN) to perform object detection and classification. YOLO has the advantage of being able to process images in real-time and can handle a large number of object categories. However, compared to HOG, YOLO may not be as computationally efficient and may require more resources to achieve the same level of accuracy.
In the field of autonomous driving, both HOG and YOLO have been used for vehicle recognition, but YOLO is more commonly used due to its ability to handle a wide range of object categories, including vehicles. However, the choice of algorithm will also depend on the specific requirements and constraints of the task, such as the processing speed, accuracy, and the complexity of the driving environment.

<br/>

### Vehicle Detection
<Develop a programming framework to integrate vehicle and lane detection into live streaming.>
<Test the system's performance under various real-world conditions.>
<br/>

### Lane Detection
<br/>

## Extended Functions
### Safe distance assessment through the speed estimation and thus driving support
<Enhance the functionality of the system by integrating GPS data to provide real-time traffic updates and rerouting suggestions.>
<br/>

### Chat GPT API (User-friendly environment settings)
<br/>

### Navigation and visualisation (Route optimization) 
<Enhance the functionality of the system by integrating GPS data to provide real-time traffic updates and rerouting suggestions.>
<br/>

### Real-time data accumulation and self-learning
<Enhance the functionality of the system by integrating GPS data to provide real-time traffic updates and rerouting suggestions.>
<br/>

### AR Enviroment
<br/>

## Conclusion
<br/>

## Outlining Future Work