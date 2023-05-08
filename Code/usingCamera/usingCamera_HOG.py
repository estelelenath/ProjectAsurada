import cv2
#import argparse                                                             # webcam Resolution

from ultralytics import YOLO                                                # YOLOv8 Model

# Load the pre-trained HOG classifier
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Initialize the webcam
cap_front = cv2.VideoCapture(0)
cap_rear = cv2.VideoCapture(1)

# ML Model #YOLO v8
#model = YOLO("yolov8l.pt")

while True:
    # Read a frame from the webcam
    ret_front, frame_front = cap_front.read()
    ret_rear, frame_rear = cap_rear.read()

    # Resize the frame to a smaller size to speed up the detection process
    frame_front = cv2.resize(frame_front, (640, 480))
    frame_rear = cv2.resize(frame_rear, (640, 480))

    # Detect vehicles in the frame using HOG algorithm
    (rects, weights) = hog.detectMultiScale(frame_front, winStride=(4, 4),
        padding=(8, 8), scale=1.05)
    
    (rects, weights) = hog.detectMultiScale(frame_rear, winStride=(4, 4),
        padding=(8, 8), scale=1.05)
    

    # Draw bounding boxes around the detected vehicles
    for (x, y, w, h) in rects:
        cv2.rectangle(frame_front, (x, y), (x+w, y+h), (0, 0, 255), 2)

    for (x, y, w, h) in rects:
        cv2.rectangle(frame_rear, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Show the output frame
    cv2.imshow("Vehicle Detection_front", frame_front)
    cv2.imshow("Vehicle Detection_rear", frame_rear)

    # Press "q" to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap_front.release()
cap_rear.release()
cv2.destroyAllWindows()