import cv2
import numpy as np
import time

# Load Yolo
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
# weights, cfg파일을 불러와서 yolo의 네트워크와 연결한다.
# yolov3.weights / cfg : yolov3의 훈련된 가중치를 저장하고 있는 이진 파일 / yolov3의 네트워크 구성을 저장하고 있는 텍스트 파일
classes = []
 #class 배열 만들기
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
# 읽어온 coco 파일을 whitespace(공백라인)를 제거하여 classes 배열 안에 넣는다.
# strip() : whitespace(띄워쓰기, 탭, 엔터)를 없애는 것, 중간에 끼어있는 것은 없어지지 않는다.
#layer_names = net.getLayerNames()
# 네트워크의 모든 레이어 이름을 가져와서 layer_names에 넣는다.
# 네트워크의 모든 레이어 이름을 가져온다. YOLOv3에는 3개의 출력 레이어(82.94.106)가 있다.
#output_layers = [layer_names[i[0]] for i in net.getUnconnectedOutLayers()]
output_layers = net.getUnconnectedOutLayersNames()
# 레이어 중 출력 레이어의 인덱스를 가져와서 output_layers에 넣는다.
# 출력 레이어를 가져온다
colors = np.random.uniform(0, 255, size=(len(classes), 3))
# 클래스의 갯수만큼 랜덤으로 BRG 배열을 생성한다. 한 사물 당 하나의 color만 사용할 수 있도록 해서 구분해야한다.
# random 모듈 안에 정의 되어 있는 두 수 사이의 랜덤 한 소수를 리턴 시켜주는 함수이다.


# Loading image(image)
# img = cv2.imread("trafficsignal.jpg")
# opencv를 통해 이미지를 가져온다.
# img = cv2.resize(img, None, fx=1, fy=1)
# 이미지 크기를 재설정한다.
# height, width, channels = img.shape
# 이미지의 속성들을 넣는다.

# Loading image(webcam)
cap_front = cv2.VideoCapture(0)
# Loading image(Video)
# cap_front = cv2.VideoCapture(r"C:\Users\tushi\Downloads\PythonGeeks\clouds.mp4")

if not cap_front.isOpened():
    print("Cannot open camera")
    exit()


while True:
    ret_front, frame_front = cap_front.read()
    img = cv2.resize(frame_front, None, fx=1, fy=1)
    height, width, channels = img.shape
    
    if ret_front == True:
        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        # 이미지를 blob 객체로 처리한다.
        # 입력 input
        # scalefactor : 입력 영상 픽셀 값에 곱할 값, 기본 값은 1 (0~1사이에 정규화해야함)
        # size : 출력 영상의 크기, 기본값은 (0.0) 여기엔 (416.416)
        # mean : 입력 영상 각 채널에서 뺼 평균값, default 는 (0.0.0)
        # swapRM : R과 B채널을 서로 바꿀 것인지 결정하는 플래그, 기본값은 False
        # crop : 크롭 수행 여부, 기본값은 false
        net.setInput(blob)
        # blob 객체에 setInput 함수를 적용한다.
        outs = net.forward(output_layers)
        #output_layers를 네트워크 순방향으로실행(추론)한다.


        # Showing informations on the screen
        # 물체의 범위를 박스 형태로 정확하게 잡아주고 그 물체가 무엇인지 labeling해주어야 한다.
        # labeling하는 기준은 신뢰도가 0.5를 넘으면 물체라고 인식을한다

        class_ids = []
        # 인식한 사물 클래스 아이디를 넣는 배열
        confidences = []
        # 0에서 1까지 사물 인식에 대한 신뢰도를 넣는 배열
        boxes = []
        # 사물을 인식해서 그릴 상자에 대한 배열
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                # scores 중에서 최대값을 색인하여 class_id에 넣는다.

                confidence = scores[class_id]
                # scores 중에서 class_id에 해당하는 값을 confidence에 넣는다.

                if confidence > 0.5:
                    # 만약 정확도가 0.5가 넘는다면 사물이 인식되었다고 판단
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # noise delete, 같은 사물에 대해서 박스가 여러 개인 것을 제거하는 Non MaximumSuppresion 작업을 한다
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


        font = cv2.FONT_HERSHEY_PLAIN
        # Font종류 중 하나인 FONT_HERSHEY_PLAIN(작은 크기 산세리프 폰트)를 적용한다.

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                # 클래스 아이디 지정해둔 것을label변수에 저장
                color = colors[i]
                #위에서 colors배열에 색상을 넣어둔 것을 color에 저장
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                # 사각형을그린다.
                # 입력 파일
                # 시작점 좌표
                # 종료점 좌표
                # 색상
                # 선두께 / 값을 -1로 하면 내부가 색칠된 사각형을 얻을수 있다.
                cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
                # yolo에서 학습된 사물의 명칭을 출력한다.
                # 이미지 파일
                # 출력할 텍스트 -> yolo 에서 미리 학습된 사물의 이름을 label변수에 넣어뒀는데 그것을 출력
                # 텍스트 이미지의 좌하단 꼭짓점
                # 폰트 스타일 지정 
                # 폰트 크기
                # bgr값으로 색상을 채움, colors배열에 저장했던 bgr값을 color에 다시 저장해뒀는데 그것을 색상으로 사용
                # 폰트 굵기


        cv2.imshow("Image", img)


        if cv2.waitKey(1) == ord('q'):
            break
        # waitKey() : 0 -> 키 입력이 있을 때까지 무한 대기한다. /ms 단위로 입력하면 그 단위에 따라 대기한다.

cap_front.release()
cv2.destroyAllWindows()