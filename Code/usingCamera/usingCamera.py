#It's simple mode for object recognition for web-cam 
#based on the jetson-inference [https://github.com/dusty-nv/jetson-inference]
#based on the Murtaza Hassan [https://github.com/murtazahassan]
import jetson.inference
import jetson.utils
import cv2

class mnSSD():
		def __init__(self, path, threshold) :
			self.path = path
			self.threshold = threshold
			self.net = jetson.inference.detectNet(self.path, self.threshold)
		
		def detect(self, img, display = False):
			imgCuda = jetson.utils.cudaFromNumpy(img)
			detections = self.net.Detect(imgCuda, overlay = "OVERLAY_NONE")

			objects=[]
			for d in detections:
				className = self.net.GetClassDesc(d.ClassID)
				objects.append([className,d])
				
				if display:
					x1,y1,x2,y2 = int(d.Left),int(d.Top),int(d.Right),int(d.Bottom)
					cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
					cv2.rectangle(img,(x1,y1),((x1+(x2-x1)),y1+15),(220,220,220),-1)
					cv2.putText(img,className,(x1+5,y1+14),cv2.FONT_HERSHEY_DUPLEX,0.75,(255,255,0),2)
					cv2.putText(img,f'FPS: {int(self.net.GetNetworkFPS())}',(30,30),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,0),2)

			return objects

def main() : 
	#Pre-Trained Models
	#net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)	# jetson-inference/data/networks/...
	#Input Strems setup 
	cap = cv2.VideoCapture(0)
	cap.set(3,640)
	cap.set(4,480)
	myModel = mnSSD("ssd-mobilenet-v2", 0.5)
	while True:
		success, img = cap.read()
		objects = myModel.detect(img, True)

		cv2.imshow("Image", img)
		cv2.waitKey(1)


if __name__ == "__main__":
    	main()
