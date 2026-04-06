from ultralytics import YOLO
from cachetools import TTLCache
import RPi.GPIO as GPIO
import time
import pyttsx3
import cv2
import math





#Function to speak object name
def speak_object(object_name, engine, cache):
	item = cache.get(object_name, None)
	if item is None:
		print ('Item %s not in Cache so we are going to speak it' %object_name)
		cache[object_name] = object_name
		
		sayit = 'Found %s object for now' %object_name
		engine.say(sayit)
		engine.runAndWait()
		
	else: 
		print ('Item %s is in Cache, we are going to skip saying it' %item)
		
	


def logInfo(line):
	print(line)


# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 720)
cap.set(4, 480)




#create text-to-speech engine
engine = pyttsx3.init()

#initiate YOLO model
model = YOLO("yolo-Weights/yolov8n.pt")


# get classes from model
class_list = model.model.names

## Initialize cache with 10 seconds time to live (TTL)
cache = TTLCache(maxsize=1000, ttl=10)



## Initialize GPIO for motion detection
GPIO.setmode(GPIO.BCM)
## GPIO 23 value in Raspberry pi 5 is 11
GPIO.setup(11, GPIO.IN) 

## loop thru till "q" is pressed
while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)
            
            # class name
            cls = int(box.cls[0])
            print("Class name from ClassList -->" , class_list[cls])
            
            if confidence >= 0.5:
                print("Greater Confidence --> %s Class Name --> %s" %(confidence , class_list[cls]))
                speak_object(class_list[cls], engine, cache)
                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, class_list[cls], org, font, fontScale, color, thickness)
            elif GPIO.input(11): #If there is a movement, PIR sensor gives input to GPIO 23
                logInfo ("There was movement")
                speak_object(" an " , engine, cache)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
engine.stop()

