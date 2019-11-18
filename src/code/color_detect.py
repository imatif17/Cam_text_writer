import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from ocr import image_segmentation, save_characters, text_extract
from keras.models import load_model
from collections import deque
import numpy as np
import timeit
import os
import cv2
import keras
os.system('clear')


timer = 0
flag = 0
window_name = 'frame'
print('Loading the model')
model = load_model('../../Model/gesture_classifier.h5')
os.system('clear')
print('Model Loaded')


lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255,255])
lower_blue = np.array([90,50,100])
upper_blue = np.array([120,255,255])
lower_red = np.array([190,120,7])
upper_red = np.array([255, 240, 25])
kernel = np.ones((5, 5), np.uint8)
blank_paper = np.zeros((800,1400,3), dtype=np.uint8)
blank_paper = cv2.cvtColor(blank_paper,cv2.COLOR_BGR2RGB)
blank_paper[:,:,:] = 255
points = deque(maxlen = 5000)

def mapper(x,y):
    if (x == 0 and y == 0):
        return "Put your hand in the box"
    elif (x == 0 and y != 0):
        return "Exit Mode"
    elif (x == 1):
        return "Waiting Mode"
    return "Writing Mode"

colors = np.array([[lower_red,upper_red],[lower_yellow,upper_yellow],[lower_blue,upper_blue]])
color_name = ['red','yellow','blue']

while (True):
    print("Choose the color :\n\t1 - Red\n\t2 - Yellow\n\t3 - Blue")
    c = int(input())
    if c not in [1,2,3]:
        print("Wrong input try again")
        continue
    break

cam = cv2.VideoCapture(0)
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
while(True):
    ret,frame = cam.read()
    if(ret):
        frame = cv2.flip(frame,1)
        frame = cv2.resize(frame,(1300,720))
        y1,y2,x1,x2 = 710, 250 , 450, 20
        roi = frame[y2:y1, x2:x1]
        frame = cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,255),3)
        roi = cv2.resize(roi,(224,224))
        roi = roi.astype('float32')
        roi = roi/255.0
        flag = model.predict(roi.reshape(1,224,224,3))
        flag = np.argmax(flag)
        
        
        cv2.putText(frame,mapper(flag,len(points)),(0,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3, cv2.LINE_AA)
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        hsv = cv2.dilate(hsv,np.ones((7,7)))
        mask = cv2.inRange(hsv,colors[c-1][0],colors[c-1][1])
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key = cv2.contourArea)
            x,y,w,h = cv2.boundingRect(largest_contour)
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),3)
            frame = cv2.drawContours(frame, largest_contour, -1, (255,255,0), 3)
            if (flag == 2):
                centre = (int(x + w/2), int(y + h/2))
                points.appendleft(centre)

        if (flag == 1 and len(points) != 0):
            temp = points.popleft()
            points.appendleft(temp)
            if (temp != (-1,-1)):
                points.appendleft((-1,-1))       

        for i in range(1, len(points)):
            if ((points[i - 1] is not None or points[i] is not None) and (points[i - 1] != (-1,-1) and points[i] != (-1,-1))):
                cv2.line(frame, points[i - 1], points[i], (0, 0, 0), 6)
                cv2.line(blank_paper, points[i - 1], points[i], (0, 0, 0), 8)


        cv2.imshow(window_name,frame)

        key = cv2.waitKey(1)
        
        if key == ord('q'):
            break
        
        if (flag == 0 and timer == 0 and len(points) != 0):
            start = timeit.default_timer()
            timer = 1
            
        if (flag != 0 and timer == 1):
            timer = 0
        
        if (flag == 0 and len(points) != 0 and timer == 1 and ((timeit.default_timer() - start) > 2)):
            blank_paper = cv2.cvtColor(blank_paper, cv2.COLOR_RGB2GRAY)
            cv2.imwrite('../../temp_data/images/new.jpg',blank_paper)
            break

    else:
        blank_paper = cv2.cvtColor(blank_paper, cv2.COLOR_RGB2GRAY)
        cv2.imwrite('../../temp_data/images/new.jpg',blank_paper)
        break
        
cam.release()
cv2.destroyAllWindows()

for i in os.listdir('../../temp_data/alphabets/'):
	os.remove('../../temp_data/alphabets/'+i)
save_characters()
text_extract()
