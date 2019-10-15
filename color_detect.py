import numpy as np
import sys
from collections import deque
#import argparse
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255,255])
lower_blue = np.array([90,50,100])
upper_blue = np.array([120,255,255])
lower_red = np.array([190,120,7])
upper_red = np.array([255, 240, 25])
kernel = np.ones((5, 5), np.uint8)
blank_paper = np.zeros((480,640,3), dtype=np.uint8)
blank_paper = cv2.cvtColor(blank_paper,cv2.COLOR_BGR2RGB)
points = deque(maxlen = 5000)

colors = np.array([[lower_red,upper_red],[lower_yellow,upper_yellow],[lower_blue,upper_blue]])
color_name = ['red','yellow','blue']

while (True):
	print("Choose the color :\n\t1 - Red\n\t2 - Yellow\n\t3 - Blue")
	c = int(input())
	if c not in [1,2,3]:
		print("Wrong input try again")
		continue
	break
flag = 1
cam = cv2.VideoCapture(0)
while(True):
	ret,frame = cam.read()
	if(ret):
		print(flag)
		frame = cv2.flip(frame,1)
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
			if (flag == 1):
				centre = (int(x + w/2), int(y + h/2))
				points.appendleft(centre)

		else:
			if(len(points) != 0):
				pass

		for i in range(1, len(points)):
			if (points[i - 1] is  not None or points[i] is not None):
				cv2.line(frame, points[i - 1], points[i], (0, 0, 0), 2)
				cv2.line(blank_paper, points[i - 1], points[i], (255, 255, 255), 8)
				

		cv2.imshow(color_name[c-1]+' colour detection',frame)
		#cv2.imshow(color_name[c-1]+' colour detection',blank_paper)
		
		key = cv2.waitKey(1)
		
		if key == ord('q'):
			blank_paper = cv2.cvtColor(blank_paper, cv2.COLOR_RGB2GRAY)
			cv2.imwrite('new.jpg',blank_paper)
			break
		
		elif key == ord('w'):
			flag = 0
			key = ord('a')

		elif key == ord('z'):
			flag = 1
			key = ord('a')
	else:
		blank_paper = cv2.cvtColor(blank_paper, cv2.COLOR_RGB2GRAY)
		cv2.imwrite('new.jpg',blank_paper)
		break
        
cam.release()
cv2.destroyAllWindows()
