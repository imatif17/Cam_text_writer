import numpy as np
import keras
import os
from os import listdir
import cv2
from keras.models import load_model
import matplotlib.pyplot as plt

def image_segmentation(image_name):
	# reading the image
	image = cv2.imread(image_name)

	# converting the image to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.erode(gray,kernel = np.ones((5,5),np.uint8))
	    #plt.imshow(gray,cmap = 'gray')

	    # threshold to convert the image to pure black and white
	thresh = cv2.threshold(gray, 0,255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	    #plt.imshow(thresh,cmap = 'gray')
	    
	    # find the contours (continous blob of pixels ) in the image 
	contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	    
	    # Hack for compatibility with different OpenCV versions
	contours = contours[0]

	letter_image_regions = []

	    # now loop through each of the letter in the image 
	for contour in contours:
		# get the rectangle that contains the contour
		x,y,w,h = cv2.boundingRect(contour)
		# compare the width and height of the contour to detect if it
		# has one letter or not
		#if w/h >1.20:
		    # this is too wide for a single letter
		    #continue
		if w<20 or h<30:
		    # this is a very small image probably a noise
		    continue
		else:
		# this is a normal letter by itself
		    letter_image_regions.append((x,y,w,h))
	    #plt.imshow(gray,cmap = 'gray')
	return letter_image_regions

def save_characters():
	# now we will read images from the folder segment the images and will produce the output
	for image_name in listdir('../../temp_data/images/'):
		counter = 1
	    # constructing the name of the file 
		file_name = '../../temp_data/images/' + image_name

	    # getting segmented images 
		letters_in_image = image_segmentation(file_name)
	    
	    # sorting the letters so that letters that appear before is addressed first 
		letters_in_image = sorted(letters_in_image, key=lambda x: x[0])
	    
		ans = ""
		for (x,y,w,h) in letters_in_image:
			image = cv2.imread(file_name,0)
			letter = image[y - 3:y + h + 3, x - 3:x + w + 3]
		
			cv2.imwrite('../../temp_data/alphabets/'+str(counter)+'.jpg', letter)
			counter = counter + 1

def text_extract():
	letters = { 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j',
	11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't',
	21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z'}
	model = load_model('../../Model/cnn_model.h5')
	word = ''
	kernel = np.ones((3, 3), np.uint8)
	for i,item in enumerate(sorted(os.listdir('../../temp_data/alphabets/'))):
		#plt.figure(figsize=(8,8))
		#ax = plt.subplot(1, len(os.listdir('../temp_data/alphabets/')), i+1)
		img = cv2.imread('../../temp_data/alphabets/'+item)
		img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		img = cv2.bitwise_not(img)

		retval, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

		h,w = img.shape
		img = np.pad(img, pad_width=100, mode='constant', constant_values=0)
		img = cv2.dilate(img,kernel,iterations = 2)
		if(h>w):
			img = img[50:150+h,50:150+h]
		else:
			img = img[50:150+w,50:150+w]
		img = cv2.dilate(img, kernel, iterations=2)
		img = cv2.resize(img,(28,28))
		img = img.astype('float32')/255

		prediction = model.predict(img.reshape(1,28,28,1))[0]
		prediction = np.argmax(prediction)
		word+=str(letters[int(prediction)+1])
	print(word)
