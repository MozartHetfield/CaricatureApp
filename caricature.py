import cv2
import numpy as np
import sys
from PIL import Image
from imutils import face_utils
import argparse
import imutils
import random

def caricature(filename, outputFilename):
	# Get user supplied values
	faceCascPath = "haarcascade_frontalface_default.xml"
	eyeCascPath = "haarcascade_eye.xml"
	
	# Create the haar cascade
	faceCascade = cv2.CascadeClassifier(faceCascPath)
	eyeCascade = cv2.CascadeClassifier(eyeCascPath)
	# Read the image
	image = cv2.imread(filename)
	im = Image.open(filename)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	# Detect faces in the image
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30),
		flags = cv2.CASCADE_SCALE_IMAGE
	)
	# Draw a rectangle around the faces
	for (x, y, w, h) in faces:
		cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = image[y:y+h, x:x+w]
		eyes = eyeCascade.detectMultiScale(gray, 1.3, 5)

		for (ex,ey,ew,eh) in eyes:
			im2 = im.crop((ex, ey, ex+ew, ey+eh))
			width, height = im2.size
			weight_value = random.uniform(1.25, 1.45)
			print("Random weight is " + str(weight_value))
			im2 = im2.resize((round(width * weight_value), round(height * weight_value)))
			im.paste(im2, (ex, ey))
			im.save(outputFilename)
			cv2.rectangle(image, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)