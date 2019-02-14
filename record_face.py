import numpy as np
import cv2
import csv
import sys
import os

cap = cv2.VideoCapture(1)
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

label = str(sys.argv[1])

directory_path = "./faces/" + label + "/"

if not os.path.exists(directory_path):
    os.makedirs(directory_path)

i = 0
while(i < 15):
    ret, frame = cap.read()
    #frame = cv2.imread('steve.jpg', 1);
    cv2.imshow('frame', frame)
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(
        frame,
        scaleFactor=1.5,
	minNeighbors=5,
	minSize=(60,60),
	flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    print len(faces)
    for (x,y,w,h) in faces:
         cv2.imshow('face', frame[y:y+h, x:x+h])
         cv2.waitKey()
    
    (x,y,w,h) = faces[0]
    crop_img = frame[y:y+h, x:x+w]

    resize_img = cv2.resize(crop_img, (200,300))

    imgPath = directory_path + str(i) + ".jpg"
    print imgPath
    i += 1
    cv2.imwrite(imgPath, resize_img)
    cv2.waitKey(1)
    
