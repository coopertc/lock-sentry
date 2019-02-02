import numpy as np
import cv2
import sys

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

img_path = sys.argv[1]
label = sys.argv[2]

# read image from file path given as first command line argument


frame = cv2.imread(img_path, 1)
cv2.imshow("raw", frame)
cv2.waitKey(0)


faces = faceCascade.detectMultiScale(
    frame,
    scaleFactor=1.5,
    minNeighbors=5,
    minSize=(10,10),
    flags=cv2.cv.CV_HAAR_SCALE_IMAGE
)

for(x,y,w,h) in faces:
    print 'face found'
    crop_img = frame[y:y+h, x:x+w]
    resize_img = cv2.resize(crop_img, (128,128))
    cv2.imshow('processed frame', resize_img)
    cv2.waitKey(0)
