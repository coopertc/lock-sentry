import numpy as np
import cv2
import csv
import sys
import os
import dlib
i = 0

while(i < 5):
	cap = cv2.VideoCapture(1)
	ret, frame = cap.read()
	cap.release()
	#cv2.imshow("image", frame)
	#cv2.waitKey()
	cv2.imwrite("./image.jpg", frame)


	predictor_path = "/home/nvidia/Documents/shape_predictor_5_face_landmarks.dat"


	detector = dlib.get_frontal_face_detector()
	sp = dlib.shape_predictor(predictor_path)

	img = dlib.load_rgb_image("./image.jpg")
	#window = dlib.image_window()
	#window.set_image(img)
	#dlib.hit_enter_to_continue()

	dets = detector(img, 1)


	num_faces = len(dets)
	if num_faces == 0:
	    print("Sorry, there were no faces found in '{}'".format(face_file_path))
	    #exit()

	faces = dlib.full_object_detections()
	for detection in dets:
	    faces.append(sp(img, detection))

	#window = dlib.image_window()

	images = dlib.get_face_chips(img, faces, size=320)
	for image in images:
	    #window.set_image(image)
	    #dlib.hit_enter_to_continue()
	    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	    cv2.imwrite("./image" + str(i) + ".jpg", image)
	    
	i += 1

