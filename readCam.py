import numpy as np
import cv2
import csv
import subprocess
import time
import requests
import base64
import json
import os
import dlib

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

gpio_base_dir = "/home/nvidia/gpio/"
gpio_init = gpio_base_dir + "initGpio.sh"
gpio_set_high = gpio_base_dir + "setGpioHigh.sh"
gpio_set_low = gpio_base_dir + "setGpioLow.sh"

image_size = 100

images_train = []
labels_train = []

images_test = []
labels_test = []

rc = subprocess.call(gpio_init)

def write_faces():
    r = requests.post("https://bd904d47.ngrok.io/api/update", data={'email': 'maxkaran2@gmail.com', 'password': '11'})
    if not r.ok:
        return
    users_json = json.loads(r.content)

    for i in users_json:
        dir_path = './faces/' + str(i['fid'])
        if not os.path.exists(dir_path):
	    os.makedirs(dir_path)
	count = 1
        
        for face in i['faces']:
            frame = base64.b64decode(face)
	    with open (dir_path + "/" + str(count) + ".jpg", 'wb') as f:
                f.write(frame)
            count += 1

def process_server_faces():
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    os.system('python create_csv.py ./faces')

    labels_train = []
    images_train = []
    predictor_path = "/home/nvidia/Documents/shape_predictor_5_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)
    

    with open('face_csv.txt', 'rb') as csvfile:
	    fileReader = csv.reader(csvfile, delimiter=";")
	    for row in fileReader:
		img = dlib.load_rgb_image(row[0])
		dets = detector(img, 1)
		faces = dlib.full_object_detections()
		if len(dets) < 1:
		    print "no face"
		    images_train.append(cv2.imread(row[0], 0))
		    labels_train.append(int(row[1][8:]))
		    continue
		else:
		    print "face"
		faces.append(sp(img, dets[0]))
		image = dlib.get_face_chip(img, faces[0], size=image_size)
		image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		normImage = np.zeros((image_size, image_size))
		normImage = cv2.normalize(image, normImage, 0, 255, cv2.NORM_MINMAX)	
		blur = cv2.GaussianBlur(normImage,(5,5),0)	
		cv2.imwrite(row[0], blur)
		img = cv2.imread(row[0], 0)
		print img.shape
		images_train.append(cv2.imread(row[0], 0))
		labels_train.append(int(row[1][8:]))
		'''
		
                faces = faceCascade.detectMultiScale(
			frame,
			scaleFactor=1.5,
			minNeighbors=5,
			minSize=(60,60),
			flags=cv2.cv.CV_HAAR_SCALE_IMAGE
		)
		
		if (len(faces) > 0):
		    (x,y,w,h) = faces[0]
		    crop_img = frame[y:y+h, x:x+w]
		    resize_img = cv2.resize(crop_img, (92,112))
		    cv2.imwrite(row[0], resize_img)
	            images_train.append(cv2.imread(row[0], 0))
		    labels_train.append(int(row[1][8:]))
		'''
		    
    labels_train = np.array(labels_train)
    return {'labels_train': labels_train, 'images_train': images_train}


write_faces()
proc_faces = process_server_faces()

images_train = proc_faces['images_train']
labels_train = proc_faces['labels_train']

model = cv2.createEigenFaceRecognizer(60)
model.train(images_train, labels_train)

print "done training"
i = 0
while(True):
    i += 1
    cap = cv2.VideoCapture(1)
    ret, frame = cap.read()
    cap.release()
    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("./current/current"+ str(i) + ".jpg", img)    
    
    
    predictor_path = "/home/nvidia/Documents/shape_predictor_5_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)
    
    img = dlib.load_rgb_image("./current/current" + str(i) + ".jpg")
    dets = detector(img, 1)
    if len(dets) < 1:
        print "no face"
        continue

    faces = dlib.full_object_detections()
    faces.append(sp(img, dets[0]))
    image = dlib.get_face_chip(img, faces[0], size=image_size)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    normImage = np.zeros((image_size, image_size))
    normImage = cv2.normalize(image, normImage, 0, 255, cv2.NORM_MINMAX) 
    cv2.imwrite("./current/current" + str(i) + ".jpg", normImage)
    #cv2.waitKey()

    pred, dist = model.predict(normImage)
    print "pred: ", pred, " distance: ", dist, "id: ", i
    if dist < 3000:
        print "unlock door"
	time.sleep(3)
'''
    faces = faceCascade.detectMultiScale(
        frame,
        scaleFactor=1.5,
	minNeighbors=5,
	minSize=(60,60),
	flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    print "faces detected: ", len(faces)
    for(x,y,w,h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resize_img = cv2.resize(crop_img, (200,200))
	
	pred, dist = model.predict(resize_img)
	print "pred: ", pred, " distance: ", dist
	cv2.imshow("max", resize_img)
        
	
    	if(cv2.waitKey(1) == 27):
	    break
	

	if(dist < 2250):
	    print "unlock door"
	    rc = subprocess.call(gpio_set_high)
	    time.sleep(3)
	    print "lock door"
            rc = subprocess.call(gpio_set_low)
'''


