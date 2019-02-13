import numpy as np
import cv2
import csv
import subprocess
import time
import requests
import base64
import json
import os

def write_faces():
    r = requests.post("http://09846fe9.ngrok.io/api/update", data={'email': 'maxkaran2@gmail.com', 'password': '11'})
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

    with open('face_csv.txt', 'rb') as csvfile:
	    fileReader = csv.reader(csvfile, delimiter=";")
	    for row in fileReader:
		#images_train.append(cv2.imread(row[0], 0))
		#labels_train.append(int(row[1]))
		frame = cv2.imread(row[0], 0)

                faces = faceCascade.detectMultiScale(
			frame,
			scaleFactor=1.5,
			minNeighbors=5,
			minSize=(60,60),
			flags=cv2.cv.CV_HAAR_SCALE_IMAGE
		)
		
		for(x,y,w,h) in faces:
		    crop_img = frame[y:y+h, x:x+w]
		    resize_img = cv2.resize(crop_img, (92,112))
		if len(faces) > 0:
		    cv2.imwrite(row[0], resize_img)
		    images_train.append(cv2.imread(row[0], 0))
		    labels_train.append(int(row[1][8:]))
		    
    labels_train = np.array(labels_train)
    return {'labels_train': labels_train, 'images_train': images_train}
   




cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

gpio_base_dir = "/home/nvidia/gpio/"
gpio_init = gpio_base_dir + "initGpio.sh"
gpio_set_high = gpio_base_dir + "setGpioHigh.sh"
gpio_set_low = gpio_base_dir + "setGpioLow.sh"

images_train = []
labels_train = []

images_test = []
labels_test = []

rc = subprocess.call(gpio_init)


#r = requests.post("http://09846fe9.ngrok.io/api/isupdated", data={'email': 'maxkaran2@gmail.com'})
#j = json.loads(r.content)

write_faces()
proc_faces = process_server_faces()

images_train = proc_faces['images_train']
labels_train = proc_faces['labels_train']



'''
j = json.loads(r.content)
F = open("user.json", "w")
F.write(str(j))
F.close()
'''

#print j[0]['fid']

#imgdata = base64.b64decode(j[0]["faces"][0])
#with open ('user.json', 'wb') as f:
#    f.write(imgdata)


'''
with open('face_csv.txt', 'rb') as csvfile:
    fileReader = csv.reader(csvfile, delimiter=";")
    for row in fileReader:
        images_train.append(cv2.imread(row[0], 0))
	labels_train.append(int(row[1]))

with open('face_csv_test.txt', 'rb') as csvfile:
    fileReader = csv.reader(csvfile, delimiter=";")
    for row in fileReader:
        images_test.append(cv2.imread(row[0], 0))
	labels_test.append(int(row[1]))

labels_train = np.array(labels_train)
labels_test = np.array(labels_test)
'''

model = cv2.createEigenFaceRecognizer(50);
model.train(images_train, labels_train)

print "done training"

test_predictions = []

'''
for i in range(len(labels_test)):
    
    pred, conf = model.predict(images_test[i])
    
    print "pred: ", pred, " conf: ", conf

count_tot = 0
count_err = 0    
for i in range(len(test_predictions)):
    count_tot += 1
    if test_predictions[i][0] != labels_test[i]:
	count_err += 1
'''

#err_rate = float(count_err) / count_tot
#accuracy = 1 - err_rate
#print accuracy

while(True):
    cap = cv2.VideoCapture(1)
    ret, frame = cap.read()
    cap.release()
    #frame = cv2.imread('steve.jpg', 1);
    #cv2.imshow('frame', frame)
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(
        frame,
        scaleFactor=1.5,
	minNeighbors=5,
	minSize=(60,60),
	flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    
    for(x,y,w,h) in faces:
	#print 'face detected'
        # cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        crop_img = frame[y:y+h, x:x+w]
        #height, width, depth = crop_img.shape
        resize_img = cv2.resize(crop_img, (92,112))
	
	pred, dist = model.predict(resize_img)
	print "pred: ", pred, " distance: ", dist
	cv2.imshow("max", resize_img)
	
    	if(cv2.waitKey(1) == 27):
	    break
	

	if(dist < 2250):
	    print "unlock door"
	    print "Label: ", pred
	    rc = subprocess.call(gpio_set_high)
	    time.sleep(10)
	    print "lock door"
            rc = subprocess.call(gpio_set_low)


