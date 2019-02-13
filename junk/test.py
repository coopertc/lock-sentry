import cv2

def show_webcam():
    cam = cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)I420, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
    faceCascade = cv2.CascadeClassifier(cascPath)

    while True:
        ret_val, img = cam.read()
        faces = faceCascade.detectMultiScale(
            img,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(120,120),
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow('webcam', img)
        if cv2.waitKey(1) == 27:
            break


show_webcam()
