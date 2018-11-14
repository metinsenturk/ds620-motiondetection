import cv2
import numpy as np

def init_face_detection():
    face_cascade = cv2.CascadeClassifier()
    eyes_cascade = cv2.CascadeClassifier()

    if not face_cascade.load('data/haarcascade_frontalface_default.xml'):
        print('--(!)Error loading face cascade')
        exit(0)
    if not eyes_cascade.load('data/haarcascade_eye.xml'):
        print('--(!)Error loading eyes cascade')
        exit(0)

    return face_cascade, eyes_cascade


def face_detect(frame, face_cascade, eyes_cascade):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    # -- Detect faces
    faces = face_cascade.detectMultiScale(
        frame_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    for (x, y, w, h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv2.ellipse(frame, center, (w//2, h//2),
                           0, 0, 360, (255, 0, 255), 4)
        faceROI = frame_gray[y:y+h, x:x+w]
        
        # -- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv2.circle(frame, eye_center, radius, (255, 0, 0), 4)
    
    return frame