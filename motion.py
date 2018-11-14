import cv2
import pandas as pd
import numpy as np

# TODO: Face Detection 1

def dist_map(frame1, frame2):
    """outputs pythagorean distance between two frames"""
    frame1_32 = np.float32(frame1)
    frame2_32 = np.float32(frame2)
    diff32 = frame1_32 - frame2_32
    norm32 = np.sqrt(diff32[:, :, 0]**2 + diff32[:, :, 1] **
                     2 + diff32[:, :, 2]**2)/np.sqrt(255**2 + 255**2 + 255**2)
    dist = np.uint8(norm32*255)
    return dist


def avg_diff(frame1, frame2):
    avg = frame1.copy().astype("float")
    cv2.accumulateWeighted(frame2, avg, 0.5)
    frameDelta = cv2.absdiff(frame2, cv2.convertScaleAbs(avg))
    return frameDelta


def diff_img(t_0, t_1, t_2):
    """
    document
    """
    d_1 = cv2.absdiff(t_2, t_1)
    d_2 = cv2.absdiff(t_2, t_0)
    return cv2.bitwise_and(d_1, d_2)

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


def face_detect(frame):
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

if __name__ == '__main__':

    face_cascade, eyes_cascade = init_face_detection()

    video = cv2.VideoCapture(0)
    video.set(3, 1920)
    video.set(4, 1080)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output_m.avi',fourcc, 4, (1920, 1080))

    while True:
        check, frame = video.read()
        check2, frame2 = video.read()

        #frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        #frame2 = cv2.resize(frame2, (0, 0), fx=0.5, fy=0.5)

        motion = 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        edge = cv2.Canny(gray, 35, 125)

        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.GaussianBlur(gray, (21, 21), 0)

        #diff_btw_background = cv2.absdiff(static_background, gray)
        diff_btw_background = dist_map(frame, frame2)
        # diff_btw_background = diff_img(static_background, gray, gray2)

        threshold_frame = cv2.threshold(
            diff_btw_background, 30, 255, cv2.THRESH_BINARY)[1]
        threshold_frame = cv2.dilate(threshold_frame, None, iterations=2)

        (_, cnts, _) = cv2.findContours(
            threshold_frame.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for countour in cnts:
            if cv2.contourArea(countour) < 10000:
                continue
            else:
                motion = 1

                (x, y, w, h) = cv2.boundingRect(countour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        frame = face_detect(frame)


        out.write(frame)
        
        cv2.imshow("frame", frame)
        cv2.imshow("gray", gray)
        cv2.imshow("edge", edge)
        cv2.imshow("diff_btw_background:", diff_btw_background)
        cv2.imshow("threshold_frame:", threshold_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    out.release()
    cv2.destroyAllWindows()
