import cv2
import pandas as pd
import numpy as np
from features.face import *
from features.hand import *
from features.motion import *

if __name__ == '__main__':
    # init face detection
    face_cascade, eyes_cascade = init_face_detection()    

    # live
    video = cv2.VideoCapture(0)
    video.set(3, 1920)
    video.set(4, 1080)
    fc = video.get(cv2.CAP_PROP_FRAME_COUNT)

    # init meanshift
    term_crit, roi_hist, track_window = init_meanshift(cv2, video.read()[1])

    # saving
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out_frame = cv2.VideoWriter('output_frame.avi',fourcc, fc, (1920, 1080))
    out_differ = cv2.VideoWriter('output_differmap.avi',fourcc, fc, (1920, 1080))
    out_thresold = cv2.VideoWriter('output_thresold.avi',fourcc, fc, (1920, 1080))

    while True:
        check2, frame2 = video.read()
        check, frame = video.read()
        
        frame_meanshift = meanshift(cv2, frame, term_crit, roi_hist, track_window)
        frame_meanshift = face_detect(frame_meanshift, face_cascade, eyes_cascade)

        gray, edge = apply_preprocessing(cv2, frame)
        gray2, edge2 = apply_preprocessing(cv2, frame2)

        differmap = dist_map(cv2, frame, frame2)
        
        threshold_frame, frame = find_contours(cv2, differmap, frame)

        frame = face_detect(frame, face_cascade, eyes_cascade)

        # saving 
        out_frame.write(frame)
        out_differ.write(differmap)
        out_thresold.write(threshold_frame)
        
        # displayin
        cv2.imshow("frame", frame)
        cv2.imshow("frame - meanshift", frame_meanshift)
        cv2.imshow("diff_btw_background:", differmap)
        cv2.imshow("threshold_frame:", threshold_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    out_frame.release()
    out_differ.release()
    out_thresold.release()
    cv2.destroyAllWindows()
