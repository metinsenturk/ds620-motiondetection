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


def diff_img(t_0, t_1, t_2):
    """
    document
    """
    d_1 = cv2.absdiff(t_2, t_1)
    d_2 = cv2.absdiff(t_2, t_0)
    return cv2.bitwise_and(d_1, d_2)


if __name__ == '__main__':
    static_background = None

    video = cv2.VideoCapture(0)
    video.set(3, 1000)
    video.set(4, 1000)

    while True:
        check, frame = video.read()
        check2, frame2 = video.read()

        frame = cv2.resize(frame, (0,0), fx=0.4, fy=0.4)
        frame2 = cv2.resize(frame2, (0,0), fx=0.5, fy=0.5) 

        motion = 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        edge = cv2.Canny(gray, 35, 125)

        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.GaussianBlur(gray, (21, 21), 0)

        if static_background is None:
            static_background = gray

        diff_btw_background = cv2.absdiff(static_background, gray)
        # diff_btw_background = dist_map(frame, frame2)
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

        cv2.imshow("frame", frame)
        cv2.imshow("gray", gray)
        cv2.imshow("edge", edge)
        cv2.imshow("diff_btw_background:", diff_btw_background)
        cv2.imshow("threshold_frame:", threshold_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
