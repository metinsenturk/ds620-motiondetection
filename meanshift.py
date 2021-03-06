import numpy as np
import cv2 as cv


if __name__ == '__main__':
    cap = cv.VideoCapture(0, )
    cap.set(3, 1920)
    cap.set(4, 1080)
    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    out = cv.VideoWriter('output.avi',fourcc, 20.0, (1920, 1080))
    # cap = cv.VideoCapture(1)
    # take first frame of the video
    ret, frame = cap.read()
    # setup initial location of window
    r, h, c, w = 250, 90, 400, 125  # simply hardcoded the values
    track_window = (c, r, w, h)
    # set up the ROI for tracking
    roi = frame[r:r+h, c:c+w]
    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)),
                    np.array((180., 255., 255.)))
    roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
    while(1):
            ret, frame = cap.read()
            if ret == True:
                hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
                # apply meanshift to get the new location
                ret, track_window = cv.CamShift(dst, track_window, term_crit)
                # Draw it on image
                x, y, w, h = track_window
                img2 = cv.rectangle(frame, (x, y), (x+w, y+h), 255, 2)

                # write the flipped frame
                out.write(img2)

                cv.imshow('img2', img2)

                if cv.waitKey(60) & 0xff == ord('q'):
                    break
            else:
                break
    
    cap.release()
    out.release()

    cv.destroyAllWindows()
