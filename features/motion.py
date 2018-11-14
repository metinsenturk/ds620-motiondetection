import numpy as np


def dist_map(cv2, frame1, frame2):
    """outputs pythagorean distance between two frames"""
    frame1_32 = np.float32(frame1)
    frame2_32 = np.float32(frame2)
    diff32 = frame1_32 - frame2_32
    norm32 = np.sqrt(diff32[:, :, 0]**2 + diff32[:, :, 1] **
                     2 + diff32[:, :, 2]**2)/np.sqrt(255**2 + 255**2 + 255**2)
    dist = np.uint8(norm32*255)
    return dist


def avg_diff(cv2, frame1, frame2):
    avg = frame1.copy().astype("float")
    cv2.accumulateWeighted(frame2, avg, 0.5)
    frame_delta = cv2.absdiff(frame2, cv2.convertScaleAbs(avg))
    return frame_delta


def diff_img(cv2, t_0, t_1, t_2):
    """finding abs diff within frames"""
    d_1 = cv2.absdiff(t_2, t_1)
    d_2 = cv2.absdiff(t_2, t_0)
    return cv2.bitwise_and(d_1, d_2)


def apply_preprocessing(cv2, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    edge = cv2.Canny(gray, 35, 125)

    return gray, edge

def find_contours(cv2, differ_map, frame):
    threshold_frame = cv2.threshold(differ_map, 30, 255, cv2.THRESH_BINARY)[1]
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
    
    return threshold_frame, frame

def init_meanshift(cv, frame):
    # setup initial location of window
    r, h, c, w = 500, 180, 800, 250  # simply hardcoded the values
    track_window = (c, r, w, h)
    # set up the ROI for tracking
    roi = frame[r:r+h, c:c+w]
    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

    return term_crit, roi_hist, track_window

def meanshift(cv, frame, term_crit, roi_hist, track_window):
    frame_inner = frame.copy() 

    hsv = cv.cvtColor(frame_inner, cv.COLOR_BGR2HSV)
    dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
    # apply meanshift to get the new location
    ret, track_window = cv.CamShift(dst, track_window, term_crit)
    # Draw it on image
    x, y, w, h = track_window
    frame_new = cv.rectangle(frame_inner, (x, y), (x+w, y+h), 255, 2)

    return frame_new