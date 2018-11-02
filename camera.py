import cv2 as cv

cam = cv.VideoCapture(1)
cv.waitKey(2000)
print(cam.isOpened())

cv.namedWindow('origin', cv.WINDOW_NORMAL)
cv.namedWindow('grayscale', cv.WINDOW_GUI_NORMAL)

firstFrame = None

while True:
    ret, frame = cam.read()    

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(frame, (21, 21), 0)
    
    if firstFrame is None:
        firstFrame = gray
    
    width = int(frame.shape[1] * 75/100)
    height = int(frame.shape[0] * 75/100)
    cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)
    cv.resize(gray, (width, height), interpolation=cv.INTER_AREA)

    cv.imshow('origin', frame)
    cv.imshow('grayscale', gray)

    # compute the absolute difference between the current frame and
    frame_delta = cv.absdiff(firstFrame, gray)
    thresh = cv.threshold(frame_delta, 25, 255, cv.THRESH_BINARY)[1]
    thresh = cv.dilate(thresh, None, iterations=2)
    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    cv.imshow("Security Feed", frame)
    cv.imshow("Thresh", thresh)
    cv.imshow("Frame Delta", frame_delta)

    for c in cnts:
        if cv.contourArea(c) < 500:
            continue
        
        (x, y, w, h) = cv.boundingRect(c)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)    

    if (cv.waitKey(1) & 0xFF == ord('q')):
        break    

cam.release()
cv.destroyAllWindows()

