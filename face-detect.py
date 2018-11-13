import cv2 as cv


def detectAndDisplay(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    # -- Detect faces
    faces = face_cascade.detectMultiScale(
        frame_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    for (x, y, w, h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv.ellipse(frame, center, (w//2, h//2),
                           0, 0, 360, (255, 0, 255), 4)
        faceROI = frame_gray[y:y+h, x:x+w]
        
        # -- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv.circle(frame, eye_center, radius, (255, 0, 0), 4)
    cv.imshow('Capture - Face detection', frame)


def init_video():
    cap = cv.VideoCapture(0)

    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)

    while True:
        ret, frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break
        detectAndDisplay(frame)
        if cv.waitKey(1) == ord('q'):
            break


def init_photo(path_to_photo):
    frame = cv.imread(path_to_photo)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    detectAndDisplay(frame)

    cv.waitKey(0)

if __name__ == "__main__":
    face_cascade = cv.CascadeClassifier()
    eyes_cascade = cv.CascadeClassifier()

    if not face_cascade.load('data/haarcascade_frontalface_default.xml'):
        print('--(!)Error loading face cascade')
        exit(0)
    if not eyes_cascade.load('data/haarcascade_eye.xml'):
        print('--(!)Error loading eyes cascade')
        exit(0)    

    # init_photo('files/messi5.jpg')
    init_video()