import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

"""
Reading image.
"""


img = cv.imread('files/messi5.jpg', cv.IMREAD_COLOR)

# open image
cv.imshow('image 1', img)
(h, w, d) = img.shape
print("height: {h} width: {w} depth:{d}".format(h=h, w=w, d=d))
roi = img[50:110, 180:215]
cv.imshow('roi', roi)

resized = cv.resize(img, (200, 200))
cv.imshow('resized', resized)

r = 300 / w
dim = (300, int(h * r))
resized2 = cv.resize(img, dim)
cv.imshow('resized2', resized2)

center = (w // 2, h // 2)
M = cv.getRotationMatrix2D(center, -45, 1)
rotated = cv.warpAffine(img, M, (w, h))
cv.imshow('rotated', rotated)

blurred = cv.GaussianBlur(img, (21, 21), 0)
cv.imshow('blurred', blurred)

img_text = img.copy()
texted = cv.putText(img_text, 'Yehoo! Goall!!', (10, 25), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
cv.imshow('img texted', texted)
k = cv.waitKey(0) & 0xFF
if k == 27:         # wait for ESC key to exit
    cv.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv.imwrite('files/messigray.png',img)
    cv.destroyAllWindows()

# open image with matplotlib
plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.xticks([])
plt.yticks([])
plt.show()


"""
Capture frame from camera.
"""

"""
# capture from camera
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    print(cap.get(4))
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    cv.imshow('frame', gray)
    if (cv.waitKey(1) & 0xFF == ord('q')):
        break

cap.release()
cv.destroyAllWindows()
"""

"""
Capture frame from a file.
"""

"""
# capture from a file
cap = cv.VideoCapture('files/vtest.avi')

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    cv.imshow('frame', gray)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
"""

"""
Capture from camera and save it to a file
"""

"""
cap = cv.VideoCapture(0)

fourcc = cv.VideoWriter_fourcc(*'MJPG')
out = cv.VideoWriter('files/output.mp4', fourcc, 20.0, (640, 480))

while(cap.isOpened()):
    ret, frame = cap.read()

    if (ret == True):
        frame = cv.flip(frame, 1)

        out.write(frame)

        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv.destroyAllWindows()
"""

"""
Drawing in images, videos
"""

"""
img = np.zeros((512, 512, 3), np.uint8)

cv.line(img, (0,0), (511, 511), (255, 0, 0), 5)
cv.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)
cv.circle(img, (447, 63), 63, (0, 0, 255), -1)
cv.ellipse(img, (256, 256), (100, 50), 0, 0, 270, 255, -1)

pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
cv.polylines(img,[pts],True,(0,255,255), 3)

font = cv.FONT_HERSHEY_COMPLEX
cv.putText(img, 'OPenCV', (10, 500),font, 4, (255, 255, 255), 2, cv.LINE_AA)

cv.imshow('frame', img)
k = cv.waitKey(0) & 0xFF
if k == 27:         # wait for ESC key to exit
    cv.destroyAllWindows()
"""

events = [i for i in dir(cv) if 'EVENT' in i]
print(events)

drawing = False
mode = True
ix, iy = -1, -1

# mouse callback
def draw_circle(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDBLCLK:
        cv.circle(img, (x, y), 100, (255, 0, 0), -1)

def draw_circle2(event, x, y, flags, param):
    global ix, iy, drawing, mode

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
            else:
                cv.circle(img, (x, y), 5, (0, 0, 255), -1)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        


img = np. zeros((512, 512, 3), np.uint8)
cv.namedWindow('image')
cv.setMouseCallback('image', draw_circle2)

while True:
    cv.imshow('image', img)
    k = cv.waitKey(1) & 0xFF

    if k == ord('m'):
        mode = not mode
    elif k == 27:
        break
    
cv.destroyAllWindows()