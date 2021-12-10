import cv2
import numpy as np


cap = cv2.VideoCapture('Files/video.mp4')

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=70)

counter_in = 0
counter_out = 0

def get_center(x, y, w, h):
    x1 = int(w/2)
    y1 = int(h/2)
    xc = x+x1
    yc = y+y1
    return xc, yc


while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 5)

    img_sub = object_detector.apply(blur)
    dilate = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)

    contours, heirarchy = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    vehicle_box = []
    vehicle_center = []

    cv2.line(frame, (40, 550), (1150, 550), (0, 0, 255), 2)

    for contour in contours:
        #cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        area = cv2.contourArea(contour)



        if area > 3500:
            #cv2.drawContours(ROI, [contour], -1, (0, 0, 255), 2)
            x, y, w, h = cv2. boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            vehicle_box.append([x, y, w, h])

            x_cen, y_cen = get_center(x, y, w, h)
            vehicle_center.append([x_cen, y_cen])

            cv2.circle(frame, (x_cen, y_cen), 1, (0, 0, 255), 3)



            for (x, y) in vehicle_center:
                if (y < 550+6) and (y > 550-6) and (40 < x < 640):
                    counter_in += 1
                    cv2.line(frame, (40, 550), (1150, 550), (0, 255, 0), 2)
                if (y < 550+6) and (y > 550-6) and (640 < x < 1150):
                    counter_out += 1
                    cv2.line(frame, (40, 550), (1150, 550), (0, 255, 0), 2)
                vehicle_center.remove([x, y])
            print(counter_in, counter_out)














    cv2.imshow('frame', frame)
    cv2.imshow('frame1', close)

    k = cv2.waitKey(24)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()