import cv2


cap = cv2.VideoCapture('Files/video.mp4')

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=70)


while True:
    ret, frame = cap.read()

    ROI = frame[350:600, 200:1100]

    mask = object_detector.apply(ROI)
    _, mask = cv2.threshold(mask, 230, 255, cv2.THRESH_BINARY)

    contours, heirarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    vehicle_box = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area >1500:
            #cv2.drawContours(ROI, [contour], -1, (0, 0, 255), 2)
            x, y, w, h = cv2. boundingRect(contour)
            cv2.rectangle(ROI, (x, y), (x+w, y+h), (0, 0, 255), 2)
            vehicle_box.append([x, y, w, h])


    cv2.imshow('Frame', ROI)
    cv2.imshow('mask', mask)

    key = cv2.waitKey(30)
    if key == 27:
        break



cap.release()
cv2.destroyAllWindows()