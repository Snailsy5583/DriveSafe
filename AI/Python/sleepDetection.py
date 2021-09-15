import cv2
import numpy as np
import dlib
import math

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def GetDist(p_width):
    return (24.2 * 705) / p_width

def CreateBox(img, points):
    x,y,w,h = cv2.boundingRect(points)
    imgCrop = img[y:y+h, x:x+w]
    imgCrop = cv2.resize(imgCrop, (0, 0), None, 10, 10)
    return imgCrop, h

def GetEyeWidth(points):
    return (points[36][0] - points[39][0])**2 + (points[36][1] - points[39][1])**2

def GetEyeHeight(points):
    return (points[37][0] - points[41][0])**2 + (points[37][1] - points[41][1])**2

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        # frame = cv2.resize(frame, (0, 0), None, 0.5, 0.5)
        imgO = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()
            imgO = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            landmarks = predictor(gray, face)
            myPoints = []
            for i in range(68):
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                myPoints.append([x, y])
                cv2.circle(imgO, (x,y), 1, (0,255, 0), cv2.FILLED)
            
            imgLeftEye, height = CreateBox(imgO, np.array(myPoints[36:42]))

            width, height = GetEyeWidth(myPoints), GetEyeHeight(myPoints)

            thresh = 5.5**2
            cv2.putText(imgO, f'{width/height}, {thresh}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            if width/height > thresh:
                cv2.circle(imgLeftEye, (0, 0), 500, (0,0,255), cv2.FILLED)
                cv2.putText(imgO, "Blinking", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            cv2.imshow("LeftEye", imgLeftEye)
            
        imgR = cv2.resize(imgO, (0, 0), None, 2, 2)


        cv2.imshow('Frame', imgR)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()