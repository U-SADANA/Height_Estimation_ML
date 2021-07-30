import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
import pandas as pd


def calculatelength(a,b):
    a=np.array(a)
    b=np.array(b)
    length=np.sqrt(((a[0]-b[0])*(a[0]-b[0]))+((a[1]-b[1])*(a[1]-b[1])))
    apt=(length*100)/2
    return apt

def height(age,gender,c):
    global ht
    print("Ulna:",c)
    if gender=="FEMALE" and int(age)<=65 :
        df = pd.read_csv(r"fb65.csv")
        etc = ExtraTreesRegressor()
        etc.fit(df[['Length']], df['Women'])
        ht = etc.predict([[c]])

    elif gender=="FEMALE" and int(age)>65:
        df = pd.read_csv(r"fa65.csv")
        etc = ExtraTreesRegressor()
        etc.fit(df[['Length']], df['Women'])
        ht = etc.predict([[c]])

    elif gender=="MALE" and int(age)<=65:
        df = pd.read_csv(r"mb65.csv")
        etc = ExtraTreesRegressor()
        etc.fit(df[['Length']], df['Men'])
        ht = etc.predict([[c]])


    elif gender=="MALE" and int(age)>65:
        df = pd.read_csv(r"ma65.csv")
        etc = ExtraTreesRegressor()
        etc.fit(df[['Length']], df['Men'])
        ht = etc.predict([[c]])



mpDraw=mp.solutions.drawing_utils
mpPose=mp.solutions.pose
pose=mpPose.Pose()

while True:
    age=input("Enter your age:")
    if age.isnumeric() and (int(age)<=150):
        break
    else:
        continue

while True:
    gen=input("Enter your gender:")
    gender=gen.upper()
    if gender=="FEMALE" or gender=="MALE":
        break
    else:
        continue

cap= cv2.VideoCapture(0)

while True:

    success, img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=pose.process(imgRGB)

    try:
        landmarks=results.pose_landmarks.landmark
        print(landmarks)
    except:
        pass

    if results.pose_landmarks:
       for id,lm in enumerate(results.pose_landmarks.landmark):
            if id==13 or id==15:
               h,w,c=img.shape
               print(id, lm)
               cx,cy=int(lm.x*w) ,int(lm.y*h)
               cv2.circle(img,(cx,cy),10,(255,0,0),cv2.FILLED)
               mpDraw.draw_landmarks(img, results.pose_landmarks,mpPose.POSE_CONNECTIONS)
               leftelbow=[landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].y]
               wrist=[landmarks[mpPose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mpPose.PoseLandmark.LEFT_WRIST].y]
               ulna=calculatelength(leftelbow,wrist)
               height(age,gender,ulna)

    cv2.imshow("img", img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        print("Height:",ht)
        break

