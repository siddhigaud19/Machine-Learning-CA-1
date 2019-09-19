import cv2,os
import numpy as np
from PIL import Image
import pickle
import sqlite3
import speech_recognition as sr
import pyttsx3

engine = pyttsx3.init()
 
r = sr.Recognizer()

path = os.path.dirname(os.path.abspath(__file__))

fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (255, 255, 255)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('D:/face_rec/trainner/trainner.yml')
cascadePath = "D:/face_rec/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

def getProfile(Id):
    conn=sqlite3.connect("D:/face_rec/FaceBase.db")
    cmd="SELECT * FROM People WHERE Id="+str(Id)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor :
        profile=row
        #conn.close()
    return profile

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces:
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
        profile=getProfile(Id)
        #last=str(profile[1])
        #print(""+last)
        #temp=""
        if(profile!=None):
           # cv2.putText(im, str("Name :"+str(profile[1])), (x,y+h+70), fontface, fontscale, fontcolor)
           # cv2.putText(im, str("Age :"+str(profile[2])), (x,y+h+100), fontface, fontscale, fontcolor)
        #if(last!=temp):
           # engine.say("hi"+last)
           # engine.runAndWait()
           # temp=str(profile[1])
       # engine.say("hi"+str(profile[1]))
       # engine.runAndWait()
           
        cv2.imshow('I Know All abt You',im)
        cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()        