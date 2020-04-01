import numpy as np
import cv2
import time
import shutil
import os


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


recognizer = cv2.face.LBPHFaceRecognizer_create()

#recognizer = cv2.face.EigenFaceRecognizer_create()

#recognizer = cv2.face.FisherFaceRecognizer_create()

recognizer.read("Trainer.yml")
cap = cv2.VideoCapture(0)
Id = 0
Name = ""
font = cv2.FONT_HERSHEY_COMPLEX

# while True:
time.sleep(1)
ret, img = cap.read()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in faces:


    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 225, 225), 2)
    
    Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
    #Here you can make any number of ids please change according to your name and id(that you have typed in your first file) 
    if Id == 1:
        Name = "Smit"
        tmpfilepath=""
        for i in range(1,30):
            tmpfilepath="C:\\Users\\SmitRL\\Desktop\\mosip\\face_recognizer\\Dataset\\user."+str(Id)+"."+str(i)+".jpg"
            if os.path.exists(tmpfilepath):
                shutil.copy(tmpfilepath, "C:\\Users\\SmitRL\\Desktop\\mosip\\face_recognizer\\RecognisedImages")
            else:
                break

    elif Id == 2:
        Name = "Rachna"
    elif Id== 3:
        Name = "Aishwarya"
    elif Id== 3:
        Name = "Jainil"
    elif Id== 3:
        Name = "Vaibhav"
        
       
            
    cv2.rectangle(img, (x - 22, y - 45), (x + w + 15, y - 22), (0, 0, 0), -1)
    cv2.putText(img,"Name:" + str(Name), (x, y - 22), font, 1, (0, 0, 255), 2)
    
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

time.sleep(3)

cap.release()
cv2.destroyAllWindows()