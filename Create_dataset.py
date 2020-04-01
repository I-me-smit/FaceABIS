import cv2
import os

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
Id = input("Enter Your ID - ")

Num = 0

while (True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.2, 5)

    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        Num = Num + 1
        if not os.path.exists("Dataset"):
            os.makedirs("Dataset")
        cv2.imwrite("DataSet/user." + Id + '.' + str(Num) + ".jpg", gray[y:y + h, x:x + w])
        cv2.imshow('frame', img)

    
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    elif Num > 10:
        break


cap.release()
cv2.destroyAllWindows()