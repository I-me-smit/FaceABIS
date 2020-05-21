import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()

path = 'Dataset'

modelId = input("Enter the model ID : ")

def Images_ID(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faces = []
    IDs = []
    
    for imagePath in imagePaths:
        face_img = Image.open(imagePath).convert('L')
        face = np.array(face_img,'uint8')
        ID = int(os.path.split(imagePath) [-1].split('.') [1])
        faces.append(face)
        print(ID)
        IDs.append(ID)
        cv2.imshow("Training_Images",face)
        cv2.waitKey(15)

    
    return np.array(IDs), faces

IDs, faces = Images_ID(path)

recognizer.train(faces,IDs)
recognizer.save("Models/"+modelId+".yml")

cv2.destroyAllWindows()