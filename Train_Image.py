#Train_Image.py

import os
import time
import cv2
import numpy as np
from PIL import Image
from threading import Thread

# -------------- image labesl ------------------------
def getImagesAndLabels(path):
    # get the path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    for f in os.listdir(path):
        print(f)

    print(imagePaths)
    # create empty face list
    faces = []
    # create empty ID list
    Ids = []
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')
        # getting the Id from the image
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids


# ----------- train images function ---------------
# def TrainImages():
#     recognizer = cv2.face_LBPHFaceRecognizer.create()
#     # Local Binary Patterns Histogram(LBPH) method
#     recognizer=cv2.face.LBPHFaceRecognizer_create()
#     harcascadePath = r"C:\Users\Lenovo\Downloads\recognition-main\recognition-main\haarcascade_frontalface_default.xml"
#     detector = cv2.CascadeClassifier(harcascadePath)
#     faces, Id = getImagesAndLabels(r"D:\Pycharm\Anu_Project_1\recognition-main\recognition-main\TrainingImage")
#     Thread(target = recognizer.train(faces, np.array(Id))).start()
#     # Below line is optional for a visual counter effect
#     Thread(target = counter_img(r"D:\Pycharm\Anu_Project_1\recognition-main\recognition-main\TrainingImage")).start()
#     recognizer.save(r"D:\Pycharm\Anu_Project_1\recognition-main\recognition-main\TrainingImageLabel"+os.sep+r"Trainner.yml")
#     print("All Images")

def TrainImages():
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        harcascadePath = r"C:\Users\user1\PycharmProjects\archa meha\recgonition mail\haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        training_image_path = r"C:\Users\user1\PycharmProjects\archa meha\recgonition mail\TrainingImage"
        faces, Ids = getImagesAndLabels(training_image_path)

        if not faces or not Ids:
            print("No valid images or IDs found for training.")
            return

        print(f"Training on {len(faces)} images...")
        recognizer.train(faces, np.array(Ids))

        model_path = os.path.join(training_image_path, "Trainner.yml")
        recognizer.save(model_path)
        print(f"Training completed. Model saved to {model_path}.")
    except Exception as e:
        print(f"An error occurred during training: {e}")



# Optional, adds a counter for images trained (You can remove it)
def counter_img(path):
    imgcounter = 1
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    for imagePath in imagePaths:
        print(str(imgcounter) + " Images Trained", end="\r")
        time.sleep(0.008)
        imgcounter += 1


print(getImagesAndLabels(r"C:\Users\user1\PycharmProjects\archa meha\recgonition mail\TrainingImage"))
