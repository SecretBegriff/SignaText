import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Modelo/keras_model.h5","Modelo/labels.txt")

offset = 20
imgSize = 300

folder = "Manos/a"
counter = 0

labels = ["A", "B", "C"]

imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # Definir imgWhite antes del bucle

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    
    try:
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[max(0, y - offset):min(y + h + offset, img.shape[0]),
                          max(0, x - offset):min(x + w + offset, img.shape[1])]

            imgCropShape = imgCrop.shape
            
            aspectRatio = h / w
            
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)      
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(img)
                print(prediction)
                
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)      
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("White", imgWhite)
            
            cv2.imshow("Ventana", img)
            cv2.waitKey(1)


    except Exception as e:
        print(f"Error: {e}")
        break;