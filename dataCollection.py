import cv2
#Detector de la mano
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)

#Detector el numero de manos
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

#Datos para la captura de las imagenes
folder = "Data/C"
counter = 0

#Captura de imagen, manos a través de la Webcam
while True:
    success, img = cap.read()
    #Detector de manos
    hands, img = detector.findHands(img)

    #Cortando la mano de la captura
    if hands:
        hand = hands[0]
        #Devolviendo un cuadro limitado en la mano con las dimensiones x, y, w, h
        x, y, w, h = hand['bbox']

        #Asignando un fondo blanco. Las medidas que se asigno a imgSize
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        #cortando la imagen segun la dimension de inicio y fin x (inicio) : x+w (fin)
        imgCrop = img[y-offset:y + h+offset, x - offset: x + w + offset]

        imgCropShape = imgCrop.shape

        #Calculando la variacion de tamaño la ventana imgWhite entre weight and high
        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgSize/h
            wCalculate = math.ceil(k*w)

            imgResize = cv2.resize(imgCrop, (wCalculate, imgSize))
            # Matriz de tres variables: alto, largo y sus canales (channels)
            imgResizeShape = imgResize.shape
            #Centrando al background
            wGap = math.ceil((imgSize - wCalculate)/2)
            #Asignango la imagen cortada de la mano en el background
            imgWhite[:, wGap:wCalculate + wGap] = imgResize

        else:
            k = imgSize / w
            hCalculate = math.ceil(k*h)

            imgResize = cv2.resize(imgCrop, (imgSize, hCalculate))
            # Matriz de tres variables: alto, largo y sus canales (channels)
            imgResizeShape = imgResize.shape
            #Centrando al background
            hGap = math.ceil((imgSize - hCalculate)/2)
            #Asignango la imagen cortada de la mano en el background
            imgWhite[hGap:hCalculate + hGap, :] = imgResize


        cv2.imshow("imgCrop", imgCrop)
        cv2.imshow("imgWhite", imgWhite)

    cv2.imshow("Image", img)
    #cv2.waitKey(1)

    #Para detener la captura de la imagen asignamos la letra s como boton de apagado
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
########################################

