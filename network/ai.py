from keras.models import load_model
import selenium
from mss import mss
import cv2
import numpy as np
import time

model = load_model('./network/dino_basic_1.h5')

start = time.time()

def predict(game_element):

    # configuration for image capture
    sct = mss()
    coordinates = { 'top': 235, 'left': 360, 'width': 210, 'height': 120}

    # image capture
   
    img = np.array(sct.grab(coordinates))

    # cropping, edge detection, resizing to fit expected model input
    img = img[::,75:615]
    img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
    img = cv2.Canny(img, threshold1=200, threshold2=300)
    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    img = img[np.newaxis, :, :, np.newaxis]
    img = np.array(img)

    # model prediction
    y_prob = model.predict(img)
    prediction = y_prob.argmax(axis=-1)

    if prediction == 1:
        # jump
        time.sleep(0.25)
        game_element.send_keys(u'\ue013')
        
        print('Should jump')
        time.sleep(.07)
    if prediction == 0:
        print('Should do nothing')
        # do nothing
        pass
    if prediction == 2:
        print('Should duck')
        game_element.send_keys(u'\ue015')
       
        # duck
    