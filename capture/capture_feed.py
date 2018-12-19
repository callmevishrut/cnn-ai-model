import cv2
from mss import mss
import numpy as np
import keyboard
import os

def preprocessing(img):
    img = img[::,75:615] # changing the img into grey scale
    img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
    img = cv2.Canny(img, threshold1 = 200, threshold2 = 300) # edge detection
    return img

# Here we start capturing the dino game frame by frame when the human player interacts with the game
def start():
    # Capturing frame and croping unnnecessary things

    sct = mss()

    coordinates = { 'top': 235, 'left': 360, 'width': 210, 'height': 120}

    with open('actions.csv', 'w') as csv:
        x = 0

        if not os.path.exists(r'./images'):
            os.mkdir(r'./images')
        while True:
            img = preprocessing(np.array(sct.grab(coordinates)))
            try:
                # whenever the up arrow is pressed write 1 in csv file
                if keyboard.is_pressed('up arrow'):
                    cv2.imwrite('./images/frame_{0}.jpg'.format(x), img)
                    csv.write('1\n')
                    print('jumping data feeded')
                    x += 1
                    continue

                # whenever the down arrow is pressed write 2 in csv file
                if keyboard.is_pressed('down arrow'):
                    cv2.imwrite('./images/frame_{0}.jpg'.format(x), img)
                    csv.write('2\n')
                    print('ducking data feeded')
                    x += 1
                    continue

                # whenever nothing happens and t is pressed, write 0 in csv file
                if keyboard.is_pressed('t'):
                    cv2.imwrite('./images/frame_{0}.jpg'.format(x), img)
                    csv.write('0\n')
                    print('nothing data feeded')
                    x += 1
                    continue
                
                # Now we break the video feed
                if cv2.waitKey(25) == ord('q'): #when esape is pressed
                    csv.close()
                    cv2.destroyAllWindows()
                    break
            except:
                break