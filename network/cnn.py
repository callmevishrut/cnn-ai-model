import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import cv2

#First we have to seed random weights
np.random.seed(3)

X=[]
Y=[]

with open('actions.csv','r') as csv:
    for line in csv:
        Y.append(line.rstrip())


all_images = []
img_num = 0
while img_num <2700:
    img = cv2.imread(r'./images/frame_{0}.jpg'.format(img_num),cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (0,0),fx=0.5, fy=0.5)
    img = img[:,:,np.newaxis]
    all_images.append(img)
    img_num+=1

X = np.array(all_images)
print(X[0].shape)

# Now we split the training set and testing set
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = .2, random_state = 5) #20% of the data becomes the testing set

img_x, img_y = 160, 270
input_shape = (img_x, img_y, 1)

# Now we convert these class vectors to binary class matrices for use in categorical_crossentropy loss function
classifications = 3 #Duck, jump, do nothing
y_train = keras.utils.to_categorical(y_train, classifications) #converts it to a binary class matrix
y_test = keras.utils.to_categorical(y_test, classifications)

# Now we design the architecture of our CNN
model = Sequential()
## First layer is a C layer with relu activation
model.add(Conv2D(32, kernel_size=(3,3), strides=(2,2), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
## Flattening the net
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(classifications, activation='softmax'))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

## activating tensorboard callback for acc and loss graphs
##tensorboard = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
history = model.fit(x_train, y_train, batch_size=250, epochs=50, validation_data=(x_test, y_test))

#saving the model post training

model.save('dino_basic_1.h5')

## Designing the confusion matrix
