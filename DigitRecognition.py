import cv2
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import model_from_json

# Training the NN (Do this or load weights directly)
# epochs = 15
#
# (trainX, trainY), (testX, testY) = mnist.load_data()
#
# trainX = trainX.reshape(trainX.shape[0], 1, 28, 28).astype('float32')
# testX = testX.reshape(testX.shape[0], 1, 28, 28).astype('float32')
#
# trainX = trainX/255
# testX = testX/255
#
# trainY = np_utils.to_categorical(trainY)
# testY = np_utils.to_categorical(testY)
# num_of_classes = testY.shape[1]
#
#
# def handModel():
#     print("Start")
#     model = Sequential()
#     model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu', data_format='channels_first'))
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(Dropout(0.2))
#     model.add(Flatten())
#     model.add(Dense(512, activation='relu'))
#     model.add(Dense(num_of_classes, activation='softmax'))
#
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     print("modelCompiled")
#     return model

# model = handModel()
# model.fit(trainX, trainY, validation_data=(testX, testY), epochs=2, batch_size=128)

# Download NN architecture and weights from 'model.json' and 'model.h5' respectively
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# Load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
model = loaded_model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Initialisation before capturing the video (capture from first web cam)
cap = cv2.VideoCapture(0)


def nothing(x):
    pass


# Track bar now set to capture blue color
cv2.namedWindow('Tracker')
cv2.createTrackbar('lh', 'Tracker', 110, 255, nothing)
cv2.createTrackbar('ls', 'Tracker', 100, 255, nothing)
cv2.createTrackbar('lv', 'Tracker', 25, 255, nothing)
cv2.createTrackbar('uh', 'Tracker', 130, 255, nothing)
cv2.createTrackbar('us', 'Tracker', 255, 255, nothing)
cv2.createTrackbar('uv', 'Tracker', 255, 255, nothing)

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

mask_num = np.zeros([480, 640], 'uint8')


while True:
    # Capture from video cam
    _, frame = cap.read()

    # Mirror image
    frame = np.flip(frame, axis=1)

    # Convert from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Set track bar to capture required color(in this case blue)
    lh = cv2.getTrackbarPos('lh', 'Tracker')
    ls = cv2.getTrackbarPos('ls', 'Tracker')
    lv = cv2.getTrackbarPos('lv', 'Tracker')
    uh = cv2.getTrackbarPos('uh', 'Tracker')
    us = cv2.getTrackbarPos('us', 'Tracker')
    uv = cv2.getTrackbarPos('uv', 'Tracker')

    lower = np.array([lh, ls, lv])
    upper = np.array([uh, us, uv])

    # Capture the blue color in the frame
    mask = cv2.inRange(hsv, lower, upper)

    # Capture the movements made by blue color in the frame(by taking 'bitwise or' of previous and current frame)
    mask_num = cv2.bitwise_or(mask, mask_num)

    # Display the image
    cv2.imshow('Number', mask_num)

    key = cv2.waitKey(1)
    if key == 114:
        # Clear screen when 'r' is pressed
        mask_num = np.zeros([480, 640], 'uint8')
    if key == 27:
        # Exit the cam and break from loop when 'esc' is pressed
        break

# Pre-processing the image
mask_num = mask_num/255
mask_num = mask_num[:, 80:560]
mask_num = cv2.resize(mask_num, (28, 28))
mask_num = mask_num[np.newaxis, ...]
mask_num = mask_num[np.newaxis, ...]
img = mask_num

# Predict the image
prediction = model.predict(img)

# Print the prediction array
print(prediction)

# Print the predicted number and the confidence of it being correct
print("The number is: " + str(np.argmax(prediction)))
print("The confidence percentage is: " + str(np.max(prediction)*100))

# Destroy all opened windows
cap.release()
cv2.destroyAllWindows()
