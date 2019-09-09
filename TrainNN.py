import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

(trainX, trainY), (testX, testY) = mnist.load_data()

x = testX.shape
print(x)

numPixels = trainX.shape[1]*trainX.shape[2]

print(trainX.shape)
print(testX.shape)
trainX = trainX.reshape(trainX.shape[0], 1, 28, 28).astype('float32')
testX = testX.reshape(testX.shape[0], 1, 28, 28).astype('float32')
x = testX.shape

trainX = trainX/255
testX = testX/255

trainY = np_utils.to_categorical(trainY)
testY = np_utils.to_categorical(testY)
num_of_classes = testY.shape[1]


def handModel():
    print("Start")
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu', data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_of_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("modelCompiled")
    return model


model = handModel()
print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)

model.fit(trainX, trainY, validation_data=(testX, testY), epochs=15, batch_size=128)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

score = model.evaluate(testX, testY)
print(score)

