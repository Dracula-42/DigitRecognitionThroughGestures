# DigitRecognitionThroughGestures

Draw a digit using gestures through webcam and the NN model predicts the digit drawn.

'model.json' and 'model.h5' have the NN architecture and NN weights respectively. Either use these weights in predicting digits or train your own model using the 'TrainNN.py' file which outputs both 'model.json' and 'model.h5' files

Run the 'DigitRecognition.py' file, your webcam will be opened and you will be able to draw the digit which tracks all the blue color present on the frame captured by webcam. Track the digit and press 'Esc' after you are done or press 'r' to reset. Once you are done, let the CNN do the magic.
Your digit will be predicted with its confidence value.
