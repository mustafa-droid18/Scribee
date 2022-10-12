import numpy as np
import os
from matplotlib import pyplot as plt
import mediapipe as mp # Go through Media pipe

NPUT_VIDEO_PATH = os.path.join('Video_Data')

VID_SEQUENCES = 90 # No. of videos recorded at a time

sequence_length = 30 # Videos are going to be 30 frames in length

label_map = {} # map of the labels
no_sequences = {} # no. of videos for each action
actions = [] # ASL alphabet gestures

'''for letter in ascii_uppercase:
    no_sequences[letter] = VID_SEQUENCES

actions = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V' ,'W', 'X', 'Y', 'Z']'''

actions=['1','2']

for action in actions:
    no_sequences[action]=VID_SEQUENCES

DATA_PATH = os.path.join('MP_DATA')

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix,accuracy_score

label_map = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences[action]):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

print(label_map)
X = np.array(sequences)
y = to_categorical(labels).astype(int)
print("Sequence shape:", np.array(sequences).shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

print(X.shape)

actions = np.asarray(actions)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,126)))
model.add(Dropout(0.25))
model.add(LSTM(256, return_sequences=True, activation='relu'))
model.add(Dropout(0.25))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(Dropout(0.25))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=1000, callbacks=[tb_callback])

res = model.predict(X_test)
print(actions[np.argmax(res[20])])
print(actions[np.argmax(y_test[20])])


yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()
print("The model has an accuracy of: ", accuracy_score(ytrue, yhat))

model.save('action_test_full_v4.h5')
