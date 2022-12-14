import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp # Go through Media pipe
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense,Dropout
from keras.callbacks import TensorBoard

mp_holistic =mp.solutions.holistic # Bringing holsitic model
mp_drawing =mp.solutions.drawing_utils # Drawing utilities
def mediapipe_detection(image,model):
    image =cv2.cvtColor(image,cv2.COLOR_BGR2RGB) # COLOR CONVERSION - BGR 2 RGB
    image.flags.writeable= False                 # Image is no longer writeable (Helps in saving memory)
    results=model.process(image)                 # Make Predictions
    image.flags.writeable=True                   # Image is now writeable
    image =cv2.cvtColor(image,cv2.COLOR_RGB2BGR) # COLOR CONVERSION - RGB 2 BGR
    return image,results
def draw_styled_landmarks(image,results):
    # Draw Face Connections
    mp_drawing.draw_landmarks(image,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,110,10),thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(256,256,121),thickness=2,circle_radius=2)
                             )
    mp_drawing.draw_landmarks(image,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245,117,66),thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2)
                             )
def extract_keypoints(results):    
    # Similarly for left hand and right hand
    lh=np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)  # The if statement is for when there is no reading for yout left hand
    rh=np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)  # The if statement is for when there is no reading for yout right hand
    return np.concatenate([lh,rh]) # RETURNS ALL THE KEYPOINTS IN ONE 2D First 132 values is pose keypoints next is 468 value value of face keypoint and so on

# Actions that we try to detect
actions = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V' ,'W', 'X', 'Y', 'Z'])

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense,Dropout
from keras.callbacks import TensorBoard
# load existing model
actions = np.asarray(actions)
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
model.load_weights('Trained_on_all_letters.h5')

colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

# New detection variables
sequence = []
sentence=[]
predictions=[]
no_hand_count = 0
threshold = 0.5
say_check = False
cap = cv2.VideoCapture(0)
letter = ""
pre_letter = ""
f=open('demo.txt',"a")
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        if not keypoints.any():
            no_hand_count += 1 
            if no_hand_count > 5:
                sequence = []
                sentence = []
                pre_letter = ''
                letter = ''
                say_check = False
        else:
            no_hand_count = 0
            sequence.append(keypoints)
            sequence = sequence[-30:]
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predictions.append(np.argmax(res))
             #3. Viz logic
            if res[np.argmax(res)] > threshold:
                pre_letter = letter
                letter =  actions[np.argmax(res)]
                if letter != pre_letter and np.unique(predictions[-10:])[0]==np.argmax(res):
                    print(letter)
                    f.write(letter+" ")
                    sentence.append(letter)
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)        
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    f.close()