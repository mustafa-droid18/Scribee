import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp # Go through Media pipe
from string import ascii_uppercase

mp_holistic =mp.solutions.holistic # Bringing holsitic model
mp_drawing =mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image,model):
    image =cv2.cvtColor(image,cv2.COLOR_BGR2RGB) # COLOR CONVERSION - BGR 2 RGB
    image.flags.writeable= False                 # Image is no longer writeable (Helps in saving memory)
    results=model.process(image)                 # Make Predictions
    image.flags.writeable=True                   # Image is now writeable
    image =cv2.cvtColor(image,cv2.COLOR_RGB2BGR) # COLOR CONVERSION - RGB 2 BGR
    return image,results

def draw_styled_landmarks(imag,results):
    # Draw Face Connections
    '''mp_drawing.draw_landmarks(image,results.face_landmarks,mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(80,110,10),thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80,256,121),thickness=1,circle_radius=1)                            
                            )
    mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,22,10),thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80,44,121),thickness=2,circle_radius=2)
                            )'''
    mp_drawing.draw_landmarks(image,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,110,10),thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(256,256,121),thickness=2,circle_radius=2)
                            )
    mp_drawing.draw_landmarks(image,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245,117,66),thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2)
                            )

def extract_keypoints(results):    
    # pose=np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    # Similarly for left hand and right hand
    lh=np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)  # The if statement is for when there is no reading for yout left hand
    rh=np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)  # The if statement is for when there is no reading for yout right hand
    # face=np.array([[res.x,res.y,res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    return np.concatenate([lh,rh]) # RETURNS ALL THE KEYPOINTS IN ONE 2D First 132 values is pose keypoints next is 468 value value of face keypoint and so on

INPUT_VIDEO_PATH = os.path.join('Video_Data')

VID_SEQUENCES = 60 # No. of videos recorded at a time

sequence_length = 30 # Videos are going to be 30 frames in length

label_map = {} # map of the labels
no_sequences = {} # no. of videos for each action
actions = [] # ASL alphabet gestures

'''for letter in ascii_uppercase:
    no_sequences[letter] = VID_SEQUENCES
'''
# actions = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V' ,'W', 'X', 'Y', 'Z']

actions=['M']

# actions=['1','2']

for action in actions:
    no_sequences[action]=VID_SEQUENCES

DATA_PATH = os.path.join('MP_DATA_M_T')

'''for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH,action,str(sequence)))
        except:
            pass'''

INPUT_VIDEO_PATH = r'D:\AI-ML\Vodafone\Sign Language Detection\Data_M_T' #local location for dataset

cap = cv2.VideoCapture(0)

writer_check = False

# Determine width and height of frame
width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create directory if it does not exist already
if not os.path.exists(INPUT_VIDEO_PATH):
    os.mkdir(INPUT_VIDEO_PATH)
break_check = False

try:
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        # Loop through alphabet
        for action in actions:
            
            # Loop through sequences aka videos
            for sequence in range(VID_SEQUENCES):
                # Loop through video length aka sequence length
                for frame_num in range(sequence_length + 1):

                    # Read feed
                    ret, frame = cap.read()

                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)

                    # Draw landmarks
                    draw_styled_landmarks(image, results)
                    
                    # NEW Apply wait logic
                    if frame_num == 0:
                        name = action + '.' + str(sequence).zfill(3) + '.mp4v'
                        file_path = os.path.join(INPUT_VIDEO_PATH, name)
                        if not os.path.exists(file_path):
                            writer_check = True
                            cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                            cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            # Show to screen
                            cv2.imshow('OpenCV Feed', image)
                            cv2.waitKey(2000)
                            writer= cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*'DIVX'), 30, (width,height))
                        else:
                            writer_check = False
                            cv2.putText(image, 'DATA ALREADY COLLECTED', (120,200), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                            cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            cv2.imshow('OpenCV Feed', image)
                            cv2.waitKey(200)
                            break
                    else: 
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                        writer.write(frame)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        writer.release()
                        cap.release()
                        cv2.destroyAllWindows()
                        break_check = True 
                        break
            
                if writer_check == True:
                    writer.release() 
                    writer_check = False
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                writer.release()
                cap.release()
                cv2.destroyAllWindows()
                break_check = True 
                break
        writer.release()          
        cap.release()
        cv2.destroyAllWindows()
except cv2.error:
    print("Exited!")


#INPUT_VIDEO_PATH = os.path.join('Video Path','Data') #local location for dataset

i = 0
for j, vid in enumerate(os.listdir(INPUT_VIDEO_PATH)):
    print(vid)
    if '.mp4' in vid:
        string = vid.split('.')
        action = string[0]
        sequence = int(string[1])
        cap = cv2.VideoCapture(vid)
        label_map[action] = i
        
        if action in no_sequences:
            no_sequences[action] += 1
        else:
            no_sequences[action] = 1
        i += 1
        if action not in actions:
            actions.append(action)
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            # Skips the video if the files already exist
            print('Directory Already Exists, passing file')
            continue
            
        vid_loc = os.path.join(INPUT_VIDEO_PATH,vid)
        cap = cv2.VideoCapture(vid_loc)
        # Set mediapipe model 
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            # this is code for reading from a video file
            frame_num = 0
            while(cap.isOpened() and frame_num < sequence_length):
                # Read feed
                ret, frame = cap.read()
                
                if ret:
                    image, results = mediapipe_detection(frame, holistic)
                    # Draw landmarks
                    draw_styled_landmarks(image, results)
                    cv2.waitKey(1)
                    
                    # UNCOMMENT THIS TO SEE THE VIDEO DISPLAYED (THIS GREATLY INCREASES THE EXECUTION TIME OF THE PROGRAM)
                    # cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # # # Show to screen
                    # cv2.imshow('OpenCV Feed', image)
                    
                    # NEW Export keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)
                    frame_num += 1
                else:
                    break
                
        cap.release()
        cv2.destroyAllWindows()