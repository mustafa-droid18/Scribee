import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from matplotlib import pyplot as plt
import mediapipe as mp # Go through Media pipe
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense,Dropout
from fileinput import filename
from PIL import Image, ImageDraw
import os
import requests
import time
from audio_recorder_streamlit import audio_recorder
from  spellchecker import SpellChecker

API_KEY_ASSEMBLYAI = '31d08ebfe16243d1b87ae65e76d2d95c' #API key provided by AssemblyAI for access

upload_endpoint = 'https://api.assemblyai.com/v2/upload'
transcript_endpoint = 'https://api.assemblyai.com/v2/transcript'

headers_auth_only = {'authorization': API_KEY_ASSEMBLYAI}

headers = {
    "authorization": API_KEY_ASSEMBLYAI,
    "content-type": "application/json"
}

CHUNK_SIZE = 5_242_880  # 5MB


def upload(filename):
    def read_file(filename):
        with open(filename, 'rb') as f:
            while True:
                data = f.read(CHUNK_SIZE)
                if not data:
                    break
                yield data

    upload_response = requests.post(upload_endpoint, headers=headers_auth_only, data=read_file(filename))
    return upload_response.json()['upload_url']


def transcribe(audio_url):
    transcript_request = {
        'audio_url': audio_url
    }

    transcript_response = requests.post(transcript_endpoint, json=transcript_request, headers=headers)
    return transcript_response.json()['id']

        
def poll(transcript_id):
    polling_endpoint = transcript_endpoint + '/' + transcript_id
    polling_response = requests.get(polling_endpoint, headers=headers)
    return polling_response.json()


def get_transcription_result_url(url):
    transcribe_id = transcribe(url)
    while True:
        data = poll(transcribe_id)
        if data['status'] == 'completed':
            return data, None
        elif data['status'] == 'error':
            return data, data['error']
            
        print("waiting for 20 seconds") #cooldown period of 20 seconds before the code sends data for conversion to the API again
        time.sleep(20)
        
  #saving the transcribed text in a txt file      
def save_transcript(url, title):
    data, error = get_transcription_result_url(url)
    
    if data:
        filename = title + '.txt'
        with open(filename, 'w') as f:
            f.write(data['text'])
        print('Transcript saved')
    elif error:
        print("Error!!!", error)

Dictionary = {
    " ": "000000",
    "a": "100000",
    "b": "110000",
    "c": "100100",
    "d": "100110",
    "e": "100010",
    "f": "110100",
    "g": "110110",
    "h": "110010",
    "i": "010100",
    "j": "010110",
    "k": "101000",
    "l": "111000",
    "m": "101100",
    "n": "101110",
    "o": "101010",
    "p": "111100",
    "q": "111110",
    "r": "111010",
    "s": "011100",
    "t": "011110",
    "u": "101001",
    "v": "111001",
    "w": "010111",
    "x": "101101",
    "y": "101111",
    "z": "101011",
    "0": "010110",
    "1": "100000",
    "2": "110000",
    "3": "100100",
    "4": "100110",
    "5": "100010",
    "6": "110100",
    "7": "110110",
    "8": "110010",
    "9": "010100",
    "-": "001001",
    ":": "010010",
    ".": "010011",
    ",": "010000",
    "'": "001000",
    "!": "011010",
    "?": "011001",
    ";": "011000",
    "@": "000000",
    "NUMERIC": "001111",
    "CAPITAL": "000001"
    #"#": "001111",
    #"+": "001101",
    #"*": "100001",
    #"=": "111111",
    #"<": "110001",
    #">": "001110",
    #"(": "111011",
    #")": "011111",
}

FONT_SIZE = 5
FONT_COLOR = "black"
MARGIN = 75
X_SPACING, Y_SPACING = 6, 20
PAPER_COLOR = "white"
PAPER_WIDTH = 850
PAPER_HEIGHT = 1100

class Character:
    def __init__(self, b_code, pixel_size=FONT_SIZE):

        b_code = str(b_code)  # Turn into a list
        self._list = "000000000000000"  # List that represents which pixels to fill, is a 3 * 5 rectangle

        self.pixel_size = pixel_size
        self.width = 3
        self.height = 5

        self.pixels = [[0 for n in range(3)].copy() for i in range(5)]  # List to be used by Paper to display a char

        # Update display_list to match the braille code, see http://braillebug.org/braille_deciphering.asp
        self.pixels[0][0] = int(float(b_code[0]))
        self.pixels[2][0] = int(float(b_code[1]))
        self.pixels[4][0] = int(float(b_code[2]))
        self.pixels[0][2] = int(float(b_code[3]))
        self.pixels[2][2] = int(float(b_code[4]))
        self.pixels[4][2] = int(float(b_code[5]))

class Paper:
    def __init__(self, name, page=1, charset=False, width=PAPER_WIDTH, height=PAPER_HEIGHT, color=PAPER_COLOR):
        """Creates an image that is associated with the object"""

        self.name = str(name)  # Name to be used when saving the file
        self._height = height
        self._width = width
        self._color = color
        self._page = page

        if charset:
            self.charset = charset
        else:
            self.charset = Dictionary  # Converts the dictionary into one with objects
            for x in self.charset:
                self.charset[x] = Character(self.charset[x])

        self._clear()  # Sets the image to a blank page

    def _clear(self):
        """Resets the image to its default color"""

        self.image = Image.new("RGB", (self._width, self._height), self._color)

    def draw(self, x, y, dx, dy, color):
        """Draws a colored rectangle onto the image using the coordinates of the top left and its size"""

        draw = ImageDraw.Draw(self.image)

        draw.rectangle([(x,y),(dx,dy)], color, outline=None)

    def save(self):
        """Saves the image to a physical file that is the name the object was created with"""

        self.image.save("./output/" + self.name + " pg" + str(self._page) + ".png")

    def show(self):
        """Opens the image in whatever is your system default, doesn't require saving"""

        self.image.show()

    def drawChar(self, char, x, y, color=FONT_COLOR):
        """Takes a Character object and draws it on the image at the given coordinates using parameters inside the
        Character"""

        pixels, width, height = char.pixels, char.width, char.height
        pixel_size = char.pixel_size
        dx, dy = 0, 0

        # Loops though the character's list that specifies where to draw
        for row in range(char.height):

            for column in range(char.width):

                if pixels[row][column]:  # If there is a 1 at the specified index in the char, draw a pixel(s)
                    self.draw(x + dx, y + dy, x + dx + pixel_size, y + dy + pixel_size, color)

                dx += pixel_size + 1  # Increase the horizontal offset

            dy += pixel_size + 1  # Increase the vertical offset
            dx = 0  # Reset the horizontal offset

    def convertBrailleCharacter(self, string):
        # First convert the string into braille letters
        braille_code = []

        numeric_conditions = False
        for letter in string:

            if letter.isupper():  # Checks for special cases
                braille_code.append(self.charset["CAPITAL"])
                braille_code.append(self.charset[letter.lower()])
                numeric_conditions = False

            elif letter.isnumeric():
                if not numeric_conditions:
                    braille_code.append(self.charset["NUMERIC"])
                    numeric_conditions = True

                braille_code.append(self.charset[letter])

            else:  # Normal condition
                braille_code.append(self.charset[letter])
                numeric_conditions = False

        return braille_code

    def drawSentence(self, braille_code, x=MARGIN, y=MARGIN,
                     wrap_width=(PAPER_WIDTH - (MARGIN * 2)),
                     x_spacing=X_SPACING,
                     y_spacing=Y_SPACING,
                     color=FONT_COLOR):
        """Draws a sentence starting at a point, wraps after passing a specified width
        (relative to the left edge of paper), requires a list of braille character objects, will create multiple pages"""

        dx, dy = 0, 0
        character_width = FONT_SIZE * 3 + x_spacing * 2
        character_height = FONT_SIZE * 5 + y_spacing

        # Displaying the letters
        for n in range(len(braille_code)):

            character = braille_code[n]
            self.drawChar(character, x + dx, y + dy, color)

            if dx + character_width >= wrap_width:  # If it has hit the right margin, wrap
                dx = 0
                dy += character_height
            else:
                dx += character_width  # Move to next char

            if dy + character_height >= PAPER_HEIGHT - MARGIN * 2:  # If it hits the end of the page
                # Make a new Paper object, have it draw remaining chars
                next_page = Paper(self.name, (self._page + 1), self.charset)
                next_page.drawSentence(braille_code[n:], x, y, wrap_width, x_spacing, y_spacing, color)
                break

        self.save()

def getInput(file):  # To open the file
    input_text = open(file, "r")
    return input_text.read().replace('\n', '')

def printing(name):
    filename=name
    audio_url = upload(filename)
    save_transcript(audio_url, 'file_title')

    st.sidebar.text('Original Audio')
    st.sidebar.audio(filename)

    ## Dashboard
    paper_name = name
    text = getInput("file_title.txt")
    for letter in text:
        if Dictionary.get(letter) is None and not letter.isupper():  # Make sure text is valid
            print("'" + letter + "' is an invalid character.""\nFix text and run the program again.")
            input("Press Enter to exit...")
            quit()  # If file_title.txt has invalid char, exit program

    if not os.path.isdir("./output"):  # If output folder doesn't exist, make one
        os.mkdir("./output")
    paper = Paper(paper_name)
    print("=" * 25 + "\nWorking...")
    paper.drawSentence(paper.convertBrailleCharacter(text))
    i=1
    while True:
        image="./output/"+ name +' pg'+str(i)+'.png'
        st.subheader('Output Image'+str(i))
        out_image=Image.open(image)
        st.image(out_image, use_column_width=True)
        i=i+1
        try: 
            Image.open("./output/"+ name +' pg'+str(i)+'.png')
        except:            
            break
    return

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
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.load_weights('Trained_on_all_letters.h5')


st.title('Scribee!')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Create Sidebar
st.sidebar.title('Sidebar')
st.sidebar.subheader('Parameter')

# Define available pages in selection box
app_mode = st.sidebar.selectbox(
    'App Mode',
    ['About','Speech To Braille','Video']
)

# Resize Images to fit Container
@st.cache()
# Get Image Dimensions
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h,w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = width/float(w)
        dim = (int(w*r),height)

    else:
        r = width/float(w)
        dim = width, int(h*r)

    # Resize image
    resized = cv2.resize(image,dim,interpolation=inter)

    return resized

# About Page

if app_mode == 'About':
    st.markdown('''
                ## About \n
                Scribee! is your personal companion to learn and understand Sign Language And Braille Translation.\n
                Our goal is helping visually and verbally differently abled people gain better access to education, politics, media, and entertainment which is available in the audio-video format.\n
                **StreamLit** is used to create the Web Graphical User Interface (GUI) \n

                
                - [Github](https://github.com/mustafa-droid18/Scribee) \n
    ''')

## Add Sidebar and Window style
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html=True,
)

elif app_mode == 'Speech To Braille':
    
    st.sidebar.markdown('---')

    ## Add Sidebar and Window style
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width: 350px
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width: 350px
            margin-left: -350px
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    record = st.sidebar.checkbox(label="Record Audio")
    st.sidebar.markdown('---')
    uploaded_filename= st.sidebar.file_uploader("Upload an Audio", type=["wav","mp3"])
    filename=''
    if record:
        audio_bytes = audio_recorder()
        wav_file= None
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            wav_file = open("audio.wav", "wb")
            wav_file.write(audio_bytes)
            printing('audio.wav')
    if uploaded_filename:
        wav_file = open(uploaded_filename.name, "wb")
        wav_file.write(uploaded_filename.getvalue())
        printing(uploaded_filename.name)
    else:
        st.markdown(''' 
        ## Speech to Braille Conversion\n
        **Please choose one of the options mentioned in the sidebar**   ''')
        

elif app_mode == 'Video':

    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Use Webcam')
    #record = st.sidebar.checkbox("Record Video")

    #if record:
        #st.checkbox('Recording', True)

    #st.sidebar.markdown('---')

    ## Add Sidebar and Window style
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width: 350px
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width: 350px
            margin-left: -350px
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0,max_value=1.0,value=0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value=0.0,max_value=1.0,value=0.5)
    st.sidebar.markdown('---')

    ## Get Video
    stframe = st.empty()
    video = cv2.VideoCapture(0)

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(video.get(cv2.CAP_PROP_FPS))

    
    sequence = []
    sentence=[]
    predictions=[]
    no_hand_count = 0
    threshold = 0.5
    say_check = False
    cap = cv2.VideoCapture(0)
    letter = ""
    pre_letter = ""
    word=""
    word1=""

    kpil, kpil2 = st.columns(2)

    with kpil:
        st.markdown('**Word**')
        kpil_text = st.markdown('')

    with kpil2:
        st.markdown('**Letter**')
        kpil2_text = st.markdown('')

    #with kpil3:
        #st.markdown('**Image Resolution**')
        #kpil3_text = st.markdown('0')

    st.markdown('<hr/>', unsafe_allow_html=True)

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
                    #word=SpellChecker().correction(word1)
                    if no_hand_count>15:
                        word1=''
                        word =''
                        kpil_text.write(f"<h1 style='text-align: center; color:red;'>{word}</h1>", unsafe_allow_html=True)
                        kpil2_text.write(f"<h1 style='text-align: center; color:red;'>{letter}</h1>", unsafe_allow_html=True)
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
                        sentence.append(letter)
                        word1=word1+letter
                        word=SpellChecker().correction(word1)
                        word=word.upper()
                        kpil_text.write(f"<h1 style='text-align: center; color:red;'>{word}</h1>", unsafe_allow_html=True)
                        kpil2_text.write(f"<h1 style='text-align: center; color:red;'>{letter}</h1>", unsafe_allow_html=True)
           

            # Dashboard
            #kpil_text.write(f"<h1 style='text-align: center; color:red;'>{int(fps)}</h1>", unsafe_allow_html=True)
            #kpil2_text.write(f"<h1 style='text-align: center; color:red;'>{letter}</h1>", unsafe_allow_html=True)
            #kpil3_text.write(f"<h1 style='text-align: center; color:red;'>{width*height}</h1>",unsafe_allow_html=True)

            frame = cv2.resize(frame,(0,0), fx=0.8, fy=0.8)
            frame = image_resize(image=frame, width=640)
            stframe.image(frame,channels='BGR', use_column_width=True)
    cv2.VideoCapture(0).release()
    cv2.destroyAllWindows()
