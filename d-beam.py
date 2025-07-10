import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import pyautogui

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model("\\mp_hand_gesture")

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()

# Initialize the webcam
cap = cv2.VideoCapture(0)

# temp = 1
# map2 = {"thumbs up":1,'peace':2, 'okay':3,'thumbs down':4, 'call me':5,'stop':6, 'rock':7,'live long':8,'fist':9, 'smile':10}

actions = {
"thumbs up": (328, 776, 1),
"peace": (428, 776, 2),
"okay": (528, 776, 3),
"thumbs down":(628,776, 4)
}

last_gesture = None

while True:
    # Read each frame from the webcam
    _, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # print(result)
    
    className = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gesture
            prediction = model.predict([landmarks])
            print(prediction)
            classID = np.argmax(prediction)
            className = classNames[classID]
        
    if className in actions and className != last_gesture:
        x, y, temp = actions[className]
        pyautogui.click(x, y, button='left')
        cv2.putText(frame, "CURRENT TRACK: "+str(temp), (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2, cv2.LINE_AA)
        last_gesture = className

    # Show the final output
    cv2.imshow("Output", frame) 
    
    if cv2.waitKey(1) == 27:
        break

cap.release()

cv2.destroyAllWindows()


