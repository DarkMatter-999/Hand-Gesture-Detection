import argparse
import csv
import cv2
import itertools
import mediapipe as mp
import numpy as np
import time

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0)
args = parser.parse_args()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

prev_frame_time = 0
new_frame_time = 0

read = False

# For webcam input:
cap = cv2.VideoCapture(args.device)
with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        key = cv2.waitKey(10)
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                landmark_list = calc_landmark_list(image, hand_landmarks)

                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                pre_processed_landmark_list = np.array(pre_processed_landmark_list)
                np.set_printoptions(precision=2)
                
                if key == ord("t"):
                    read = not read
                    print(read)
                
                if ord('0') <= key <= ord('9') or ord('A') <= key <= ord('Z'):
                    number = key - 48
                    print(number, read)
                    if read:
                        logging_csv(number, pre_processed_landmark_list)

        new_frame_time = time.time()

        fps = 1//(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(fps)

        # Flip the image horizontally for a selfie-view display.
        image = cv2.flip(image, 1)

        cv2.putText(image, fps, (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('Hand Gesture Detection', image)
        if key & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()