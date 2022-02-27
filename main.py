import argparse
import cv2
import mediapipe as mp
import numpy as np
import time

from utils import *

from model.keypoint_classifier import KeyPointClassifier

def draw_info(image, landmarks, info_text):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)
    
    if info_text:
        image = cv2.flip(image, 1)
        cv2.putText(image, info_text, (image.shape[1] - x  - w + 5, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        image = cv2.flip(image, 1)
        
    return image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    prev_frame_time = 0
    new_frame_time = 0

    # For webcam input:
    cap = cv2.VideoCapture(args.device)
    
    keypoint_classifier = KeyPointClassifier()
    with open('model/keypoint_classifier_label.csv',
            encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]

    with mp_hands.Hands(
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
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
                    confidence, hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                    if(confidence > 0.75):
                        info_text = keypoint_classifier_labels[hand_sign_id-1]
                    else:
                        info_text = None
                    image = draw_info(image, hand_landmarks, info_text)

            new_frame_time = time.time()

            fps = 1//(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            fps = str(fps)

            # Flip the image horizontally for a selfie-view display.
            image = cv2.flip(image, 1)

            cv2.putText(image, fps, (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('MediaPipe Hands Demo', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cap.release()

if __name__ == "__main__":
    main()