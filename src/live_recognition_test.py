import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import Counter, deque

# Load model, scaler, and label encoder
with open('model/gesture_model_v2.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/gesture_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Webcam setup
cap = cv2.VideoCapture(0)

sequence = []
prediction_history = deque(maxlen=10)  # Store last 10 predictions

SEQ_LENGTH = 3  # Frames used per prediction

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            sequence.append(landmarks)

            if len(sequence) == SEQ_LENGTH:
                input_data = np.array(sequence).flatten().reshape(1, -1)
                input_data = scaler.transform(input_data)
                prediction = model.predict(input_data)
                predicted_label = label_encoder.inverse_transform(prediction)[0]

                prediction_history.append(predicted_label)  # Save prediction
                sequence = []  # Reset sequence buffer

    # Majority vote from prediction history
    if prediction_history:
        most_common_prediction = Counter(prediction_history).most_common(1)[0][0]
        cv2.putText(frame, f'Gesture: {most_common_prediction}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Live Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
