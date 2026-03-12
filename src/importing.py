import cv2
import mediapipe as mp
import numpy as np
import os
from collections import deque
import time

# Name of the gesture you're collecting (change this as needed)
gesture_name = "fist"  # Change for different gestures
samples_needed = 50  # How many samples you want to collect
collected = []  # Store collected gesture data

# Set up MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Camera setup
cap = cv2.VideoCapture(0)  # Use your webcam
buffer = deque(maxlen=30)  # Number of frames per gesture

# Make sure you have the folder to save the data
os.makedirs("data", exist_ok=True)

print("Start performing the gesture...")

while len(collected) < samples_needed:
    ret, frame = cap.read()  # Capture frame from webcam
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to RGB for MediaPipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image)

    # Check if hands are detected
    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        landmarks = []

        # Collect landmarks for each hand detected
        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])  # Store x, y, z

        buffer.append(landmarks)  # Add to buffer

        # Draw the hand landmarks on the image for visual feedback
        mp.solutions.drawing_utils.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    # Show the image with landmarks (the webcam feed)
    cv2.imshow("Gesture Collection - Press 'c' to Capture, 'q' to Quit", frame)

    # Capture frame when 'c' is pressed
    if cv2.waitKey(1) & 0xFF == ord('c'):
        # Save the current frame as an image (e.g., "gesture_capture.jpg")
        cv2.imwrite(os.path.join(DATA_DIR, f"{gesture_name}_capture.jpg"), frame)
        print(f"Frame captured and saved as {gesture_name}_capture.jpg")

    # Once you have enough frames, save the gesture
    if len(buffer) == 30:  # 30 frames collected
        collected.append(np.array(buffer))  # Append the collected data
        print(f"Collected {len(collected)}/{samples_needed}")
        buffer.clear()  # Clear the buffer after each gesture
        time.sleep(1)  # Pause for a moment between gestures

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the collected data as a .npy file
np.save(os.path.join(DATA_DIR, f"{gesture_name}.npy"), np.array(collected))

# Clean up
cap.release()
cv2.destroyAllWindows()

print("Data collection complete!")
