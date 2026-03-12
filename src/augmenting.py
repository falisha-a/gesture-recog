
import numpy as np
import os
import random
import math

# Config
DATA_DIR = "data"
AUG_DIR = os.path.join(DATA_DIR, "augmented")
os.makedirs(AUG_DIR, exist_ok=True)

def flip_landmarks(landmarks):
    flipped_landmarks = []
    for i in range(0, len(landmarks), 3):  # Step by 3 to handle x, y, z for each landmark
        x, y, z = landmarks[i], landmarks[i+1], landmarks[i+2]
        flipped_landmarks.extend([1 - x, y, z])  # Flip x-coordinate
    return np.array(flipped_landmarks)

def add_noise(landmarks, noise_strength=0.01):
    noise = np.random.normal(0, noise_strength, landmarks.shape)
    return landmarks + noise

def scale_shift(landmarks, scale=(1, 1, 1), shift=(0, 0, 0)):
    reshaped_landmarks = landmarks.reshape(-1, 3)
    scaled_shifted_landmarks = reshaped_landmarks * scale + shift
    return scaled_shifted_landmarks.flatten()

def rotate_landmarks(landmarks, angle):
    angle = math.radians(angle)  # Convert angle to radians
    rotation_matrix = np.array([
        [math.cos(angle), -math.sin(angle), 0],
        [math.sin(angle), math.cos(angle), 0],
        [0, 0, 1]
    ])
    
    rotated_landmarks = []
    for x, y, z in landmarks.reshape(-1, 3):
        rotated_point = np.dot(rotation_matrix, [x, y, z])
        rotated_landmarks.extend(rotated_point)
    
    return np.array(rotated_landmarks)

def augment(landmarks):
    aug = []
    
    # Flip the landmarks
    aug.append(flip_landmarks(landmarks))
    
    # Rotate the landmarks by random angles
    aug.append(rotate_landmarks(landmarks, random.uniform(-30, 30)))  # Random rotation between -30 and 30 degrees
    
    # Apply random scaling and shifting
    scale = (random.uniform(0.8, 1.2), random.uniform(0.8, 1.2), random.uniform(0.8, 1.2))  # Random scale for x, y, z
    shift = (random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1))  # Random shift for x, y, z
    aug.append(scale_shift(landmarks, scale=scale, shift=shift))
    
    return aug

# List of all gesture names
gesture_names = ["fist", "OK", "open_palm", "peace_sign", "point_up", "thumbs_up"]

# Augment and save for all gestures
for gesture_name in gesture_names:
    gesture_data = np.load(os.path.join(DATA_DIR, f"{gesture_name}.npy"), allow_pickle=True)
    
    augmented_data = []  # To store augmented data for the current gesture
    
    for gesture in gesture_data:
        for landmarks in gesture:
            augmented_data.append(augment(landmarks))  # Generate augmentations for each sample

    # Save each gesture's augmented data in the main augmented folder (without subfolders)
    np.save(os.path.join(AUG_DIR, f"{gesture_name}_augmented.npy"), np.array(augmented_data))
    print(f"Augmented data for {gesture_name} saved to {os.path.join(AUG_DIR, f'{gesture_name}_augmented.npy')}")

print("All augmented data saved!")
