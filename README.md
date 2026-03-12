# gesture-recog
Real-time hand gesture recognition ML project

# Overview
  This project is a real-time hand gesture recognition system using 
  MediaPipe and a custom-trained machine learning model. It collects hand
  landmark data, applies data augmentation, trains a classifier, and
  recognizes gestures live via webcam.

# Features
  - Collect hand gesture data
  - Augment data (flip, rotate, scale, shift)
  - Train ML model for gesture classification
  - Real-time gesture recognition with webcam
  - Majority vote smoothing for stable predictions

# Folder Structure
```
New_Data/
├── data/ # Raw and augmented gesture data (.npy files)
│ ├── augmented/ # Augmented gesture data
│ ├── fist.npy
│ ├── OK.npy
│ ├── open_palm.npy
│ ├── peace_sign.npy
│ ├── point_up.npy
│ └── thumbs_up.npy
├── model/ # Trained ML models and related files
│ ├── gesture_model_v2.pkl
│ ├── gesture_scaler.pkl
│ └── label_encoder.pkl
├── src/ # Source code
│ ├── importing.py # Gesture data collection
│ ├── augmenting.py # Data augmentation
│ └── live_recognition_test.py # Live gesture recognition
├── .gitignore
└── README.md
```

# Setup
1. Clone the repository:
      git clone https://github.com/falisha-a/gesture-recog.git
      cd gesture-recog/New_Data
2. Install dependencies:
      pip install opencv-python mediapipe numpy scikit-learn
3. Make sure the data/ and model/ folders are in place.

# Usage
Collect gestures
  python src/importing.py
  Press 'c' to capture frames and 'q' to quit.
Augment data
  python src/augmenting.py

Run live recognition
  python src/live_recognition_test.py
  Press 'q' to exit.

Notes

Folder paths use cross-platform os.path.join for compatibility.

The project uses MediaPipe Hands for landmark detection.

Model files in model/ include the trained classifier, scaler, and label encoder.
