# Smart Vision – Real-Time Computer Vision Applications

A collection of real-time computer vision systems built using **OpenCV** and **MediaPipe**.  
This project explores pose estimation, hand tracking, facial analysis, and gesture-based interaction in live webcam environments.

All applications run in real time using a standard webcam and demonstrate modular CV system design.

---

## Overview

This repository contains multiple independent computer vision applications:

- AI Personal Trainer (Pose-based rep counter)
- Face Detection
- Face Mesh Tracking
- Hand Tracking
- Finger Counter
- Gesture-Based Volume Control
- Pose Estimation Module

Each module is structured with reusable detection classes and a real-time execution script.

---

## Project Structure

```
Smart-Vision/
│
├── AI Personal Trainer/
├── FaceDetection/
├── FaceMesh/
├── Finger Counter/
├── Gesture Volume Control/
├── HandTracking/
├── PoseEstimation/
└── README.md
```

Each folder contains:

- A main execution script
- A reusable detection module class (where applicable)

---

# Applications

---

## 1. AI Personal Trainer

Real-time dumbbell curl repetition counter using pose estimation.

### Key Features

- Uses MediaPipe Pose model
- Calculates joint angles (shoulder → elbow → wrist)
- Maps angle range to repetition percentage
- Direction-based rep counting logic
- Real-time FPS monitoring
- On-screen progress bar

### Technical Approach

- Angle calculated using vector orientation between 3 landmarks
- Angle mapped from (20°–170°) to 0–100% completion
- Rep counted when movement direction reverses
- Visual feedback drawn directly on frame

This demonstrates real-time biomechanical analysis using vision-based tracking.

Run with:

```
python AI Personal Trainer/AITrainer.py
```

---

## 2. Face Detection

Real-time face detection using MediaPipe Face Detection model.

### Features

- Bounding box detection
- Confidence score display
- FPS tracking
- Modular `FaceDetector` class implementation

Demonstrates object detection pipeline using RGB frame processing.

Run with:

```
python FaceDetection/main.py
```

---

## 3. Face Mesh

468-point facial landmark detection with iris tracking.

### Features

- Face mesh tessellation
- Facial contour detection
- Iris landmark detection
- Custom drawing specifications
- Real-time FPS monitoring

This application demonstrates dense landmark tracking and geometric facial mapping.

Run with:

```
python FaceMesh/main.py
```

---

## 4. Hand Tracking

Real-time hand landmark tracking using MediaPipe Hands.

### Features

- Multi-hand detection
- 21-point landmark tracking
- Custom drawing overlays
- Modular `HandDetector` class

Demonstrates spatial landmark extraction in live video.

Run with:

```
python HandTracking/main.py
```

---

## 5. Finger Counter

Counts number of extended fingers using landmark geometry.

### Technical Approach

- Compares tip landmark positions to intermediate joints
- Uses x-axis logic for thumb
- Uses y-axis logic for other fingers
- Counts binary finger states
- Displays total fingers in real time

This demonstrates gesture logic derived from geometric landmark relationships.

Run with:

```
python "Finger Counter/main.py"
```

---

## 6. Gesture Volume Control (macOS)

Controls system volume using thumb-index pinch distance.

### Technical Approach

- Measures Euclidean distance between landmarks 4 and 8
- Maps distance range to volume (0–100)
- Uses `osascript` to control macOS volume
- Visual feedback when fingers pinch

Demonstrates:

- Gesture-to-action mapping
- Interfacing CV system with OS-level control
- Real-time signal mapping via interpolation

Run with:

```
python "Gesture Volume Control/main.py"
```

---

# Core Technologies

- Python 3
- OpenCV
- MediaPipe
- NumPy
- macOS AppleScript (volume control module)

---

# Design Philosophy

This project emphasizes:

- Modular detector classes
- Real-time performance monitoring
- Reusable landmark extraction logic
- Clear separation between detection and application logic
- Practical human-computer interaction use cases

Rather than building a single large system, this repository explores multiple focused CV pipelines.

---

# Future Improvements

- Multi-exercise recognition in AI Trainer
- Smoothing filters (Kalman / EMA) for angle stability
- Cross-platform volume control
- Gesture classification model instead of rule-based logic
- Performance benchmarking

---

# Installation

Install dependencies:

```
pip install opencv-python mediapipe numpy
```

Then run any module from its directory.

---

# Author

Florian Mealing  
MEng Mechatronic & Robotic Engineering  
University of Birmingham
