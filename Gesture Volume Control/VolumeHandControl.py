import math
import subprocess
import time

import cv2 as cv
import mediapipe as mp
import numpy as np

import HandTrackingModule as htm

######################################################
wCam, hCam = 640, 480
######################################################


def set_volume_mac(volume_level):
    # volume_level is between 0 and 100 (macOS uses 0-10 internally)
    volume_level = int(volume_level // 10)  # Convert 0-100 to 0-10
    subprocess.run(["osascript", "-e", f"set volume output volume {volume_level}"])


cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0

detector = htm.HandDetector(detectionCon=0.7)

# Volume range (0 to 100, mapped from hand range)
volumeRange = [0, 100]

while True:
    success, img = cap.read()
    if not success:
        break

    # Detect hands
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # Get thumb and index finger positions (landmark 4 is thumb tip, landmark 8 is index finger tip)
        x1, y1 = lmList[4][1], lmList[4][2]  # Thumb tip
        x2, y2 = lmList[8][1], lmList[8][2]  # Index finger tip

        # Calculate the center point between thumb and index finger
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw circles and line connecting thumb and index finger
        cv.circle(img, (x1, y1), 15, (255, 0, 255), cv.FILLED)
        cv.circle(img, (x2, y2), 15, (255, 0, 255), cv.FILLED)
        cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv.circle(img, (cx, cy), 15, (255, 0, 255), cv.FILLED)

        # Calculate the distance between thumb and index finger
        length = math.hypot(x2 - x1, y2 - y1)
        print(f"Finger distance: {length}")

        # Map the length to the volume range (hand distance: 50 to 300 mapped to volume: 0 to 100)
        lowerHandRange = 30
        upperHandRange = 90
        volume = np.interp(length, [lowerHandRange, upperHandRange], volumeRange)
        print(f"Volume Level: {volume}")

        # Set the macOS system volume based on the hand distance
        set_volume_mac(volume)

        # Visual cue when the thumb and index fingers are close (e.g., when they are pinched)
        if length < lowerHandRange:
            cv.circle(img, (cx, cy), 15, (0, 255, 0), cv.FILLED)

    # Calculate and display Frames Per Second (FPS)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(
        img, f"FPS: {int(fps)}", (10, 70), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2
    )

    # Show the resulting image
    cv.imshow("Image", img)

    # Exit when 'q' key is pressed
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv.destroyAllWindows()
