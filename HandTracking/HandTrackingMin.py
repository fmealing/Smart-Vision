import time

import cv2 as cv
import mediapipe as mp

# Open the camera
cap = cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()

    # Convert the image to RGB
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Process the image
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 0:
                    cv.circle(img, (cx, cy), 15, (255, 0, 255), cv.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # Display the frame
    cv.imshow("Image", img)

    # WaitKey(1) returns -1 if no key is pressed; it will refresh the window continuously
    if cv.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
        break

# Release the camera and close all OpenCV windows
cap.release()
cv.destroyAllWindows()
