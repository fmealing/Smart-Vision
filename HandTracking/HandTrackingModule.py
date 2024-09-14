import time

import cv2 as cv
import mediapipe as mp


# Class to handle hand detection
class handDetector:
    def __init__(
        self,
        mode=False,
        maxHands=2,
        detectionCon=0.5,
        trackCon=0.5,
    ):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,  # Explicitly name the parameters
            max_num_hands=self.maxHands,
            min_detection_confidence=float(self.detectionCon),  # Cast to float
            min_tracking_confidence=float(self.trackCon),  # Cast to float
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv.cvtColor(
            img, cv.COLOR_BGR2RGB
        )  # Convert image to RGB for mediapipe
        self.results = self.hands.process(
            imgRGB
        )  # Process the image for hand landmarks

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # Draw hand landmarks
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )

        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

                if draw:
                    # Draw a circle on the landmark
                    cv.circle(img, (cx, cy), 7, (255, 0, 255), cv.FILLED)

        return lmList


# Main function to capture video and apply hand detection
def main():
    pTime = 0
    cTime = 0
    cap = cv.VideoCapture(0)  # Open the camera

    detector = handDetector()

    while True:
        success, img = cap.read()  # Read frame from the camera
        if not success:
            print("Failed to capture image.")
            break

        img = detector.findHands(img)  # Detect hands in the image
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])  # Print the position of the tip of the index finger

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # Display FPS on the image
        cv.putText(
            img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3
        )

        # Show the image
        cv.imshow("Image", img)

        # Break the loop if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
