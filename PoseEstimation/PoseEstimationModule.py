import time

import cv2 as cv
import mediapipe as mp


class poseDetector:
    def __init__(
        self,
        mode=False,
        model_complexity=1,
        smooth=True,
        detectionCon=0.5,
        trackCon=0.5,
    ):
        self.mode = mode
        self.model_complexity = model_complexity
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=self.model_complexity,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
        )

    def findPose(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(
                img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS
            )

        return img

    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

                if draw:
                    cv.circle(img, (cx, cy), 5, (255, 0, 0), cv.FILLED)

        return lmList


def main():
    cap = cv.VideoCapture(0)
    pTime = 0
    detector = poseDetector()

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            break

        img = detector.findPose(img)
        lmList = detector.findPosition(img)

        if lmList:
            print(lmList[0])  # Print the first landmark for reference

        # Calculate and display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(
            img,
            f"FPS: {int(fps)}",
            (70, 50),
            cv.FONT_HERSHEY_PLAIN,
            3,
            (255, 0, 255),
            3,
        )

        # Show the image
        cv.imshow("Image", img)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
