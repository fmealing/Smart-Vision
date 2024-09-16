import time
import cv2 as cv
import numpy as np
import PoseEstimationModule as pm

detector = pm.poseDetector()
count = 0
direction = 0

##############################################
wCam, hCam = 1280, 720
##############################################

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0


def draw_text_with_background(
    img, text, position, font, font_scale, text_color, bg_color, thickness=1, padding=5
):
    """Function to draw text with a background rectangle"""
    text_size = cv.getTextSize(text, font, font_scale, thickness)[0]
    x, y = position
    bg_top_left = (x, y)
    bg_bottom_right = (x + text_size[0] + padding * 2, y + text_size[1] + padding * 2)
    cv.rectangle(img, bg_top_left, bg_bottom_right, bg_color, -1)  # Filled rectangle
    cv.putText(
        img,
        text,
        (x + padding, y + text_size[1] + padding),
        font,
        font_scale,
        text_color,
        thickness,
    )


while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # Left Arm Curl: Get the angle between shoulder, elbow, and wrist
        angle = detector.findAngle(img, 11, 13, 15)
        # Map the angle to a percentage (e.g., 20° to 170° mapped to 0% to 100%)
        per = np.interp(angle, (20, 170), (0, 100))
        # Calculate the length of the progress bar based on the percentage
        bar = np.interp(angle, (20, 170), (650, 100))

        # Check for dumbbell curl completion
        if per == 100:
            if direction == 0:
                count += 0.5
                direction = 1
        if per == 0:
            if direction == 1:
                count += 0.5
                direction = 0

        # Display the number of reps
        draw_text_with_background(
            img,
            f"Reps: {int(count)}",
            (50, 100),
            cv.FONT_HERSHEY_SIMPLEX,
            3,
            (255, 255, 255),
            (0, 0, 255),  # Red background for reps
            thickness=2,
            padding=10,
        )

        # Draw a progress bar on the side to show how far along the rep is
        cv.rectangle(
            img, (1100, 100), (1175, 650), (0, 255, 0), 3
        )  # Outline of the progress bar
        cv.rectangle(
            img, (1100, int(bar)), (1175, 650), (0, 255, 0), cv.FILLED
        )  # Filled progress bar
        # Display percentage inside the progress bar
        draw_text_with_background(
            img,
            f"{int(per)}%",
            (1100, 75),
            cv.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            (0, 0, 255),
            thickness=2,
            padding=10,
        )

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    draw_text_with_background(
        img,
        f"FPS: {int(fps)}",
        (50, 200),
        cv.FONT_HERSHEY_SIMPLEX,
        2,
        (255, 255, 255),
        (0, 0, 0, 128),  # Black semi-transparent background for FPS
        thickness=2,
        padding=10,
    )

    # Show the resulting image
    cv.imshow("Image", img)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
