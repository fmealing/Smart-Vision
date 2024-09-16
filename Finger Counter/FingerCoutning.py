import time
import cv2 as cv
import HandTrackingModule as htm

##############################################
wCam, hCam = 640, 480
##############################################

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0

detector = htm.HandDetector(detectionCon=0.75)
tipIds = [4, 8, 12, 16, 20]


def draw_text_with_background(
    img, text, position, font, font_scale, text_color, bg_color, thickness=1, padding=5
):
    """Function to draw text with a background rectangle"""
    # Get the text size
    text_size = cv.getTextSize(text, font, font_scale, thickness)[0]
    # Define the rectangle's top-left and bottom-right points
    x, y = position
    bg_top_left = (x, y)
    bg_bottom_right = (x + text_size[0] + padding * 2, y + text_size[1] + padding * 2)
    # Draw the background rectangle (semi-transparent if desired)
    cv.rectangle(
        img, bg_top_left, bg_bottom_right, bg_color, -1
    )  # Thickness = -1 for filled rectangle
    # Draw the text on top of the rectangle
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
    img = detector.findHands(img)

    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        fingers = []

        # Thumb check (left/right hand check based on x-coordinates)
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Four Fingers (y-coordinate check to determine if fingers are folded or not)
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # Count how many fingers are up
        totalFingers = fingers.count(1)

        # Display the number of fingers with a background
        draw_text_with_background(
            img,
            f"Fingers: {totalFingers}",  # The text to display
            (10, 70),  # The position on the screen
            cv.FONT_HERSHEY_SIMPLEX,  # The font
            2,  # Font scale (size)
            (255, 255, 255),  # Text color (white)
            (0, 0, 0, 128),  # Background color (black with some transparency)
            thickness=2,  # Text thickness
            padding=10,  # Padding around the text
        )

    # Calculate and display FPS (also with a background)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    draw_text_with_background(
        img,
        f"FPS: {int(fps)}",
        (10, 150),
        cv.FONT_HERSHEY_SIMPLEX,
        1.5,
        (255, 255, 255),
        (0, 0, 255, 128),  # Red background with transparency
        thickness=2,
        padding=10,
    )

    # Show the resulting image
    cv.imshow("Image", img)

    # Exit when 'q' key is pressed
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
