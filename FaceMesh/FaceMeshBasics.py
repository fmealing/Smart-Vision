import time

import cv2 as cv
import mediapipe as mp

# Initialize video capture
cap = cv.VideoCapture(0)
pTime = 0

# Initialize MediaPipe drawing and face mesh modules
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh

# Create FaceMesh object with proper parameters
faceMesh = mpFaceMesh.FaceMesh(
    max_num_faces=1,  # Maximum number of faces to detect
    refine_landmarks=True,  # Enables iris tracking
    min_detection_confidence=0.5,  # Confidence threshold for detection
    min_tracking_confidence=0.5,  # Confidence threshold for tracking
)

# Drawing specifications
drawSpec = mpDraw.DrawingSpec(color=(3, 160, 98), thickness=1, circle_radius=2)

while True:
    success, img = cap.read()
    if not success:
        break

    # Convert the image to RGB for processing with MediaPipe
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    # Draw face landmarks if detected
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            # Use FACEMESH_TESSELATION and FACEMESH_CONTOURS instead of FACE_CONNECTIONS
            mpDraw.draw_landmarks(
                image=img,
                landmark_list=faceLms,
                connections=mpFaceMesh.FACEMESH_TESSELATION,  # Use tessellation for face mesh
                landmark_drawing_spec=drawSpec,  # Customize landmark appearance
                connection_drawing_spec=drawSpec,  # Customize connection appearance
            )

            mpDraw.draw_landmarks(
                image=img,
                landmark_list=faceLms,
                connections=mpFaceMesh.FACEMESH_CONTOURS,  # Use contours for facial outlines
                landmark_drawing_spec=drawSpec,
                connection_drawing_spec=drawSpec,
            )

            # Optionally draw irises
            mpDraw.draw_landmarks(
                image=img,
                landmark_list=faceLms,
                connections=mpFaceMesh.FACEMESH_IRISES,
                landmark_drawing_spec=drawSpec,
                connection_drawing_spec=drawSpec,
            )

            for id, lm in enumerate(faceLms.landmark):
                ih, iw, ic = img.shape
                x, y = int(lm.x * iw), int(lm.y * ih)

    # Calculate Frames Per Second (FPS)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Display FPS on the image
    cv.putText(
        img, f"FPS: {int(fps)}", (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3
    )

    # Show the resulting image
    cv.imshow("Image", img)

    # Exit when 'q' key is pressed
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv.destroyAllWindows()
