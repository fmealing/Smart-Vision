import time

import cv2 as cv
import mediapipe as mp


class FaceMeshModule:
    def __init__(
        self,
        max_faces=1,
        refine_landmarks=True,
        detection_confidence=0.5,
        tracking_confidence=0.5,
        draw_color=(3, 160, 98),
        thickness=1,
        circle_radius=2,
    ):
        # Initialize FaceMesh with given parameters
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        # Set drawing specifications
        self.drawSpec = self.mpDraw.DrawingSpec(
            color=draw_color, thickness=thickness, circle_radius=circle_radius
        )

    def process_frame(self, img):
        """
        Process a single frame for face mesh detection.
        Args:
            img: The input image frame (BGR).

        Returns:
            The image with face landmarks drawn (if detected).
        """
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                # Draw tessellation and contours
                self.mpDraw.draw_landmarks(
                    image=img,
                    landmark_list=faceLms,
                    connections=self.mpFaceMesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.drawSpec,
                    connection_drawing_spec=self.drawSpec,
                )

                self.mpDraw.draw_landmarks(
                    image=img,
                    landmark_list=faceLms,
                    connections=self.mpFaceMesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=self.drawSpec,
                    connection_drawing_spec=self.drawSpec,
                )

                # Optionally draw irises
                self.mpDraw.draw_landmarks(
                    image=img,
                    landmark_list=faceLms,
                    connections=self.mpFaceMesh.FACEMESH_IRISES,
                    landmark_drawing_spec=self.drawSpec,
                    connection_drawing_spec=self.drawSpec,
                )

        return img


def calculate_fps(pTime, cTime):
    """
    Calculate Frames Per Second (FPS).

    Args:
        pTime: Previous time.
        cTime: Current time.

    Returns:
        fps: Frames per second.
    """
    fps = 1 / (cTime - pTime)
    return fps


def main():
    # Initialize the FaceMeshModule
    face_mesh_module = FaceMeshModule()

    # Initialize video capture
    cap = cv.VideoCapture(0)
    pTime = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        # Process the image frame
        img = face_mesh_module.process_frame(img)

        # Calculate FPS
        cTime = time.time()
        fps = calculate_fps(pTime, cTime)
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


if __name__ == "__main__":
    main()
