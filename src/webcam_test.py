import cv2
import mediapipe as mp
import os
import numpy as np

def test_camera_and_mediapipe():
    # Model file ka path set karna
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'hand_landmarker.task')
    
    if not os.path.exists(model_path):
        print(f"Error: Please download 'hand_landmarker.task' and place it in the models folder.")
        return

    # New Tasks API Configuration
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    # Custom function points draw karne ke liye (Bypassing solutions.drawing_utils)
    def draw_dots(image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks
        h, w, _ = image.shape
        for hand_landmarks in hand_landmarks_list:
            for landmark in hand_landmarks:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)

    # Options setup for LIVE STREAM / VIDEO mode
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    print("Webcam successfully initialized. Press 'q' to close the window.")
    
    # Initialize the tracker using the downloaded model file
    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            # MediaPipe tasks requires standard RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Generate a continuous timestamp for the video tracking frame
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC)) + 1
            if timestamp_ms <= 0:
                timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

            # Process hand detection
            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            # Draw custom circles using pure OpenCV
            if detection_result.hand_landmarks:
                draw_dots(frame, detection_result)

            cv2.imshow('PSL Platform - New Tasks API Test', frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera_and_mediapipe()
