import os
import cv2
import numpy as np
import mediapipe as mp

def extract_coordinates_from_dataset():
    # Paths Setup
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, 'models', 'hand_landmarker.task')
    raw_data_dir = os.path.join(base_dir, 'data', 'raw')
    processed_data_dir = os.path.join(base_dir, 'data', 'processed')
    os.makedirs(processed_data_dir, exist_ok=True)

    # Configuration for MediaPipe Tasks
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE, # Processing frame-by-frame as static images
        num_hands=2,
        min_hand_detection_confidence=0.5
    )

    dataset_features = []
    dataset_labels = []

    print("Starting Feature Extraction from raw videos...")

    with HandLandmarker.create_from_options(options) as landmarker:
        # Loop through each sign folder
        for sign_name in os.listdir(raw_data_dir):
            sign_folder = os.path.join(raw_data_dir, sign_name)
            if not os.path.isdir(sign_folder):
                continue

            print(f"Processing category: {sign_name}")
            
            # Loop through videos/images inside the sign folder
            for file_name in os.listdir(sign_folder):
                file_path = os.path.join(sign_folder, file_name)
                
                # Reading image/frame
                frame = cv2.imread(file_path)
                if frame is None:
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                # Detect landmarks
                detection_result = landmarker.detect(mp_image)

                # If hands detected, flatten the 21 coordinates (X, Y, Z) for both hands
                if detection_result.hand_landmarks:
                    # Initialize empty array for 2 hands (2 hands * 21 landmarks * 3 coordinates = 126 values)
                    hand_matrix = np.zeros((2, 21, 3))
                    
                    for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                        if idx >= 2: break # Max 2 hands
                        for l_idx, landmark in enumerate(hand_landmarks):
                            hand_matrix[idx, l_idx] = [landmark.x, landmark.y, landmark.z]
                    
                    flattened_features = hand_matrix.flatten()
                    dataset_features.append(flattened_features)
                    dataset_labels.append(sign_name)

        # Save extracted matrices to data/processed/
        if dataset_features:
            np.save(os.path.join(processed_data_dir, 'X_features.npy'), np.array(dataset_features))
            np.save(os.path.join(processed_data_dir, 'y_labels.npy'), np.array(dataset_labels))
            print(f"Extraction Complete! Saved {len(dataset_features)} frames data safely.")
        else:
            print("No features extracted. Please add raw image/video samples in data/raw/ folders first.")

if __name__ == "__main__":
    extract_coordinates_from_dataset()
