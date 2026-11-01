import cv2
import numpy as np
import os
import tensorflow as tf
from keras.models import load_model
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 1. Configurations & Paths ---
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, 'models', 'psl_ann_model.h5')
task_path = os.path.join(base_dir, 'models', 'hand_landmarker.task')

TARGET_SIGNS = [
    "A", "B", "C", "D", "E",
    "Alif", "Bay", "Pay", "Te", "Se",
    "Pakistan", "ThankYou", "Water", "Yes", "No", 
    "Hello", "Good", "Please", "Sorry", "Teacher", 
    "Student", "Help", "Food", "Eat", "Drink", 
    "Sleep", "Go", "Come", "Stop", "School"
]

print(" Loading trained PSL AI Model Brain...")
model = load_model(model_path)

print("Connecting to hand_landmarker.task inside the models folder...")
base_options = python.BaseOptions(model_asset_path=task_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

def extract_live_features(detection_result):
    """Processes upgraded HandLandmarker result directly into matching dimensions (258 features)."""
    pose_placeholder = np.zeros(33 * 4) 
    lh_flat = np.zeros(21 * 3)
    rh_flat = np.zeros(21 * 3)
    
    # --- FIXED NESTED LIST ATTRIBUTE ERROR ---
    if detection_result.hand_landmarks and detection_result.handedness:
        for idx, handedness_list in enumerate(detection_result.handedness):
            if len(handedness_list) > 0:
                # Accessing the first element of the inner list safely
                hand_label = handedness_list[0].category_name
                
                # Check to prevent out-of-bounds index errors if hand detection arrays mismatch
                if idx < len(detection_result.hand_landmarks):
                    landmarks = detection_result.hand_landmarks[idx]
                    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
                    
                    if hand_label == "Left" or hand_label == "Left_Hand":
                        lh_flat = coords
                    else:
                        rh_flat = coords
                
    return np.concatenate([pose_placeholder, lh_flat, rh_flat])

# --- 3. Capture Loop ---
cap = cv2.VideoCapture(0)
sequence = []
sentence = []
predictions = []
threshold = 0.7

print(" Activating Webcam Stream window safely...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Camera frame capture failure.")
        break
        
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Format frame correctly for the Image Tasks API
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    detection_result = detector.detect(mp_image)
    
    # Mapping points data arrays
    keypoints = extract_live_features(detection_result)
    sequence.append(keypoints)
    sequence = sequence[-30:] # Last 30 frames window buffer
    
    if len(sequence) == 30:
        input_data = np.expand_dims(sequence, axis=0)
        res = model.predict(input_data, verbose=0)
        
        confidence_scores = res.flatten()
        predicted_idx = int(np.argmax(confidence_scores))
        predictions.append(predicted_idx)
        
        if len(predictions) >= 10 and np.all(np.array(predictions[-10:]) == predicted_idx):
            if float(confidence_scores[predicted_idx]) > threshold:
                predicted_sign = TARGET_SIGNS[predicted_idx]
                if len(sentence) > 0:
                    if predicted_sign != sentence[-1]:
                        sentence.append(predicted_sign)
                else:
                    sentence.append(predicted_sign)
                    
        if len(sentence) > 3:
            sentence = sentence[-3:]
            
    # Draw points layout markers
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
                
    # UI Banner Rendering
    cv2.rectangle(frame, (0, 0), (640, 45), (245, 117, 16), -1)
    display_text = ' '.join(sentence)
    cv2.putText(frame, f"PREDICTION: {display_text}", (15, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
    cv2.imshow('PSL Live Sign Translation Engine', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
