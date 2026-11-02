import os
import numpy as np

def load_hand_features_only():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    
    X_path = os.path.join(processed_dir, 'X_train.npy')
    y_path = os.path.join(processed_dir, 'y_train.npy')
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print("Error: Extracted pre-extracted keypoints dataset not found!")
        return None, None
        
    X = np.load(X_path)
    y = np.load(y_path)
    
    print(f"Original Dataset Shape: {X.shape}")
    
    
    if X.shape[-1] == 258:
        print(" Slicing dataset to isolate Hand Landmarks only (Columns 132 to 258)...")
        
        X_hands = X[:, :, 132:]
    else:
        X_hands = X
        
    print(f" Optimized Hand Dataset Shape: {X_hands.shape} (Perfect for 126 columns architecture)")
    return X_hands, y

if __name__ == "__main__":
    load_hand_features_only()
