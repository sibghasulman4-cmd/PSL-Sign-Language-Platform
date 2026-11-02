import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from load_features import load_hand_features_only

def train_optimized_hand_model():
    # 1. Load the chopped hand-only data matrices
    X, y = load_hand_features_only()
    if X is None: return
    
    # Encode textual target signs labels into numerical numbers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(np.unique(y_encoded))
    
    # Train / Test / Validation Split Strategy (Mam's Requirement)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # 2. Build Dense ANN Classifier optimized for 126 inputs (No Pose noise)
    model = Sequential([
        Flatten(input_shape=(30, 126)), # 30 frames sequence window * 126 keypoints metrics
        Dense(256, activation='relu'),
        Dropout(0.3), # Dropout layer to actively prevent overfitting
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax') # Final classification output probability
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print("\n🚀 Starting Model Training with batch size 32 to avoid system load...")
    # Training with active validation data splits
    model.fit(X_train, y_train, epochs=40, batch_size=32, validation_data=(X_test, y_test))
    
    # Save the polished accurate brain weights
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_save_path = os.path.join(base_dir, 'models', 'psl_ann_model.h5')
    model.save(model_save_path)
    
    print(f"\n🎉 High-Accuracy Model successfully saved at: {model_save_path}")

if __name__ == "__main__":
    train_optimized_hand_model()
