import os
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping
import logging

# --- Logging Setup ---
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_file = os.path.join(base_dir, 'logs', 'project.log')
logging.basicConfig(
    filename=log_file, 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def train_psl_model():
    print("🚀 Loading prepared data splits from data/processed/...")
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    models_dir = os.path.join(base_dir, 'models')
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    # Hot-loading the exactly matching array data splits (7,000 Train / 1,500 Val)
    X_train = np.load(os.path.join(processed_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(processed_dir, 'y_train.npy'))
    X_val = np.load(os.path.join(processed_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(processed_dir, 'y_val.npy'))

    print(f"📊 Training Matrix Shape: {X_train.shape} (7,000 Samples)")
    print(f"📊 Input shape per sequence: {X_train.shape[1:]} (Frames x Keypoints)")
    
    # --- Deep Dense Neural Network (ANN) Architecture ---
    model = Sequential([
        # Safely passing dynamic sequence length shapes
        Flatten(input_shape=(X_train.shape[1], X_train.shape[2])),
        
        Dense(256, activation='relu'),
        Dropout(0.3),  # Anti-Overfitting Mechanism 1: Drops 30% random connections
        
        Dense(128, activation='relu'),
        Dropout(0.3),  # Anti-Overfitting Mechanism 2: Drops 30% random connections
        
        Dense(64, activation='relu'),
        Dense(30, activation='softmax')  # 30 Output Classes (Alphabets + Conversational Words)
    ])
    
    model.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    model.summary()
    
    # --- Early Stopping Setup (Anti-Overfitting Mechanism 3) ---
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )
    
    print("\n🏋️ Starting Model Training using Mini-Batch Processing (Batch Size: 32)...")
    logging.info("ANN Training cycle initialized with 7,000 samples.")
    
    # Training the model on exact instructor configurations
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,  # Smooth batch processing for 16GB RAM stability
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )
    
    # Saving trained model brain weights securely
    model_path = os.path.join(models_dir, 'psl_ann_model.h5')
    model.save(model_path)
    
    print(f"\n💾 Model trained successfully! Saved weights at: models/psl_ann_model.h5")
    logging.info("ANN Training cycle completed and weights saved successfully.")

if __name__ == "__main__":
    train_psl_model()
