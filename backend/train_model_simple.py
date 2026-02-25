"""
Simple Model Training Script (No Plotting)
Trains the MLP classifier without matplotlib dependencies
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import json


def main():
    """Main training pipeline"""
    print("="*60)
    print("🧠 SIGN LANGUAGE GESTURE MODEL TRAINING")
    print("="*60)
    
    # Load data
    print("\n📂 Loading data from CSV...")
    data_path = "data/gesture_data.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ Dataset not found: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} samples")
    
    # Display class distribution
    print("\n📊 Class Distribution:")
    class_counts = df['label'].value_counts()
    for label, count in class_counts.items():
        print(f"   {label}: {count} samples")
    
    # Separate features and labels
    X = df.drop('label', axis=1).values
    y = df['label'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    
    print(f"\n🏷️  Total classes: {num_classes}")
    print(f"📐 Feature shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=0.2, 
        random_state=42,
        stratify=y_encoded
    )
    
    print(f"\n📊 Data Split:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    # Create model
    print("\n🏗️  Building model architecture...")
    
    model = keras.Sequential([
        layers.Input(shape=(X.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Model created")
    print("\n📋 Model Summary:")
    model.summary()
    
    # Compute class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_encoded),
        y=y_encoded
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    # Train model
    print(f"\n🚀 Starting training...")
    print(f"   Epochs: 50")
    print(f"   Batch size: 32")
    print(f"   Validation split: 0.2")
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n✅ Training completed!")
    
    # Evaluate on test set
    print("\n📊 Evaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test Loss: {loss:.4f}")
    print(f"   Test Accuracy: {accuracy*100:.2f}%")
    
    # Save model
    print(f"\n💾 Saving model...")
    model.save("model.h5")
    print(f"✅ Model saved to: model.h5")
    
    # Save metadata
    metadata = {
        "num_classes": num_classes,
        "classes": label_encoder.classes_.tolist(),
        "input_shape": 63,
        "architecture": "MLP",
        "test_accuracy": float(accuracy),
        "test_loss": float(loss)
    }
    
    with open("model_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata saved to: model_metadata.json")
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print("\n📁 Generated files:")
    print("   - model.h5 (trained model)")
    print("   - model_metadata.json (class labels and info)")
    print(f"\n🎯 Final Test Accuracy: {accuracy*100:.2f}%")


if __name__ == "__main__":
    main()
