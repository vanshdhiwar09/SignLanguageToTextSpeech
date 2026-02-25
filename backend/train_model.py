"""
Model Training Script
Trains the MLP classifier on collected gesture data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
import json


class GestureModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, data_path: str = "data/gesture_data.csv"):
        self.data_path = data_path
        self.model = None
        self.label_encoder = None
        self.history = None
        self.num_classes = 0
        
    def load_data(self):
        """Load and preprocess data from CSV"""
        print("📂 Loading data from CSV...")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")
        
        # Load CSV
        df = pd.read_csv(self.data_path)
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
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        self.num_classes = len(self.label_encoder.classes_)
        
        print(f"\n🏷️  Total classes: {self.num_classes}")
        print(f"📐 Feature shape: {X.shape}")
        
        return X, y_encoded
    
    def create_model(self, input_shape: int = 63):
        """
        Create the MLP model architecture
        
        Args:
            input_shape: Number of input features (63 for 21 landmarks × 3)
        """
        print("\n🏗️  Building model architecture...")
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(input_shape,)),
            
            # First hidden layer
            layers.Dense(128, activation='relu', name='dense_1'),
            layers.Dropout(0.2, name='dropout_1'),
            
            # Second hidden layer
            layers.Dense(128, activation='relu', name='dense_2'),
            layers.Dropout(0.2, name='dropout_2'),
            
            # Third hidden layer
            layers.Dense(64, activation='relu', name='dense_3'),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        print("✅ Model created")
        print("\n📋 Model Summary:")
        model.summary()
        
        return model
    
    def train(self, X, y, epochs: int = 50, batch_size: int = 32, validation_split: float = 0.2):
        """
        Train the model
        
        Args:
            X: Feature array
            y: Encoded labels
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split ratio
        """
        print(f"\n🚀 Starting training...")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Validation split: {validation_split}")
        
        # Compute class weights to handle imbalance
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y),
            y=y
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        # Callbacks
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
        
        # Train model
        self.history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✅ Training completed!")
        
    def evaluate(self, X_test, y_test):
        """Evaluate model on test set"""
        print("\n📊 Evaluating model...")
        
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"   Test Loss: {loss:.4f}")
        print(f"   Test Accuracy: {accuracy*100:.2f}%")
        
        return loss, accuracy
    
    def plot_training_history(self, save_path: str = "training_history.png"):
        """Plot training history"""
        if self.history is None:
            print("⚠️  No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Train Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Val Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Train Loss')
        ax2.plot(self.history.history['val_loss'], label='Val Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Model Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Training history plot saved to: {save_path}")
        plt.close()
    
    def save_model(self, model_path: str = "model.h5", metadata_path: str = "model_metadata.json"):
        """Save model and metadata"""
        print(f"\n💾 Saving model...")
        
        # Save model
        self.model.save(model_path)
        print(f"✅ Model saved to: {model_path}")
        
        # Save metadata
        metadata = {
            "num_classes": self.num_classes,
            "classes": self.label_encoder.classes_.tolist(),
            "input_shape": 63,
            "architecture": "MLP",
            "training_samples": len(self.history.history['loss']) if self.history else 0
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Metadata saved to: {metadata_path}")


def main():
    """Main training pipeline"""
    print("="*60)
    print("🧠 SIGN LANGUAGE GESTURE MODEL TRAINING")
    print("="*60)
    
    # Initialize trainer
    trainer = GestureModelTrainer(data_path="data/gesture_data.csv")
    
    # Load data
    X, y = trainer.load_data()
    
    # Split data (stratified to maintain class distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    print(f"\n📊 Data Split:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    # Create model
    trainer.create_model(input_shape=X.shape[1])
    
    # Train model
    trainer.train(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
    
    # Evaluate on test set
    trainer.evaluate(X_test, y_test)
    
    # Plot training history
    trainer.plot_training_history("training_history.png")
    
    # Save model
    trainer.save_model("model.h5", "model_metadata.json")
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print("\n📁 Generated files:")
    print("   - model.h5 (trained model)")
    print("   - model_metadata.json (class labels and info)")
    print("   - training_history.png (accuracy/loss plots)")


if __name__ == "__main__":
    main()
