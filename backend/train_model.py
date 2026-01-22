import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Load processed data
X = np.load("backend/processed/X.npy")
y = np.load("backend/processed/y.npy")

num_classes = len(set(y))
y_cat = to_categorical(y, num_classes=num_classes)

# Build model
model = Sequential([
    Dense(128, activation="relu", input_shape=(63,)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train model
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X, y_cat,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop]
)

# Save model
model.save("backend/model/sign_model.h5")

print("✅ Model training complete and saved as sign_model.h5")
