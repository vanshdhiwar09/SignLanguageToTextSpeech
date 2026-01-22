import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model

# -------- CONFIG --------
MODEL_PATH = "backend/model/sign_model.h5"
GESTURES = ["A", "B", "C", "D"]  # Must match training order
SMOOTHING_FRAMES = 10            # Increase = more stable
# ------------------------

# Load trained model
model = load_model(MODEL_PATH)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # One hand for prediction (recommended)
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not accessible")
    exit()

# Buffer for smoothing predictions
pred_buffer = deque(maxlen=SMOOTHING_FRAMES)
current_text = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Extract 63 landmark values
        row = []
        for lm in hand_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z])

        if len(row) == 63:
            X = np.array(row).reshape(1, -1)
            preds = model.predict(X, verbose=0)
            pred_index = np.argmax(preds)
            pred_buffer.append(pred_index)

            # Majority vote for stability
            stable_index = max(set(pred_buffer), key=pred_buffer.count)
            current_text = GESTURES[stable_index]

        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
        )

    # Display predicted text
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Detected Sign: {current_text}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    cv2.imshow("Day 7 - Real-Time Sign Prediction (Press Q to Exit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
