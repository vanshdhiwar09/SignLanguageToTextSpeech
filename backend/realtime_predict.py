import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# -------- CONFIG --------
CONFIDENCE_THRESHOLD = 0.70
STABLE_FRAMES = 4

MODEL_PATH = "backend/model/sign_model.h5"
GESTURES = ["A", "B", "C", "D"]

WORD_BUFFER = ""
LETTER_COOLDOWN = 8
PAUSE_FRAMES = 30
# ------------------------

# Load model
model = load_model(MODEL_PATH)

# MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not accessible")
    exit()

# -------- STATE --------
last_letter = ""
cooldown = 0
pause_count = 0
stable_letter = ""
stable_count = 0
best_confidence = 0.0

# ----------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        pause_count = 0

        if cooldown > 0:
            cooldown -= 1

        hand_landmarks = results.multi_hand_landmarks[0]

        # Wrist normalization
        base_x = hand_landmarks.landmark[0].x
        base_y = hand_landmarks.landmark[0].y
        base_z = hand_landmarks.landmark[0].z

        row = []
        for lm in hand_landmarks.landmark:
            row.extend([
                lm.x - base_x,
                lm.y - base_y,
                lm.z - base_z
            ])

        if len(row) == 63:
            X = np.array(row).reshape(1, -1)
            preds = model.predict(X, verbose=0)

            pred_index = np.argmax(preds)
            confidence = preds[0][pred_index]
            letter = GESTURES[pred_index]
            print(letter, f"{confidence:.2f}")


            if confidence >= CONFIDENCE_THRESHOLD:
                if letter == stable_letter:
                    stable_count += 1
                else:
                    stable_letter = letter
                    stable_count = 1

                if stable_count >= STABLE_FRAMES and cooldown == 0:
                    if letter != last_letter and confidence > best_confidence:
                        WORD_BUFFER += letter
                        last_letter = letter
                        cooldown = LETTER_COOLDOWN
                        best_confidence = confidence
                    stable_letter = ""
                    stable_count = 0

            else:
                stable_letter = ""
                stable_count = 0

        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
        )

    else:
        pause_count += 1
        if pause_count > PAUSE_FRAMES:
            WORD_BUFFER += " "
            pause_count = 0
            last_letter = ""
            stable_letter = ""
            stable_count = 0
            best_confidence = 0.0


    # -------- DISPLAY --------
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Text: {WORD_BUFFER}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    cv2.imshow("Day 8 - Word Formation (Press Q to Exit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
