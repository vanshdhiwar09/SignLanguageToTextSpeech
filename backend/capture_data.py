import cv2
import mediapipe as mp
import csv
import os

# -------- CONFIG --------
LABEL = "D"            # Change this for each gesture
SAMPLES = 200           # Number of samples to collect
DATASET_PATH = f"backend/datasets/{LABEL}"
# ------------------------

os.makedirs(DATASET_PATH, exist_ok=True)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

count = 0
csv_file = open(f"{DATASET_PATH}/{LABEL}.csv", "w", newline="")
csv_writer = csv.writer(csv_file)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        row = []

        for lm in hand_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z])

        csv_writer.writerow(row)
        count += 1

        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
        )

        cv2.putText(
            frame,
            f"Collecting {LABEL}: {count}/{SAMPLES}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow("Day 4 - Data Collection (Press Q to Quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= SAMPLES:
        break

csv_file.close()
cap.release()
cv2.destroyAllWindows()
