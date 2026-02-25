import cv2
import mediapipe as mp
import os
import csv

# -------- CONFIG --------
IMAGE_DATASET_PATH = "processed_combine_asl_dataset"
OUTPUT_DATASET_PATH = "backend/datasets"
CLASSES = ["a", "b", "c", "d"]   # lowercase folders
MAX_IMAGES = 500                # limit per class (speed + stability)
# ------------------------

os.makedirs(OUTPUT_DATASET_PATH, exist_ok=True)

mp_hands = mp.solutions.hands

# IMPORTANT: tuned for static images
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

for label in CLASSES:
    input_dir = os.path.join(IMAGE_DATASET_PATH, label)
    output_dir = os.path.join(OUTPUT_DATASET_PATH, label.upper())
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f"{label.upper()}.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)

    count = 0
    processed = 0

    for img_name in os.listdir(input_dir):
        if processed >= MAX_IMAGES:
            break

        img_path = os.path.join(input_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        # ---- CRITICAL FIXES ----
        # Resize improves detection a LOT
        image = cv2.resize(image, (640, 480))

        # Ensure 3-channel image
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        processed += 1
        # ------------------------

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # Wrist-based normalization
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
                csv_writer.writerow(row)
                count += 1

    csv_file.close()
    print(f"✅ {label.upper()}: {count} samples saved (from {processed} images)")

hands.close()
print("🎉 Image-to-landmark conversion complete")
