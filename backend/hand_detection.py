import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not accessible")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror view
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Performance optimization
    frame.flags.writeable = False
    results = hands.process(rgb)
    frame.flags.writeable = True

    # Draw landmarks with handedness
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            label = handedness.classification[0].label  # Left / Right
            score = handedness.classification[0].score

            # Color coding
            color = (0, 255, 0) if label == "Right" else (255, 0, 0)

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=color, thickness=2)
            )

            # Wrist position for label
            h, w, _ = frame.shape
            wrist = hand_landmarks.landmark[0]
            x, y = int(wrist.x * w), int(wrist.y * h)

            cv2.putText(
                frame,
                f"{label} ({score:.2f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    cv2.imshow("Day 3 - Two Hand Detection (Press Q to Exit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
