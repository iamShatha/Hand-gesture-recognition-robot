import cv2
import csv
import os
from mediapipe.python import solutions

mp_hands = solutions.hands
mp_drawing = solutions.drawing_utils

# save file on Desktop
CSV_FILE = os.path.join(os.path.expanduser("~"), "Desktop", "gestures.csv")
print("Saving to:", CSV_FILE)

# create file only if not exists
if not os.path.exists(CSV_FILE):
    header = []
    for i in range(21):
        header.extend([f"x{i}", f"y{i}", f"z{i}"])
    header.extend(["hand_side", "finger_count", "command"])

    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

# count existing rows (minus header)
with open(CSV_FILE, "r", newline="") as f:
    sample_count = max(sum(1 for _ in f) - 1, 0)

print("Current samples:", sample_count)

cap = cv2.VideoCapture(0)

def get_command(finger_count):
    if finger_count == 0:
        return "S"
    elif finger_count == 5:
        return "F"
    elif finger_count == 1:
        return "R"
    elif finger_count == 2:
        return "L"
    elif finger_count == 3:
        return "B"
    else:
        return "S"

with mp_hands.Hands(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    max_num_hands=1
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        features = None
        hand_side = "Unknown"
        finger_count = 0
        command = "S"

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # extract landmarks
            features = []
            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])

            # detect hand side
            if results.multi_handedness:
                hand_side = results.multi_handedness[0].classification[0].label

            # count fingers
            tip_ids = [4, 8, 12, 16, 20]
            finger_states = []

            for tip_id in tip_ids:
                tip = hand_landmarks.landmark[tip_id]
                pip = hand_landmarks.landmark[tip_id - 2]

                if tip_id == 4:
                    if hand_side == "Right":
                        finger_states.append(tip.x < pip.x)
                    elif hand_side == "Left":
                        finger_states.append(tip.x > pip.x)
                    else:
                        finger_states.append(False)
                else:
                    finger_states.append(tip.y < pip.y)

            finger_count = finger_states.count(True)
            command = get_command(finger_count)

        # display info
        cv2.putText(frame, f"Hand: {hand_side}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(frame, f"Fingers: {finger_count}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.putText(frame, f"Command: {command}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 0), 2)

        cv2.putText(frame, "Press S = save (only if correct)", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

        cv2.putText(frame, "Press Q = quit", (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

        cv2.imshow("Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            if features is not None:
                with open(CSV_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(features + [hand_side, finger_count, command])

                sample_count += 1
                print(f"Saved #{sample_count} | {hand_side} | Fingers={finger_count} | Cmd={command}")
            else:
                print("No hand detected")

        elif key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()