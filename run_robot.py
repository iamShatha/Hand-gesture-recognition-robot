import cv2
import serial
import time
import os
import joblib
import numpy as np
from mediapipe.python import solutions

SERIAL_PORT = "/dev/cu.usbserial-1110"
BAUD_RATE = 9600


# Load trained model
model_path = "gesture_model.pkl"
model = joblib.load(model_path)
print("Model loaded from:", model_path)

mp_hands = solutions.hands
mp_drawing = solutions.drawing_utils

last_sent_command = ""
last_send_time = 0

hand_side_map = {
    "Left": 0,
    "Right": 1,
    "Unknown": 2
}


def send_command(command: str) -> None:
    global last_sent_command, last_send_time

    current_time = time.time()

    if command == last_sent_command and (current_time - last_send_time) < 0.2:
        return

    try:
        ser.write((command + '\n').encode())
        print("Sent:", command)

        last_sent_command = command
        last_send_time = current_time

    except Exception as e:
        print("Serial error:", e)


def count_fingers(hand_landmarks, hand_label):
    fingers = []

    # Thumb
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]

    if hand_label == "Right":
        fingers.append(thumb_tip.x < thumb_ip.x)
    else:
        fingers.append(thumb_tip.x > thumb_ip.x)

    # Other 4 fingers
    tip_ids = [8, 12, 16, 20]
    pip_ids = [6, 10, 14, 18]

    for tip_id, pip_id in zip(tip_ids, pip_ids):
        finger_tip = hand_landmarks.landmark[tip_id]
        finger_pip = hand_landmarks.landmark[pip_id]
        fingers.append(finger_tip.y < finger_pip.y)

    return fingers.count(True)


def extract_features(hand_landmarks, hand_label):
    features = []

    # 21 landmarks × (x, y, z) = 63
    for lm in hand_landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])

    # hand side
    hand_side_num = hand_side_map.get(hand_label, 2)
    features.append(hand_side_num)

    # finger count
    finger_count = count_fingers(hand_landmarks, hand_label)
    features.append(finger_count)

    return np.array(features).reshape(1, -1), finger_count


cap = cv2.VideoCapture(0)

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    ser.reset_input_buffer()
    ser.reset_output_buffer()

except Exception as e:
    print("Could not open serial port:", e)
    cap.release()
    raise SystemExit

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        finger_count = 0
        command = "S"
        hand_label = "Unknown"

        if results.multi_hand_landmarks and results.multi_handedness:
            hand_landmarks = results.multi_hand_landmarks[0]
            hand_label = results.multi_handedness[0].classification[0].label

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            X_input, finger_count = extract_features(hand_landmarks, hand_label)

            try:
                predicted = model.predict(X_input)[0]
                print("Predicted:", predicted)

                if predicted in ["F", "B", "L", "R", "S"]:
                    command = predicted
                else:
                    command = "S"

            except Exception as e:
                print("Prediction error:", e)
                command = "S"

        else:
            command = "S"

        send_command(command)

        cv2.putText(frame, f"Hand: {hand_label}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.putText(frame, f"Fingers: {finger_count}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.putText(frame, f"Cmd: {command}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow("Hand Gesture Robot Control - Model", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            send_command("S")
            break

cap.release()
ser.close()
cv2.destroyAllWindows()