import cv2
import mediapipe as mp
import numpy as np
import sys

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose()
hands = mp_hands.Hands(max_num_hands=2)

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam.")
    sys.exit(1)
else:
    print("✅ Webcam opened.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame.")
            break

        # Process frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        hand_results = hands.process(rgb_frame)

        if results.pose_landmarks:
            print("🧍 Pose landmarks detected.")
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

        if hand_results.multi_hand_landmarks:
            print("✋ Hand landmarks detected.")
            for handLms in hand_results.multi_hand_landmarks:
                lmList = []
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append((cx, cy))
                if len(lmList) >= 17:
                    print(f"👈 Landmark 16: {lmList[16][0]}, 👉 Landmark 15: {lmList[15][0]}")

        cv2.imshow("Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Exiting...")
            break

except KeyboardInterrupt:
    print("👋 Interrupted.")

finally:
    cap.release()
    cv2.destroyAllWindows()
