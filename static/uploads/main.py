import cv2
import numpy as np
import mediapipe as mp
import os
import json
from flask import Flask, render_template, request, jsonify, Response

app = Flask(__name__)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Clothing catalog (would normally be in a database)
CLOTHING_CATALOG = {
    "shirts": {
        "tshirt_red": "static/clothing/shirts/tshirt_red.png",
        "tshirt_blue": "static/clothing/shirts/tshirt_blue.png",
        "shirt_formal": "static/clothing/shirts/shirt_formal.png"
    },
    "pants": {
        "jeans": "static/clothing/pants/jeans.png",
        "chinos": "static/clothing/pants/chinos.png",
        "shorts": "static/clothing/pants/shorts.png"
    }
}

# Global variables for selected clothing
current_shirt = None
current_pant = None


def validate_image(img):
    if img is None:
        raise ValueError("Image is None")
    if len(img.shape) not in [2, 3]:
        raise ValueError(f"Invalid image shape: {img.shape}")
    return img


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = validate_image(img)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def calculate_body_measurements(landmarks, image_width, image_height):
    measurements = {}
    try:
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]

        measurements['shoulder_width'] = abs(left_shoulder.x - right_shoulder.x) * image_width
        measurements['hip_width'] = abs(left_hip.x - right_hip.x) * image_width
        measurements['torso_length'] = abs(left_shoulder.y - left_hip.y) * image_height
        measurements['leg_length'] = abs(left_hip.y - left_ankle.y) * image_height

    except Exception as e:
        print(f"Error calculating measurements: {e}")
        measurements['shoulder_width'] = 100
        measurements['hip_width'] = 80
        measurements['torso_length'] = 150
        measurements['leg_length'] = 200

    return measurements


def smart_resize(item_img, measurements, item_type):
    try:
        if item_type == 'shirt':
            target_width = max(10, measurements['shoulder_width'] * 1.15)
            target_height = max(10, measurements['torso_length'] * 1.5)
        elif item_type == 'pant':
            target_width = max(10, measurements['hip_width'] * 1.25)
            target_height = max(10, measurements['leg_length'] * 1)
        else:
            raise ValueError(f"Unknown item type: {item_type}")

        orig_height, orig_width = item_img.shape[:2]
        aspect_ratio = orig_width / orig_height

        if target_width / target_height > aspect_ratio:
            new_width = int(target_width)
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = int(target_height)
            new_width = int(new_height * aspect_ratio)

        return cv2.resize(item_img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    except Exception as e:
        print(f"Error resizing {item_type}: {e}")
        return item_img


def natural_positioning(resized_item, landmarks, item_type, image_width, image_height):
    try:
        if item_type == 'shirt':
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

            # Neck approximation
            neck_y = (left_shoulder.y + right_shoulder.y) / 2 - 0.03

            x = int((left_shoulder.x + right_shoulder.x) / 2 * image_width - resized_item.shape[1] / 2)
            y = int(neck_y * image_height - resized_item.shape[0] * 0.1)

            y = max(0, y)  # Ensure shirt doesn't go above frame

        elif item_type == 'pant':
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

            x = int((left_hip.x + right_hip.x) / 2 * image_width - resized_item.shape[1] / 2)
            y = int((left_hip.y + right_hip.y) / 2 * image_height - resized_item.shape[0] * 0.2)

        else:
            raise ValueError(f"Unknown item type: {item_type}")

        return x, y
    except Exception as e:
        print(f"Error positioning {item_type}: {e}")
        return 0, 0


def overlay_image(background, overlay, x, y):
    try:
        if background is None or overlay is None:
            return background

        if background.shape[2] == 3:
            background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)
        if overlay.shape[2] == 3:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

        h, w = background.shape[:2]
        oh, ow = overlay.shape[:2]

        x1, y1 = max(int(x), 0), max(int(y), 0)
        x2, y2 = min(int(x + ow), w), min(int(y + oh), h)

        if x1 >= x2 or y1 >= y2:
            return background

        ox1 = max(0, -int(x))
        oy1 = max(0, -int(y))
        ox2 = ox1 + (x2 - x1)
        oy2 = oy1 + (y2 - y1)

        overlay_crop = overlay[oy1:oy2, ox1:ox2]
        background_crop = background[y1:y2, x1:x2]

        overlay_alpha = overlay_crop[:, :, 3] / 255.0
        background_alpha = background_crop[:, :, 3] / 255.0

        for c in range(3):
            background_crop[:, :, c] = (
                    overlay_alpha * overlay_crop[:, :, c] +
                    (1 - overlay_alpha) * background_crop[:, :, c]
            )

        combined_alpha = 1 - (1 - overlay_alpha) * (1 - background_alpha)
        background_crop[:, :, 3] = combined_alpha * 255

        background[y1:y2, x1:x2] = background_crop
        return background

    except Exception as e:
        print(f"Overlay error: {e}")
        return background


def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)  # Mirror the frame

        if current_shirt or current_pant:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)

                if results.pose_landmarks:
                    measurements = calculate_body_measurements(
                        results.pose_landmarks.landmark,
                        frame.shape[1],
                        frame.shape[0]
                    )

                    if current_pant:
                        resized_pant = smart_resize(current_pant, measurements, 'pant')
                        pant_x, pant_y = natural_positioning(
                            resized_pant, results.pose_landmarks.landmark, 'pant',
                            frame.shape[1], frame.shape[0]
                        )
                        frame = overlay_image(frame, resized_pant, pant_x, pant_y)

                    if current_shirt:
                        resized_shirt = smart_resize(current_shirt, measurements, 'shirt')
                        shirt_x, shirt_y = natural_positioning(
                            resized_shirt, results.pose_landmarks.landmark, 'shirt',
                            frame.shape[1], frame.shape[0]
                        )
                        frame = overlay_image(frame, resized_shirt, shirt_x, shirt_y)

            except Exception as e:
                print(f"Virtual try-on error: {e}")

        ret, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR))
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('index.html', catalog=CLOTHING_CATALOG)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/select_item', methods=['POST'])
def select_item():
    global current_shirt, current_pant

    data = request.json
    item_type = data.get('type')
    item_id = data.get('id')

    try:
        if item_type == 'shirt':
            current_shirt = load_image(CLOTHING_CATALOG['shirts'][item_id])
        elif item_type == 'pant':
            current_pant = load_image(CLOTHING_CATALOG['pants'][item_id])
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/clear_item', methods=['POST'])
def clear_item():
    global current_shirt, current_pant

    data = request.json
    item_type = data.get('type')

    if item_type == 'shirt':
        current_shirt = None
    elif item_type == 'pant':
        current_pant = None

    return jsonify({"status": "success"})


if __name__ == '__main__':
    os.makedirs('../clothing/shirts', exist_ok=True)
    os.makedirs('../clothing/pants', exist_ok=True)
    app.run(debug=True)