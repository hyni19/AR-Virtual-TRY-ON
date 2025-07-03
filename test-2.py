import cv2
import numpy as np
import mediapipe as mp
import os
import sys
import time
import math

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.6, min_tracking_confidence=0.6)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

def validate_image(img):
    if img is None:
        raise ValueError("Image is None")
    if len(img.shape) not in [2, 3, 4]:
        raise ValueError(f"Invalid image shape: {img.shape}")
    return img

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = validate_image(img)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img

def calculate_body_measurements(landmarks, image_width, image_height):
    m = {'shoulder_width': 100, 'hip_width': 80, 'torso_length': 150, 'leg_length': 200,
         'shoulder_y': 0, 'ankle_y': image_height, 'hip_y': image_height*0.6,
         'l_ankle': (int(image_width*0.25), int(image_height*0.85)),
         'r_ankle': (int(image_width*0.75), int(image_height*0.85))}
    try:
        lm = mp_pose.PoseLandmark
        lsh, rsh = landmarks[lm.LEFT_SHOULDER], landmarks[lm.RIGHT_SHOULDER]
        lhip, rhip = landmarks[lm.LEFT_HIP], landmarks[lm.RIGHT_HIP]
        lankle, rankle = landmarks[lm.LEFT_ANKLE], landmarks[lm.RIGHT_ANKLE]
        shleft = np.array([lsh.x*image_width, lsh.y*image_height])
        shright = np.array([rsh.x*image_width, rsh.y*image_height])
        hipleft = np.array([lhip.x*image_width, lhip.y*image_height])
        hipright = np.array([rhip.x*image_width, rhip.y*image_height])
        ankleft = np.array([lankle.x*image_width, lankle.y*image_height])
        ankright = np.array([rankle.x*image_width, rankle.y*image_height])
        m['shoulder_width'] = np.linalg.norm(shleft - shright)
        m['hip_width'] = np.linalg.norm(hipleft - hipright)
        m['torso_length'] = np.mean([np.linalg.norm(shleft-hipleft), np.linalg.norm(shright-hipright)])
        m['leg_length'] = np.mean([np.linalg.norm(ankleft-hipleft), np.linalg.norm(ankright-hipright)])
        m['shoulder_y'] = min(lsh.y, rsh.y) * image_height
        m['ankle_y'] = max(lankle.y, rankle.y) * image_height
        m['hip_y'] = (lhip.y + rhip.y)/2 * image_height
        m['l_ankle'] = (int(ankleft[0]), int(ankleft[1]))
        m['r_ankle'] = (int(ankright[0]), int(ankright[1]))
    except Exception as e:
        print(f"Error calculating measurements: {e}")
    return m

def safe_resize(img, width, height):
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid resize: {width} x {height}")
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

def rotate_button(img, angle_deg):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1)
    rotated = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return rotated

def resize_clothing_item(img, m, kind):
    try:
        oh, ow = img.shape[:2]
        aspect = ow/oh
        if kind == 'shirt':
            width = m['shoulder_width']*1.3
            height = m['torso_length']*1.2
        elif kind == 'pant':
            width = m['hip_width']*1.4
            height = (m['ankle_y'] - m['hip_y'])*1.1
        elif kind == 'saree':
            width = m['shoulder_width'] * 1.32
            height = (m['ankle_y'] - m['shoulder_y']) * 1.04
        else:
            width = ow
            height = oh
        if width/height > aspect:
            nw, nh = width, width/aspect
        else:
            nh, nw = height, height*aspect
        return safe_resize(img, nw, nh)
    except Exception as e:
        print(f"Error resizing {kind}: {e}")
        return img

def position_clothing_item(item, lm, kind, imgw, imgh, m=None, x_shift=0, y_shift=0):
    try:
        lms = mp_pose.PoseLandmark
        if kind == 'shirt':
            lsh, rsh = lm[lms.LEFT_SHOULDER], lm[lms.RIGHT_SHOULDER]
            cx = int((lsh.x + rsh.x)/2 * imgw)
            y = int(min(lsh.y, rsh.y) * imgh - item.shape[0]*0.15)
            x = int(cx - item.shape[1]/2)
        elif kind == 'pant':
            lhip, rhip = lm[lms.LEFT_HIP], lm[lms.RIGHT_HIP]
            cx = int((lhip.x + rhip.x)/2 * imgw)
            y = int(((lhip.y + rhip.y)/2) * imgh) - int(item.shape[0]*0.05)
            x = int(cx - item.shape[1]/2)
        elif kind == 'saree':
            lsh, rsh = lm[lms.LEFT_SHOULDER], lm[lms.RIGHT_SHOULDER]
            cx = int((lsh.x + rsh.x)/2 * imgw)
            y = int(m['shoulder_y'] - item.shape[0]*0.03)
            x = int(cx - item.shape[1]/2)
            x += x_shift
            y += y_shift
        else:
            x, y = 0, 0
        return int(x), int(y)
    except Exception as e:
        print(f"Error positioning {kind}: {e}")
        return 0, 0

def overlay_image(background, overlay, x, y):
    try:
        if background.shape[2] == 3:
            background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)
        if overlay.shape[2] == 3:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)
        h, w = background.shape[:2]
        oh, ow = overlay.shape[:2]
        x1, y1 = max(int(x), 0), max(int(y), 0)
        x2, y2 = min(int(x+ow), w), min(int(y+oh), h)
        if x1>=x2 or y1>=y2: return background
        ox1, oy1 = max(0,-int(x)), max(0,-int(y))
        ox2, oy2 = ox1+(x2-x1), oy1+(y2-y1)
        overlay_crop = overlay[oy1:oy2, ox1:ox2]
        background_crop = background[y1:y2, x1:x2]
        overlay_alpha = overlay_crop[:,:,3]/255.0
        background_alpha = background_crop[:,:,3]/255.0
        for c in range(3):
            background_crop[:,:,c] = (
                overlay_alpha*overlay_crop[:,:,c] +
                (1-overlay_alpha)*background_crop[:,:,c]
            )
        combined_alpha = 1 - (1 - overlay_alpha) * (1 - background_alpha)
        background_crop[:,:,3] = combined_alpha * 255
        background[y1:y2, x1:x2] = background_crop
        return background
    except Exception as e:
        print(f"Overlay error: {e}")
        return background

def hex_to_bgr(hex_color):
    h = hex_color.lstrip('#')
    if len(h) != 6:
        return (255,255,255)
    return (int(h[4:6],16), int(h[2:4],16), int(h[0:2],16))

def recolor_with_texture(img, target_bgr):
    if img.shape[2] != 4:
        raise ValueError("Image must have alpha channel (BGRA)")
    alpha = img[:,:,3]
    mask = alpha > 0
    orig_rgb = img[:,:,:3].astype(np.float32)
    luminance = cv2.cvtColor(orig_rgb, cv2.COLOR_BGR2GRAY)/255.0
    lum_vals = luminance[mask]
    min_lum = np.min(lum_vals) if len(lum_vals)>0 else 0.0
    max_lum = np.max(lum_vals) if len(lum_vals)>0 else 1.0
    denom = max(max_lum - min_lum, 1e-5)
    min_level = 0.16
    luminance_norm = (luminance - min_lum)/denom
    luminance_norm = luminance_norm*(1-min_level) + min_level
    result = np.zeros_like(img)
    for c in range(3):
        temp = (luminance_norm * target_bgr[c])
        temp = np.clip(temp,0,255)
        result[:,:,c][mask] = temp[mask]
    result[:,:,3] = alpha
    return result.astype(np.uint8)

def get_all_images_sorted(folder):
    valid_ext = ('.png', '.jpg', '.jpeg')
    files = [f for f in os.listdir(folder) if f.lower().endswith(valid_ext)]
    files = sorted(files)
    return files

if len(sys.argv) != 7:
    print("Usage: python test-2.py <shirt_img or ''> <pant_img or ''> <saree_img or ''> <shirt_color> <pant_color> <saree_color>")
    sys.exit(1)

shirt_path = sys.argv[1]
pant_path  = sys.argv[2]
saree_path = sys.argv[3]
shirt_color = sys.argv[4]
pant_color  = sys.argv[5]
saree_color = sys.argv[6]

SHIRT_DIR = 'static/images/shirts'
PANT_DIR  = 'static/images/pants'
shirt_files = get_all_images_sorted(SHIRT_DIR)
pant_files  = get_all_images_sorted(PANT_DIR)
cur_shirt_idx = shirt_files.index(os.path.basename(shirt_path)) if shirt_path and os.path.basename(shirt_path) in shirt_files else 0
cur_pant_idx = pant_files.index(os.path.basename(pant_path)) if pant_path and os.path.basename(pant_path) in pant_files else 0

def load_shirt(idx):
    if 0 <= idx < len(shirt_files):
        fullpath = os.path.join(SHIRT_DIR, shirt_files[idx])
        img = load_image(fullpath)
        if shirt_color.lower() != "#ffffff":
            img = recolor_with_texture(img, hex_to_bgr(shirt_color))
        return img
    else:
        return None

def load_pant(idx):
    if 0 <= idx < len(pant_files):
        fullpath = os.path.join(PANT_DIR, pant_files[idx])
        img = load_image(fullpath)
        if pant_color.lower() != "#ffffff":
            img = recolor_with_texture(img, hex_to_bgr(pant_color))
        return img
    else:
        return None

shirt_img = load_shirt(cur_shirt_idx) if shirt_path and shirt_files else None
pant_img = load_pant(cur_pant_idx) if pant_path and pant_files else None
saree_img = load_image(saree_path) if saree_path and os.path.exists(saree_path) else None
if saree_img is not None and saree_color.lower() != "#ffffff":
    saree_img = recolor_with_texture(saree_img, hex_to_bgr(saree_color))

BUTTON_PATH = "Resources/button.png"
if not os.path.exists(BUTTON_PATH):
    print(f"Resource missing: {BUTTON_PATH}")
    sys.exit(1)
base_btn_img = cv2.imread(BUTTON_PATH, cv2.IMREAD_UNCHANGED)
button_scale = 0.13

def draw_green_ring(canvas, center, radius, percent):
    overlay = canvas.copy()
    angle = int(360*percent)
    cv2.ellipse(overlay, center, (radius, radius), -90, 0, angle, (40, 230, 70), 8, cv2.LINE_AA)
    if percent > 0:
        tip_deg = -90+angle
        tip_x = int(center[0] + radius * np.cos(np.deg2rad(tip_deg)))
        tip_y = int(center[1] + radius * np.sin(np.deg2rad(tip_deg)))
        cv2.circle(overlay, (tip_x, tip_y), 11, (120,255,120), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0, dst=canvas)

cap = cv2.VideoCapture(0)
if not cap or not cap.isOpened():
    print("❌ Could not open a webcam device.")
    sys.exit(1)

cv2.namedWindow('Virtual Try-On', cv2.WINDOW_NORMAL)

def create_saree_sliders():
    cv2.createTrackbar('Saree Size (%)', 'Virtual Try-On', 100, 200, lambda x:None)
    cv2.createTrackbar('Saree Y Shift', 'Virtual Try-On', 100, 200, lambda x:None)
    cv2.createTrackbar('Saree X Shift', 'Virtual Try-On', 100, 200, lambda x:None)

if saree_img is not None:
    create_saree_sliders()

GESTURE_TIME = 2.0
last_change = dict(shirt=0.0, pant=0.0)
change_delay = 1.0

left_hover_start, right_hover_start = None, None
left_loading, right_loading = False, False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame grab failure.")
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    hand_results = hands.process(rgb_frame)
    h, w = frame.shape[:2]

    if saree_img is None:
        try:
            cv2.destroyWindow('Saree Size (%)')
            cv2.destroyWindow('Saree Y Shift')
            cv2.destroyWindow('Saree X Shift')
        except Exception:
            pass

    show_shirt_btns = shirt_img is not None and saree_img is None
    show_pant_btns  = pant_img is not None and saree_img is None

    m = None
    lm = None
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        m = calculate_body_measurements(lm, w, h)

    show_frame = frame.copy()

    button_w_s, button_h_s = int(h*button_scale), int(h*button_scale)
    shirt_left = [12, int(h*0.26), 12+button_w_s, int(h*0.26)+button_h_s]
    shirt_right = [w-12-button_w_s, int(h*0.26), w-12, int(h*0.26)+button_h_s]

    if m is not None:
        ankL = m['l_ankle']
        ankR = m['r_ankle']
        p_offset = int(h*0.085)
        p_size = int(h*button_scale)
        pant_left = [ankL[0]-p_size//2, ankL[1]-p_size-p_offset, ankL[0]+p_size//2, ankL[1]-p_offset]
        pant_right = [ankR[0]-p_size//2, ankR[1]-p_size-p_offset, ankR[0]+p_size//2, ankR[1]-p_offset]
    else:
        pant_left = [int(w*0.27), int(h*0.78), int(w*0.27)+button_w_s, int(h*0.78)+button_h_s]
        pant_right = [int(w*0.73)-button_w_s, int(h*0.78), int(w*0.73), int(h*0.78)+button_h_s]

    hover_left, hover_right = False, False
    which_mode = None

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            hx = int(hand_landmarks.landmark[0].x * w)
            hy = int(hand_landmarks.landmark[0].y * h)
            if show_shirt_btns:
                if shirt_left[0]<=hx<=shirt_left[2] and shirt_left[1]<=hy<=shirt_left[3]:
                    hover_left = True; which_mode = "shirt"
                elif shirt_right[0]<=hx<=shirt_right[2] and shirt_right[1]<=hy<=shirt_right[3]:
                    hover_right = True; which_mode = "shirt"

    if show_pant_btns and m is not None:
        la = m['l_ankle']
        ra = m['r_ankle']
        for ax, ay, rect, isleft in [
                (la[0], la[1], pant_left, True),
                (ra[0], ra[1], pant_right, False)
            ]:
            bx1, by1, bx2, by2 = rect
            cx = (bx1+bx2)//2
            cy = (by1+by2)//2
            dist = math.hypot(ax - cx, ay - cy)
            threshold = int((bx2-bx1)*0.68)
            if dist < threshold:
                if isleft:
                    hover_left = True
                    which_mode = "pant"
                else:
                    hover_right = True
                    which_mode = "pant"

    now = time.time()

    if hover_left:
        if left_hover_start is None: left_hover_start = now
        left_loading = True
        percent = min((now - left_hover_start) / GESTURE_TIME, 1.0)
        if now - left_hover_start >= GESTURE_TIME and now - last_change.get(which_mode, 0) > change_delay:
            if which_mode == "shirt" and shirt_files:
                cur_shirt_idx = (cur_shirt_idx - 1) % len(shirt_files)
                shirt_img = load_shirt(cur_shirt_idx)
            if which_mode == "pant" and pant_files:
                cur_pant_idx = (cur_pant_idx - 1) % len(pant_files)
                pant_img = load_pant(cur_pant_idx)
            last_change[which_mode] = now
            left_hover_start = now
    else:
        left_hover_start = None
        left_loading = False
        percent = 0.0

    if hover_right:
        if right_hover_start is None: right_hover_start = now
        right_loading = True
        percent2 = min((now - right_hover_start) / GESTURE_TIME, 1.0)
        if now - right_hover_start >= GESTURE_TIME and now - last_change.get(which_mode, 0) > change_delay:
            if which_mode == "shirt" and shirt_files:
                cur_shirt_idx = (cur_shirt_idx + 1) % len(shirt_files)
                shirt_img = load_shirt(cur_shirt_idx)
            if which_mode == "pant" and pant_files:
                cur_pant_idx = (cur_pant_idx + 1) % len(pant_files)
                pant_img = load_pant(cur_pant_idx)
            last_change[which_mode] = now
            right_hover_start = now
    else:
        right_hover_start = None
        right_loading = False
        percent2 = 0.0

        # ... [all previous code unchanged up to button drawing] ...

    # Draw Shirt Buttons (left: rotate 180; right: 0)
    if show_shirt_btns:
        for idx, rect, angle, is_loading, perc in [
            (0, shirt_left, 180, left_loading and which_mode == "shirt", percent),
            (1, shirt_right, 0, right_loading and which_mode == "shirt", percent2),
        ]:
            x1, y1, x2, y2 = rect
            sz = x2 - x1
            btn_img = safe_resize(rotate_button(base_btn_img, angle), sz, sz)
            overlay = show_frame[y1:y2, x1:x2]
            # Only blend if overlay region and button have same, nonzero shape!
            if overlay.shape[:2] == btn_img.shape[:2] and overlay.size > 0:
                alpha = 1.0 if is_loading else 0.8
                mask = btn_img[:, :, 3] / 255.0 * alpha
                for c in range(3):
                    overlay[:, :, c] = overlay[:, :, c] * (1 - mask) + btn_img[:, :, c] * mask
                show_frame[y1:y2, x1:x2] = overlay
                if is_loading:
                    draw_green_ring(show_frame, (x1 + sz // 2, y1 + sz // 2), int(sz * 0.37), perc)

    # Draw Pant Buttons (left: -90; right: 90)
    if show_pant_btns:
        for idx, rect, angle, is_loading, perc in [
            (0, pant_left, -90, left_loading and which_mode == "pant", percent),
            (1, pant_right, 90, right_loading and which_mode == "pant", percent2),
        ]:
            x1, y1, x2, y2 = rect
            sz = x2 - x1
            btn_img = safe_resize(rotate_button(base_btn_img, angle), sz, sz)
            overlay = show_frame[y1:y2, x1:x2]
            if overlay.shape[:2] == btn_img.shape[:2] and overlay.size > 0:
                alpha = 1.0 if is_loading else 0.8
                mask = btn_img[:, :, 3] / 255.0 * alpha
                for c in range(3):
                    overlay[:, :, c] = overlay[:, :, c] * (1 - mask) + btn_img[:, :, c] * mask
                show_frame[y1:y2, x1:x2] = overlay
                if is_loading:
                    draw_green_ring(show_frame, (x1 + sz // 2, y1 + sz // 2), int(sz * 0.37), perc)

    # ... [rest of code unchanged] ...
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        m = calculate_body_measurements(lm, w, h)
        if saree_img is not None:
            if cv2.getWindowProperty('Virtual Try-On', cv2.WND_PROP_VISIBLE) >= 1:
                saree_scale = cv2.getTrackbarPos('Saree Size (%)', 'Virtual Try-On')
                saree_yshift = cv2.getTrackbarPos('Saree Y Shift', 'Virtual Try-On') - 100
                saree_xshift = cv2.getTrackbarPos('Saree X Shift', 'Virtual Try-On') - 100
            else:
                saree_scale, saree_yshift, saree_xshift = 100, 0, 0
            resized_saree = resize_clothing_item(saree_img, m, 'saree')
            scale_factor = saree_scale / 100.0
            new_width = int(resized_saree.shape[1] * scale_factor)
            new_height = int(resized_saree.shape[0] * scale_factor)
            if new_width > 5 and new_height > 5:
                resized_saree = safe_resize(resized_saree, new_width, new_height)
            x_sar, y_sar = position_clothing_item(resized_saree, lm, 'saree', w, h, m, x_shift=saree_xshift, y_shift=saree_yshift)
            show_frame = overlay_image(show_frame, resized_saree, x_sar, y_sar)
        else:
            if shirt_img is not None:
                resized_shirt = resize_clothing_item(shirt_img, m, 'shirt')
                x_s, y_s = position_clothing_item(resized_shirt, lm, 'shirt', w, h, m)
                show_frame = overlay_image(show_frame, resized_shirt, x_s, y_s)
            if pant_img is not None:
                resized_pant = resize_clothing_item(pant_img, m, 'pant')
                x_p, y_p = position_clothing_item(resized_pant, lm, 'pant', w, h, m)
                show_frame = overlay_image(show_frame, resized_pant, x_p, y_p)

    cv2.imshow('Virtual Try-On', show_frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()