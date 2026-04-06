import cv2
import mediapipe as mp
import numpy as np
import joblib
import collections

#  Load trained model
model    = joblib.load('gaitsense_model.pkl')
features = joblib.load('gaitsense_features.pkl')

print("Model loaded successfully")
print(f"Expects features: {features}")

#  Setup MediaPipe
mp_drawings  = mp.solutions.drawing_utils
mp_draw_spec = mp.solutions.drawing_styles
mp_pose      = mp.solutions.pose

# Feature buffers
frame_buffer = collections.deque(maxlen=300)

def get_coord(landmark, idx):
    lm = landmark[idx]
    return np.array([lm.x, lm.y])

def angle_between(a, b, c):
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))

def extract_frame_features(landmarks):
    lm = landmarks.landmark

    l_shoulder = get_coord(lm, 11)
    r_shoulder = get_coord(lm, 12)
    l_hip      = get_coord(lm, 23)
    r_hip      = get_coord(lm, 24)
    l_knee     = get_coord(lm, 25)
    r_knee     = get_coord(lm, 26)
    l_ankle    = get_coord(lm, 27)
    r_ankle    = get_coord(lm, 28)
    l_wrist    = get_coord(lm, 15)
    r_wrist    = get_coord(lm, 16)
    l_elbow    = get_coord(lm, 13)
    r_elbow    = get_coord(lm, 14)

    l_arm_angle = angle_between(l_shoulder, l_elbow, l_wrist)
    r_arm_angle = angle_between(r_shoulder, r_elbow, r_wrist)

    l_knee_angle = angle_between(l_hip, l_knee, l_ankle)
    r_knee_angle = angle_between(r_hip, r_knee, r_ankle)

    step_width = abs(l_ankle[0] - r_ankle[0])
    hip_y = (l_hip[1] + r_hip[1]) / 2

    shoulder_center = (l_shoulder + r_shoulder) / 2
    hip_center      = (l_hip + r_hip) / 2
    trunk_lean      = abs(shoulder_center[0] - hip_center[0])

    return {
        'l_arm_angle':   l_arm_angle,
        'r_arm_angle':   r_arm_angle,
        'arm_asymmetry': abs(l_arm_angle - r_arm_angle),
        'l_knee_angle':  l_knee_angle,
        'r_knee_angle':  r_knee_angle,
        'step_width':    step_width,
        'hip_y':         hip_y,
        'trunk_lean':    trunk_lean,
    }

def build_model_input(buffer):
    if len(buffer) < 30:
        return None

    arr = np.array([list(f.values()) for f in buffer])
    keys = list(buffer[0].keys())

    row = {}
    for i, key in enumerate(keys):
        row[f'{key}_mean'] = arr[:, i].mean()
        row[f'{key}_std']  = arr[:, i].std()

    row['swing_asymmetry_sec'] = abs(
        arr[:, 0].mean() - arr[:, 1].mean()
    )
    row['overall_variability'] = (
        arr[:, 0].std() + arr[:, 1].std()
    ) / 2

    return row


risk_score   = None
risk_label   = "Collecting data..."
label_color  = (200, 200, 200)
frame_count  = 0


cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        frame_count += 1

        # Process
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        image = cv2.flip(image, 1)

        if results.pose_landmarks:
            frame_feats = extract_frame_features(results.pose_landmarks)
            frame_buffer.append(frame_feats)

            if len(frame_buffer) >= 60 and frame_count % 30 == 0:
                model_input = build_model_input(frame_buffer)
                if model_input:
                    try:
                        input_vals = np.array(
                            [model_input.get(f, 0.0) for f in features]
                        ).reshape(1, -1)

                        prob       = model.predict_proba(input_vals)[0][1]
                        risk_score = prob

                        if prob >= 0.7:
                            risk_label = f"HIGH RISK — {prob:.0%}"
                            label_color = (0, 0, 255)
                        elif prob >= 0.4:
                            risk_label = f"MODERATE — {prob:.0%}"
                            label_color = (0, 165, 255)
                        else:
                            risk_label = f"LOW RISK — {prob:.0%}"
                            label_color = (0, 200, 100)

                    except Exception:
                        risk_label  = "Processing..."
                        label_color = (200, 200, 200)

            mp_drawings.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_draw_spec.get_default_pose_landmarks_style()
            )

        # UI overlay
        cv2.rectangle(image, (0, 0), (640, 50), (30, 30, 30), -1)

        cv2.putText(image, "GaitSense",
                    (10, 33), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2)

        cv2.putText(image, risk_label,
                    (200, 33), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, label_color, 2)

        cv2.putText(image, f"Frames: {len(frame_buffer)}/60",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (180, 180, 180), 1)

        cv2.putText(image, "Walk in front of camera — Press ESC to quit",
                    (10, image.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (180, 180, 180), 1)

        cv2.imshow("GaitSense — Parkinson's Gait Screening", image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()