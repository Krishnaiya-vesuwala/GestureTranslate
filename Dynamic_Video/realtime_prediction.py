import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pandas as pd
from collections import deque, Counter

# ====================== CONFIG ======================
MODEL_PATH = "best_cnn_gru_model.h5"
MEAN_PATH = "feature_mean.csv"
STD_PATH = "feature_std.csv"

SEQ_LEN = 30
NUM_LANDMARKS = 21
NUM_FEATURES = 63
LABEL_NAMES = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
               5: "5", 6: "6", 7: "7", 8: "8", 9: "9"}

CAM_INDEX = 0
CONFIDENCE_THRESHOLD = 0.65
PREDICTION_COOLDOWN = 8
TEMPERATURE = 1.0   # was 1.5 -- that was artificially flattening confidence
                     # on top of an already-mismatched feature space.
                     # Keep at 1.0 while validating the fix; only raise it
                     # later if the model is genuinely overconfident.
# ====================================================

print("Loading model and normalization stats...")
model = tf.keras.models.load_model(MODEL_PATH)

mean_df = pd.read_csv(MEAN_PATH, index_col=0)
std_df = pd.read_csv(STD_PATH, index_col=0)
mean_values = mean_df.values.flatten()
std_values = std_df.values.flatten()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)

frame_buffer = deque(maxlen=SEQ_LEN)
prediction_cooldown = 0


def normalize_landmarks(landmarks):
    """
    Wrist-relative, scale-normalized -- MUST match
    normalize_landmarks_relative() in 2_preprocess.py exactly.
    """
    if not landmarks:
        return np.zeros(NUM_FEATURES)
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    wrist = coords[0]
    rel = coords - wrist
    max_val = np.max(np.abs(rel))
    if max_val < 1e-6:
        max_val = 1.0
    return (rel / max_val).flatten()


def extract_landmarks(results):
    if results.multi_hand_landmarks:
        return normalize_landmarks(results.multi_hand_landmarks[0].landmark)
    return np.zeros(NUM_FEATURES)


def normalize_sequence(seq):
    return (seq - mean_values) / std_values


def predict_gesture(buffer):
    if len(buffer) < SEQ_LEN:
        return None, 0.0

    sequence = np.array(buffer)
    sequence = normalize_sequence(sequence)
    sequence = np.expand_dims(sequence, axis=0)

    pred = model.predict(sequence, verbose=0)[0]

    if TEMPERATURE != 1.0:
        pred = np.exp(np.log(np.clip(pred, 1e-8, 1.0)) / TEMPERATURE)
        pred = pred / np.sum(pred)

    class_id = int(np.argmax(pred))
    confidence = float(np.max(pred))
    return class_id, confidence


# ====================== MAIN ======================
cap = cv2.VideoCapture(CAM_INDEX)
print("Real-Time Prediction Started (Press 'q' to quit)")

current_prediction = "None"
current_confidence = 0.0
recent_preds = deque(maxlen=5)   # simple majority-vote smoothing, optional

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    feats = extract_landmarks(results)
    frame_buffer.append(feats)

    if results.multi_hand_landmarks:
        for hlm in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)

    if len(frame_buffer) == SEQ_LEN and prediction_cooldown == 0:
        class_id, confidence = predict_gesture(frame_buffer)

        if class_id is not None and confidence >= CONFIDENCE_THRESHOLD:
            recent_preds.append(class_id)
            # majority vote over the last few accepted predictions
            majority_class = Counter(recent_preds).most_common(1)[0][0]
            current_prediction = LABEL_NAMES.get(majority_class, "Unknown")
            current_confidence = confidence
            prediction_cooldown = PREDICTION_COOLDOWN
        else:
            current_prediction = "Low Confidence"
            current_confidence = confidence

    if prediction_cooldown > 0:
        prediction_cooldown -= 1

    cv2.putText(frame, f"Prediction: {current_prediction}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(frame, f"Confidence: {current_confidence:.2f}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Improved Real-Time Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()