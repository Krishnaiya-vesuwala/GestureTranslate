import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pandas as pd
from collections import deque, Counter
from PIL import ImageFont, ImageDraw, Image
import os

# ====================== CONFIG ======================
MODEL_PATH = "best_cnn_gru_model.h5"
MEAN_PATH  = "feature_mean.csv"
STD_PATH   = "feature_std.csv"

SEQ_LEN       = 30
NUM_LANDMARKS = 21
NUM_FEATURES  = 63

# Gujarati digit labels
LABEL_NAMES = {
    0: "૦", 1: "૧", 2: "૨", 3: "૩", 4: "૪",
    5: "૫", 6: "૬", 7: "૭", 8: "૮", 9: "૯",
}

LABEL_NAMES_EN = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
    5: "5", 6: "6", 7: "7", 8: "8", 9: "9"
}

CAM_INDEX = 0

# ── TUNED FOR SPEED ──────────────────────────────────
CONFIDENCE_THRESHOLD     = 0.75   # ↑ raised: avoids false "1" predictions
PREDICTION_COOLDOWN      = 5      # ↓ reduced: faster prediction refresh
TEMPERATURE              = 1.0

STABILITY_COUNT_REQUIRED = 2      # ↓ reduced: accept digit faster
ACCEPT_COOLDOWN          = 10      # ↓ reduced: 5 cycles not 20 seconds
RECENT_PRED_WINDOW       = 3      # ↓ reduced: faster majority vote
# ────────────────────────────────────────────────────

# ── FONT SETUP ───────────────────────────────────────
FONT_PATH        = "NotoSansGujarati-Regular.ttf"
FONT_SIZE_LARGE  = 36
FONT_SIZE_MEDIUM = 28
FONT_SIZE_SMALL  = 22

def load_font(size):
    if os.path.exists(FONT_PATH):
        return ImageFont.truetype(FONT_PATH, size)
    else:
        print(f"WARNING: '{FONT_PATH}' not found!")
        print("Download: https://fonts.google.com/noto/specimen/Noto+Sans+Gujarati")
        return ImageFont.load_default()

font_large  = load_font(FONT_SIZE_LARGE)
font_medium = load_font(FONT_SIZE_MEDIUM)
font_small  = load_font(FONT_SIZE_SMALL)
# ─────────────────────────────────────────────────────

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

mean_df     = pd.read_csv(MEAN_PATH, index_col=0)
std_df      = pd.read_csv(STD_PATH,  index_col=0)
mean_values = mean_df.values.flatten()
std_values  = std_df.values.flatten()

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

frame_buffer        = deque(maxlen=SEQ_LEN)
prediction_cooldown = 0

# ── Sentence Generation State ─────────────────────────
current_word_digits = []
sentence            = ""
stability_counter   = 0
last_stable_class   = None
accept_cooldown     = 0
recent_preds        = deque(maxlen=RECENT_PRED_WINDOW)
# ─────────────────────────────────────────────────────


# ════════════════════════════════════════════════════
# EXISTING FUNCTIONS (UNCHANGED)
# ════════════════════════════════════════════════════

def normalize_landmarks(landmarks):
    if not landmarks:
        return np.zeros(NUM_FEATURES)
    coords  = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    wrist   = coords[0]
    rel     = coords - wrist
    max_val = np.max(np.abs(rel))
    if max_val < 1e-6:
        max_val = 1.0
    return (rel / max_val).flatten()


def extract_landmarks(results):
    if results.multi_hand_landmarks:
        return normalize_landmarks(
            results.multi_hand_landmarks[0].landmark
        )
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
    return int(np.argmax(pred)), float(np.max(pred))


# ════════════════════════════════════════════════════
# SENTENCE GENERATION HELPERS
# ════════════════════════════════════════════════════

def get_current_word():
    return "".join(current_word_digits)


def get_display_sentence():
    return sentence + get_current_word()


def commit_word():
    global sentence, current_word_digits
    word = get_current_word()
    if word:
        sentence += word + " "
        print(f"  Committed: '{word}' | Sentence: '{sentence}'")
    current_word_digits = []


def backspace_action():
    global sentence, current_word_digits
    if current_word_digits:
        removed = current_word_digits.pop()
        print(f"  Removed: '{removed}' | Building: '{get_current_word()}'")
    elif sentence:
        sentence = sentence[:-1]
        print(f"  Sentence: '{sentence}'")


def clear_action():
    global sentence, current_word_digits
    sentence            = ""
    current_word_digits = []
    print("  Cleared.")


def accept_digit(digit_str):
    current_word_digits.append(digit_str)
    print(f"  Accepted: '{digit_str}' | Building: '{get_current_word()}'")


def try_accept_prediction(class_id, confidence):
    global stability_counter, last_stable_class, accept_cooldown

    # ── Debounce ─────────────────────────────────────
    if accept_cooldown > 0:
        accept_cooldown -= 1
        return False

    # ── Low confidence → reset ────────────────────────
    if confidence < CONFIDENCE_THRESHOLD or class_id is None:
        stability_counter = 0
        last_stable_class = None
        return False

    # ── Track stability ───────────────────────────────
    if class_id == last_stable_class:
        stability_counter += 1
    else:
        last_stable_class = class_id
        stability_counter = 1

    # ── Accept if stable enough ───────────────────────
    if stability_counter >= STABILITY_COUNT_REQUIRED:
        digit_str = LABEL_NAMES.get(class_id, "?")
        accept_digit(digit_str)
        stability_counter = 0
        accept_cooldown   = ACCEPT_COOLDOWN
        return True

    return False


def reset_stability():
    global stability_counter, last_stable_class
    stability_counter = 0
    last_stable_class = None


# ════════════════════════════════════════════════════
# GUJARATI TEXT RENDERING
# ════════════════════════════════════════════════════

def put_gujarati_text(frame, text, pos, font,
                      color=(255, 255, 255),
                      bg_color=None):
    if not text:
        return
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)
    x, y   = pos
    if bg_color is not None:
        try:
            bbox = draw.textbbox((x, y), text, font=font)
            pad  = 4
            draw.rectangle(
                [bbox[0]-pad, bbox[1]-pad,
                 bbox[2]+pad, bbox[3]+pad],
                fill=bg_color
            )
        except Exception:
            pass
    draw.text((x, y), text, font=font, fill=color)
    frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def draw_gujarati_label(frame, english_label, gujarati_value,
                         pos, font,
                         label_color=(0, 255, 0),
                         value_color=(0, 255, 0)):
    x, y = pos
    cv2.rectangle(frame, (x-4, y-4), (x+300, y+45), (0,0,0), -1)
    cv2.putText(frame, english_label,
                (x, y+30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85, label_color, 2)
    (lw, _), _ = cv2.getTextSize(
        english_label, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2
    )
    put_gujarati_text(
        frame,
        text     = gujarati_value,
        pos      = (x + lw, y + 4),
        font     = font,
        color    = value_color,
        bg_color = None
    )


def wrap_gujarati_text(text, max_chars=20):
    words = text.split(" ")
    lines, line = [], ""
    for w in words:
        if not w:
            continue
        if len(line) + len(w) + 1 <= max_chars:
            line = (line + w + " ") if line else (w + " ")
        else:
            if line:
                lines.append(line.strip())
            line = w + " "
    if line.strip():
        lines.append(line.strip())
    return lines if lines else [""]


# ════════════════════════════════════════════════════
# MAIN LOOP
# ════════════════════════════════════════════════════
cap = cv2.VideoCapture(CAM_INDEX)

print("\n" + "="*50)
print("  ISL Real-Time | Gujarati Digit Output")
print("="*50)
print("  CONTROLS:")
print("    SPACE     → commit word")
print("    BACKSPACE → delete last")
print("    C         → clear all")
print("    Q         → quit")
print("="*50 + "\n")

current_prediction = "---"
current_confidence = 0.0
hand_detected      = False   # ← NEW: track hand presence

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame   = cv2.flip(frame, 1)
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # ── NEW: Check if hand is present ────────────────
    hand_detected = results.multi_hand_landmarks is not None

    # ── Extract features ─────────────────────────────
    feats = extract_landmarks(results)
    frame_buffer.append(feats)

    # Draw hand landmarks
    if hand_detected:
        for hlm in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hlm, mp_hands.HAND_CONNECTIONS
            )

    # ── CNN-GRU Prediction ────────────────────────────
    if len(frame_buffer) == SEQ_LEN and prediction_cooldown == 0:

        # ── FIX: Only predict if hand is detected ─────
        if not hand_detected:
            # No hand → clear prediction display
            # Do NOT run model on zero features
            current_prediction = "---"
            current_confidence = 0.0
            reset_stability()

        else:
            raw_class_id, raw_confidence = predict_gesture(frame_buffer)

            if (raw_class_id is not None
                    and raw_confidence >= CONFIDENCE_THRESHOLD):
                recent_preds.append(raw_class_id)
                majority_class      = Counter(recent_preds).most_common(1)[0][0]
                current_prediction  = LABEL_NAMES.get(majority_class, "?")
                current_confidence  = raw_confidence
                prediction_cooldown = PREDICTION_COOLDOWN
                try_accept_prediction(majority_class, raw_confidence)

            else:
                current_prediction = "Low Conf"
                current_confidence = raw_confidence \
                                     if raw_class_id is not None else 0.0
                reset_stability()

    if prediction_cooldown > 0:
        prediction_cooldown -= 1

    # ── Keyboard ──────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        commit_word()
    elif key == 8:
        backspace_action()
    elif key == ord('c'):
        clear_action()

    # ════════════════════════════════════════════════
    # DRAW UI
    # ════════════════════════════════════════════════
    h, w = frame.shape[:2]

    # ── Hand status indicator (NEW) ───────────────────
    if hand_detected:
        cv2.circle(frame, (w-20, 20), 10, (0, 255, 0), -1)   # green dot
        cv2.putText(frame, "Hand ON",
                    (w-100, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)
    else:
        cv2.circle(frame, (w-20, 20), 10, (0, 0, 255), -1)   # red dot
        cv2.putText(frame, "No Hand",
                    (w-100, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1)

    # ── 1. Prediction ─────────────────────────────────
    draw_gujarati_label(
        frame,
        english_label  = "Prediction : ",
        gujarati_value = current_prediction,
        pos            = (10, 10),
        font           = font_large,
        label_color    = (0, 255, 0),
        value_color    = (0, 255, 0)
    )

    # ── 2. Confidence ─────────────────────────────────
    cv2.putText(
        frame,
        f"Confidence : {current_confidence:.2f}",
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 255, 255), 2
    )

    # ── 3. Stability ──────────────────────────────────
    cv2.putText(
        frame,
        f"Stability  : {stability_counter}/{STABILITY_COUNT_REQUIRED}",
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (200, 200, 255), 2
    )

    # ── 4. Progress bar ───────────────────────────────
    bar_x, bar_y, bar_w = 10, 110, 200
    filled = int(bar_w
                 * min(stability_counter, STABILITY_COUNT_REQUIRED)
                 / STABILITY_COUNT_REQUIRED)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + 10), (50,50,50), -1)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + filled, bar_y + 10), (0,200,100), -1)

    # ── 5. Building ───────────────────────────────────
    building_str = get_current_word()
    build_color  = (255, 200, 0) if building_str else (100, 100, 100)
    prefix = "Building   : ["
    cv2.rectangle(frame, (6, 123), (6+340, 123+38), (0,0,0), -1)
    cv2.putText(frame, prefix,
                (10, 148),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75, build_color, 2)
    (prefix_w, _), _ = cv2.getTextSize(
        prefix, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2
    )
    if building_str:
        put_gujarati_text(
            frame,
            text     = building_str,
            pos      = (10 + prefix_w, 124),
            font     = font_medium,
            color    = build_color,
            bg_color = None
        )
    digit_w = len(building_str) * 22 if building_str else 0
    cv2.putText(frame, "]",
                (10 + prefix_w + digit_w, 148),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75, build_color, 2)

    # ── 6. Cooldown ───────────────────────────────────
    if accept_cooldown > 0:
        cv2.putText(
            frame,
            f"Cooldown   : {accept_cooldown}",
            (10, 172),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (50, 50, 255), 2
        )

    # ── 7. Sentence panel ─────────────────────────────
    display_text  = get_display_sentence()
    wrapped_lines = wrap_gujarati_text(display_text, max_chars=20)

    panel_h   = 50 + 42 * max(len(wrapped_lines), 1)
    panel_top = h - panel_h - 10

    cv2.rectangle(frame, (5, panel_top), (w-5, h-10),
                  (20,20,20), -1)
    cv2.rectangle(frame, (5, panel_top), (w-5, h-10),
                  (70,70,70), 1)

    cv2.putText(frame, "Recognized Text:",
                (10, panel_top + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (180,180,180), 1)

    for i, line in enumerate(wrapped_lines):
        put_gujarati_text(
            frame,
            text  = line,
            pos   = (10, panel_top + 28 + i * 42),
            font  = font_large,
            color = (255, 255, 255)
        )

    # ── 8. Controls hint ──────────────────────────────
    cv2.putText(
        frame,
        "SPACE=word | BKSP=del | C=clear | Q=quit",
        (5, h-8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42, (150,150,150), 1
    )

    cv2.imshow("ISL | Gujarati Sign Language Translator", frame)

cap.release()
cv2.destroyAllWindows()
hands.close()