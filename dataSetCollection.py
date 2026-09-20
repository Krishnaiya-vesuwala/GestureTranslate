"""
Dynamic Sign Language Data Collection Script
ગુજરાતી લેબલ પ્રદર્શન સાથે (with Gujarati label display)
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import csv
import time
import random
import uuid
from PIL import ImageFont, ImageDraw, Image

# --------------------- CONFIG ---------------------
# Internal labels (integers) used for CSV storage
# Display labels shown on screen (Gujarati)
LABEL_NAMES = {
    0: "૦", 1: "૧", 2: "૨", 3: "૩", 4: "૪",
    5: "૫", 6: "૬", 7: "૭", 8: "૮", 9: "૯"
}

RECORD_LEN             = 30
OUTPUT_CSV             = "raw_dynamic_signs.csv"
CAM_INDEX              = 0
COUNTDOWN_SECONDS      = 3
CLIPS_PER_LABEL_TARGET = 40

NUM_LANDMARKS     = 21
COORDS            = 3
FEATURES_PER_HAND = NUM_LANDMARKS * COORDS
TOTAL_FEATURES    = FEATURES_PER_HAND

HEADER = ["clip_id", "frame_number", "label", "session_id"] + \
         [f"f{i}" for i in range(TOTAL_FEATURES)]

mp_hands   = mp.solutions.hands
mp_draw    = mp.solutions.drawing_utils
SESSION_ID = str(uuid.uuid4())[:8]

# ── Font setup ──────────────────────────────────────
FONT_PATH = "NotoSansGujarati-Regular.ttf"

def load_font(size):
    if os.path.exists(FONT_PATH):
        return ImageFont.truetype(FONT_PATH, size)
    print(f"WARNING: '{FONT_PATH}' not found! Gujarati may not render.")
    return ImageFont.load_default()

font_medium = load_font(28)
font_small  = load_font(22)


def put_gujarati_text(frame, text, pos, font,
                      color=(255, 255, 255), bg_color=None):
    """Render Unicode/Gujarati text on OpenCV frame via PIL."""
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)
    x, y   = pos
    if bg_color is not None:
        bbox = draw.textbbox((x, y), text, font=font)
        draw.rectangle(
            [bbox[0]-4, bbox[1]-4, bbox[2]+4, bbox[3]+4],
            fill=bg_color
        )
    draw.text((x, y), text, font=font, fill=color)
    frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def ensure_header(csv_path):
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            f.write(",".join(HEADER) + "\n")
        return
    with open(csv_path, "r") as f:
        first_line = f.readline().strip()
    if first_line != ",".join(HEADER):
        raise ValueError(f"Header mismatch in {csv_path}!")


def extract_landmarks(results):
    features = np.zeros(FEATURES_PER_HAND)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        coords = []
        for lm in hand_landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        features = np.array(coords)
    return features


def save_clip_to_csv(rows, csv_path):
    if not rows:
        return
    ensure_header(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        for row in rows:
            writer.writerow(row)
    print(f"  -> Saved {len(rows)} rows to {csv_path}")


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    current_label    = None
    recording        = False
    countdown_active = False
    countdown_start  = None
    frame_buffer     = []
    clip_id          = 0
    clips_this_session = {k: 0 for k in LABEL_NAMES}

    if os.path.exists(OUTPUT_CSV):
        try:
            existing = pd.read_csv(OUTPUT_CSV)
            if len(existing) > 0:
                clip_id = int(existing["clip_id"].max()) + 1
        except Exception:
            pass

    order = list(LABEL_NAMES.keys())
    random.shuffle(order)

    print(f"\nSession ID: {SESSION_ID}")
    print("રેકોર્ડિંગ ક્રમ (Recording order):", order)
    print("\nનિયંત્રણો (Controls):")
    print("  0-9 → લેબલ પસંદ કરો (Select label)")
    print("  r   → રેકોર્ડિંગ શરૂ કરો (Start recording)")
    print("  q   → બહાર નીકળો (Quit)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame   = cv2.flip(frame, 1)
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hlm in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)

        # Gujarati label display
        label_text   = LABEL_NAMES.get(current_label, "કોઈ નહીં")  # "None"
        clips_so_far = clips_this_session.get(current_label, 0)

        put_gujarati_text(
            frame,
            text     = f"લેબલ: {label_text}  (ક્લિપ્સ: {clips_so_far}/{CLIPS_PER_LABEL_TARGET})",
            pos      = (10, 8),
            font     = font_medium,
            color    = (0, 255, 0),
            bg_color = (0, 0, 0)
        )

        # Countdown
        if countdown_active:
            elapsed   = time.time() - countdown_start
            remaining = COUNTDOWN_SECONDS - elapsed
            if remaining > 0:
                put_gujarati_text(
                    frame,
                    text     = f"તૈયાર રહો: {int(remaining) + 1}",  # "Get ready"
                    pos      = (10, 90),
                    font     = font_medium,
                    color    = (0, 165, 255),
                    bg_color = (0, 0, 0)
                )
            else:
                countdown_active = False
                recording        = True
                frame_buffer     = []
                print(f"  રેકોર્ડિંગ શરૂ: લેબલ {current_label} ({label_text})")

        # Recording
        if recording:
            feats = extract_landmarks(results)
            frame_buffer.append(feats)

            put_gujarati_text(
                frame,
                text     = f"રેક {len(frame_buffer)}/{RECORD_LEN}",
                pos      = (10, 50),
                font     = font_medium,
                color    = (0, 0, 255),
                bg_color = (0, 0, 0)
            )

            if len(frame_buffer) >= RECORD_LEN:
                clip_rows = []
                for i, f_feats in enumerate(frame_buffer):
                    row = {
                        "clip_id":      clip_id,
                        "frame_number": i,
                        "label":        current_label,
                        "session_id":   SESSION_ID,
                    }
                    row.update(
                        {f"f{j}": float(f_feats[j])
                         for j in range(TOTAL_FEATURES)}
                    )
                    clip_rows.append(row)

                save_clip_to_csv(clip_rows, OUTPUT_CSV)
                print(f"  Saved clip_id={clip_id} | Label: {label_text}")

                clips_this_session[current_label] = \
                    clips_this_session.get(current_label, 0) + 1
                clip_id     += 1
                frame_buffer = []
                recording    = False

        # Controls hint
        put_gujarati_text(
            frame,
            text  = "0-9=લેબલ | r=રેક | q=બહાર",
            pos   = (5, frame.shape[0] - 30),
            font  = font_small,
            color = (150, 150, 150)
        )

        cv2.imshow("ગુજરાતી સાંકેતિક ભાષા ડેટા સંગ્રહ", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in [ord(str(d)) for d in range(10)]:
            current_label = int(chr(key))
            print(f"  લેબલ: {current_label} ({LABEL_NAMES.get(current_label)})")

        elif key == ord('r'):
            if current_label is None:
                print("ચેતવણી: પહેલા લેબલ પસંદ કરો!")
            elif not recording and not countdown_active:
                countdown_active = True
                countdown_start  = time.time()

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\nસત્ર સારાંશ (Session Summary):")
    for lbl, count in clips_this_session.items():
        print(f"  {LABEL_NAMES.get(lbl,'?')} ({lbl}): {count} clips")
    print(f"\nડેટા સાચવ્યો: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()