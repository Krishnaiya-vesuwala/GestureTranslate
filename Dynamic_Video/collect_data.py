"""
Dynamic Sign Language Data Collection Script (Session-aware version)

Captures SEQUENCES (clips) of hand-landmark frames for dynamic gestures.
Saves RAW (image-relative) landmarks -- normalization happens later in
preprocessing, applied identically to train and inference data.

NEW vs old version:
  - Tracks a `session_id` per run so you can (and should) record across
    multiple sessions with different lighting / position / background.
  - 3-2-1 countdown before recording so every clip starts from a
    consistent "ready" hand pose instead of a random mid-motion frame.
  - Prints a randomized label order suggestion each session to avoid
    always recording 0,1,2...9 in the same sequence (which can
    correlate label with fatigue/drift over the session).
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

# --------------------- CONFIG ---------------------
LABEL_NAMES = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
    5: "5", 6: "6", 7: "7", 8: "8", 9: "9"
}

RECORD_LEN = 30
OUTPUT_CSV = "raw_dynamic_signs.csv"
CAM_INDEX = 0
COUNTDOWN_SECONDS = 3
CLIPS_PER_LABEL_TARGET = 40   # per session -- see README for full protocol

NUM_LANDMARKS = 21
COORDS = 3
FEATURES_PER_HAND = NUM_LANDMARKS * COORDS   # 63 features
TOTAL_FEATURES = FEATURES_PER_HAND

# Explicit header, defined once, used everywhere a row is written or
# appended. This is the single source of truth for column order -- we
# do NOT rely on pandas/dict key ordering to get this right, since that
# silently breaks if the row-building code ever changes order.
HEADER = ["clip_id", "frame_number", "label", "session_id"] + \
         [f"f{i}" for i in range(TOTAL_FEATURES)]

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

SESSION_ID = str(uuid.uuid4())[:8]


def ensure_header(csv_path):
    """
    Make sure csv_path exists and starts with the correct header.
    - If the file doesn't exist: create it with just the header row.
    - If it exists: verify the first line matches HEADER exactly, and
      fail loudly (instead of silently corrupting the file) if not.
    """
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            f.write(",".join(HEADER) + "\n")
        print(f"Created {csv_path} with header:\n  {','.join(HEADER)}")
        return

    with open(csv_path, "r") as f:
        first_line = f.readline().strip()

    if first_line != ",".join(HEADER):
        raise ValueError(
            f"Header mismatch in {csv_path}!\n"
            f"  Found    : {first_line}\n"
            f"  Expected : {','.join(HEADER)}\n"
            f"This file was likely created by an older version of this "
            f"script (e.g. missing session_id) or has a corrupted/missing "
            f"header. Fix it with fix_missing_header.py, or rename this "
            f"file and let the script create a fresh one."
        )


def extract_landmarks(results):
    """Returns 63 RAW features for ONE hand. If no hand detected -> zeros."""
    features = np.zeros(FEATURES_PER_HAND)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        coords = []
        for lm in hand_landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        features = np.array(coords)
    return features


def save_clip_to_csv(rows, csv_path):
    """
    rows: list of dicts, each already containing keys matching HEADER
    (clip_id, frame_number, label, session_id, f0..f62).
    Writes using csv.DictWriter with fieldnames=HEADER explicitly, so
    column order is always exactly HEADER -- never dependent on dict
    insertion order or pandas' inferred column order.
    """
    if not rows:
        return

    ensure_header(csv_path)  # creates file+header if missing, else validates it

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

    current_label = None
    recording = False
    countdown_active = False
    countdown_start = None
    frame_buffer = []
    clip_id = 0
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
    print("Suggested recording order this session (randomized):", order)
    print("\nControls:")
    print("  0-9 -> Select label")
    print("  r   -> Start countdown + recording clip")
    print("  q   -> Quit and save\n")
    print(f"Target clips/label this session: {CLIPS_PER_LABEL_TARGET}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hlm in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)

        label_text = LABEL_NAMES.get(current_label, "None")
        cv2.putText(frame, f"Label: {label_text}  (clips so far: {clips_this_session.get(current_label, 0)})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Countdown before recording
        if countdown_active:
            elapsed = time.time() - countdown_start
            remaining = COUNTDOWN_SECONDS - elapsed
            if remaining > 0:
                cv2.putText(frame, f"Get ready: {int(remaining) + 1}",
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
            else:
                countdown_active = False
                recording = True
                frame_buffer = []
                print(f"Recording started for label {current_label}")

        # Recording logic
        if recording:
            feats = extract_landmarks(results)
            frame_buffer.append(feats)

            cv2.putText(frame, f"REC {len(frame_buffer)}/{RECORD_LEN}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if len(frame_buffer) >= RECORD_LEN:
                clip_rows = []
                for i, feats in enumerate(frame_buffer):
                    row = {
                        "clip_id": clip_id,
                        "frame_number": i,
                        "label": current_label,
                        "session_id": SESSION_ID,
                    }
                    row.update({f"f{j}": float(feats[j]) for j in range(TOTAL_FEATURES)})
                    clip_rows.append(row)

                save_clip_to_csv(clip_rows, OUTPUT_CSV)
                print(f"Saved clip_id={clip_id} | Label: {label_text} ({RECORD_LEN} frames)")

                clips_this_session[current_label] = clips_this_session.get(current_label, 0) + 1
                clip_id += 1
                frame_buffer = []
                recording = False

        cv2.imshow("Dynamic Sign Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in [ord(str(d)) for d in range(10)]:
            current_label = int(chr(key))
            print(f"Label selected: {current_label} ({LABEL_NAMES.get(current_label)})")

        elif key == ord('r'):
            if current_label is None:
                print("WARNING: Select a label first (press 0-9)!")
            elif not recording and not countdown_active:
                countdown_active = True
                countdown_start = time.time()

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if frame_buffer:
        print("Warning: Partial clip was not saved.")

    print("\nSession summary:", clips_this_session)
    print("Collection finished. Data saved to:", OUTPUT_CSV)


if __name__ == "__main__":
    main()