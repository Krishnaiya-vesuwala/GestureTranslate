# Dynamic Sign Language Dataset — Full Pipeline (for 1D-CNN + GRU)

This folder contains three scripts that take you from raw webcam capture to a
model-ready dataset, matching the schema you described:

```
clip_id, frame_number, label, f0, f1, ..., f125
```

- **126 features** = 21 MediaPipe hand landmarks × 3 coordinates (x, y, z) × 2 hands
  (Left hand = f0–f62, Right hand = f63–f125; a missing hand is zero-filled)
- **clip_id** groups all frames of one gesture performance together
- **frame_number** is the frame's position (0, 1, 2, ...) inside that clip

## Files
| File | Purpose |
|---|---|
| `1_collect_data.py` | Captures webcam clips, extracts hand landmarks, appends to `raw_dynamic_signs.csv` |
| `2_preprocess_and_split.py` | Pads/truncates clips to a fixed length, splits by **clip** into train/val/test, normalizes, writes `train.csv` / `val.csv` / `test.csv` |
| `3_train_1dcnn_gru.py` | Reshapes CSV rows back into `(num_clips, SEQ_LEN, 126)` arrays and trains a 1D-CNN + GRU classifier |

Install once:
```bash
pip install opencv-python mediapipe pandas numpy scikit-learn tensorflow
```

---

## How to Collect the Data Properly

Dynamic signs depend on **motion**, so the collection process matters much more
than for static (single-frame) signs. Follow these rules:

### 1. Decide your gesture vocabulary first
List every dynamic gesture you want to recognize (e.g. "Hello", "Thank You",
"Please") and assign each a numeric label. Edit `LABEL_NAMES` in
`1_collect_data.py` accordingly.

### 2. Record each gesture as a full clip, not a single photo
Run `1_collect_data.py`, press the number key for the gesture, then press `r`.
It will automatically capture `RECORD_LEN` (default 30) consecutive frames —
this is one **clip**. Perform the full motion of the sign naturally during
those frames (don't freeze).

### 3. Collect many repetitions per gesture
- Aim for **at least 30–50 clips per gesture** (more is better — 100+ is ideal
  for good generalization).
- Vary: hand speed (slow/fast), starting hand position, distance from camera,
  slight rotation of the wrist/body.
- Record with **multiple people** if possible, and across **different sessions/
  days** (lighting, clothing, background changes).
- Keep some hand/arm movement natural at clip start and end (don't only
  capture the "peak" pose) — this is what teaches the GRU temporal structure.

### 4. Keep conditions realistic but varied
- Good, consistent lighting (avoid strong backlight).
- Plain background is easiest to start with; add busier backgrounds later for
  robustness once the model works.
- Keep hands inside the frame for the whole gesture.

### 5. Frame count per clip (RECORD_LEN)
- 8–15 frames: short/simple gestures.
- 20–30 frames: longer gestures (writing in air, multi-step motions).
- Keep it **consistent** while collecting raw data — the preprocessing script
  will pad/truncate to a fixed `SEQ_LEN` anyway, but similar raw lengths mean
  less distortion.

### 6. Avoid label imbalance
Try to collect a similar number of clips per gesture. If one gesture ends up
with far fewer clips, the model will be biased toward the more frequent ones.
`2_preprocess_and_split.py` uses **stratified** splitting to keep class ratios
consistent across train/val/test, but it can't fix an imbalance that exists in
the raw data.

---

## Step-by-Step Usage

### Step 1 — Collect raw data
```bash
python 1_collect_data.py
```
This produces `raw_dynamic_signs.csv`. Run it across multiple sessions — new
clips are appended (clip_id auto-increments) instead of overwriting.

### Step 2 — Preprocess + split
```bash
python 2_preprocess_and_split.py
```
This:
1. Pads (repeats last frame) or truncates (uniform sampling) every clip to a
   fixed `SEQ_LEN` (default 30 frames) so all sequences are equal length.
2. Splits **by clip_id** (70/15/15 train/val/test) so frames from the same
   gesture performance never appear in two different splits — this avoids
   inflated accuracy from leakage.
3. Z-score normalizes features using **train-set statistics only** (applied
   to val/test too, to avoid leakage).
4. Writes `train.csv`, `val.csv`, `test.csv` in the same long format, plus
   `feature_mean.csv` / `feature_std.csv` (needed to normalize new/live data
   the same way at inference time).

### Step 3 — Train the 1D-CNN + GRU model
```bash
python 3_train_1dcnn_gru.py
```
`load_sequences()` groups each CSV back by `clip_id` into a 3D array of shape
`(num_clips, SEQ_LEN, 126)`, which is exactly the input shape a `Conv1D` +
`GRU` model expects: `(batch, timesteps, features)`.

Model architecture used:
```
Input (SEQ_LEN, 126)
 -> Conv1D(64) -> BatchNorm -> Conv1D(128) -> BatchNorm -> MaxPool1D
 -> GRU(128, return_sequences=True) -> GRU(64)
 -> Dense(64) -> Dropout(0.4) -> Dense(num_classes, softmax)
```
The CNN layers pick up short local motion patterns (finger flicks, small
directional changes) and the GRU layers learn the longer-range temporal order
of the gesture — a good combination for gesture/sign sequences.

---

## Notes for Live Inference (later)
When you later run this model on a live webcam feed:
1. Extract landmarks the same way as `1_collect_data.py` (`extract_landmarks`)
   for each incoming frame.
2. Buffer the last `SEQ_LEN` frames in a sliding window.
3. Normalize with the **saved** `feature_mean.csv` / `feature_std.csv` (not
   newly computed stats).
4. Feed the `(1, SEQ_LEN, 126)` window into the trained model for prediction.
