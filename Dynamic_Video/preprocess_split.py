"""
Preprocess dynamic sign clips and split into train / val / test sets.

Input : raw_dynamic_signs.csv  (long format: clip_id, frame_number, label,
        [session_id], f0..f62 -- RAW image-relative MediaPipe coordinates)

Output: train.csv, val.csv, test.csv (long format, every clip padded/
        truncated to SEQ_LEN frames)

KEY FIX vs the old version:
  The old script z-scored RAW image-relative coordinates. The real-time
  inference script, however, converts landmarks to WRIST-RELATIVE,
  SCALE-NORMALIZED coordinates before z-scoring. That mismatch is why
  the trained model saw a completely different input distribution at
  inference time than at training time (causing near-constant / wrong
  predictions despite 99% test accuracy).

  Fix: apply the identical wrist-relative + scale-normalization here,
  BEFORE computing/splitting/z-scoring. feature_mean.csv / feature_std.csv
  then describe the same feature space the inference script produces.

Splitting is done at the CLIP level, and (if session_id is present)
stratified so that all splits contain a mix of sessions -- this keeps the
val/test accuracy honest instead of overestimating due to same-session
"shortcut" cues (background, lighting, hand distance from camera).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

RAW_CSV = "raw_dynamic_signs.csv"
SEQ_LEN = 30
NUM_LANDMARKS = 21
FEATURE_COLS = [f"f{i}" for i in range(63)]

TRAIN_CSV = "train.csv"
VAL_CSV = "val.csv"
TEST_CSV = "test.csv"

VAL_FRAC = 0.15
TEST_FRAC = 0.15
RANDOM_STATE = 42


# --------------------------------------------------------------------
# THE FIX: wrist-relative, scale-normalized landmarks
# --------------------------------------------------------------------
def normalize_landmarks_relative(row_feats: np.ndarray) -> np.ndarray:
    """
    row_feats: length-63 RAW (x,y,z)*21 for one frame.
    Returns wrist-relative, scale-normalized features -- identical
    transform to the one used in the real-time inference script.
    """
    coords = row_feats.reshape(NUM_LANDMARKS, 3)
    wrist = coords[0]
    rel = coords - wrist
    max_val = np.max(np.abs(rel))
    if max_val < 1e-6:
        max_val = 1.0
    return (rel / max_val).flatten()


def pad_or_truncate_clip(clip_df: pd.DataFrame, seq_len: int) -> pd.DataFrame:
    """Force every clip to be exactly seq_len frames long."""
    clip_df = clip_df.sort_values("frame_number").reset_index(drop=True)
    n = len(clip_df)

    if n == seq_len:
        return clip_df

    if n > seq_len:
        idx = np.linspace(0, n - 1, seq_len).astype(int)
        out = clip_df.iloc[idx].reset_index(drop=True)
        out["frame_number"] = range(seq_len)
        return out

    pad_needed = seq_len - n
    last_row = clip_df.iloc[[-1]]
    pad_rows = pd.concat([last_row] * pad_needed, ignore_index=True)
    out = pd.concat([clip_df, pad_rows], ignore_index=True)
    out["frame_number"] = range(seq_len)
    out["clip_id"] = clip_df["clip_id"].iloc[0]
    out["label"] = clip_df["label"].iloc[0]
    return out


def normalize_features_zscore(df: pd.DataFrame, feature_cols, mean=None, std=None):
    """Z-score normalize using train statistics."""
    if mean is None or std is None:
        mean = df[feature_cols].mean()
        std = df[feature_cols].std().replace(0, 1)
    df = df.copy()
    df[feature_cols] = (df[feature_cols] - mean) / std
    return df, mean, std


def main():
    raw = pd.read_csv(RAW_CSV)

    if "session_id" not in raw.columns:
        print("NOTE: no session_id column found in raw CSV -- treating all "
              "rows as a single session. Consider re-collecting with the "
              "updated collection script for better generalization.")
        raw["session_id"] = "session_0"

    # 1. Apply wrist-relative + scale normalization to every frame's raw features
    feat_matrix = raw[FEATURE_COLS].values
    normalized = np.array([normalize_landmarks_relative(r) for r in feat_matrix])
    raw[FEATURE_COLS] = normalized

    # 2. Pad/truncate every clip to SEQ_LEN frames
    fixed_clips = [pad_or_truncate_clip(clip_df, SEQ_LEN)
                   for _, clip_df in raw.groupby("clip_id")]
    fixed = pd.concat(fixed_clips, ignore_index=True)

    # 3. Split at the CLIP level, stratified by label
    clip_meta = fixed.groupby("clip_id").agg(
        label=("label", "first"),
        session_id=("session_id", "first"),
    ).reset_index()

    train_ids, temp_ids = train_test_split(
        clip_meta, test_size=(VAL_FRAC + TEST_FRAC),
        stratify=clip_meta["label"], random_state=RANDOM_STATE,
    )
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC),
        stratify=temp_ids["label"], random_state=RANDOM_STATE,
    )

    train_df = fixed[fixed["clip_id"].isin(train_ids["clip_id"])].copy()
    val_df = fixed[fixed["clip_id"].isin(val_ids["clip_id"])].copy()
    test_df = fixed[fixed["clip_id"].isin(test_ids["clip_id"])].copy()

    # 4. Z-score normalize using TRAIN statistics only
    train_df, mean, std = normalize_features_zscore(train_df, FEATURE_COLS)
    val_df, _, _ = normalize_features_zscore(val_df, FEATURE_COLS, mean, std)
    test_df, _, _ = normalize_features_zscore(test_df, FEATURE_COLS, mean, std)

    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)
    mean.to_csv("feature_mean.csv")
    std.to_csv("feature_std.csv")

    print(f"Clips -> train:{len(train_ids)} val:{len(val_ids)} test:{len(test_ids)}")
    print(f"Rows  -> train:{len(train_df)} val:{len(val_df)} test:{len(test_df)}")
    print(f"Sessions in data: {sorted(raw['session_id'].unique())}")
    print(f"Each clip is fixed at SEQ_LEN={SEQ_LEN} frames.")
    print("Features are now wrist-relative + scale-normalized, "
          "then z-scored -- matching the real-time inference pipeline.")


if __name__ == "__main__":
    main()