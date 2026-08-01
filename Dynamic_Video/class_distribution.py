import pandas as pd

df = pd.read_csv("train.csv")

# Count number of clips per label
clip_counts = df.groupby("label")["clip_id"].nunique().sort_index()

print("=== Number of Clips per Class in Training Set ===\n")
print(clip_counts)
print("\nTotal clips:", clip_counts.sum())

# Also show percentage
print("\n=== Percentage Distribution ===")
print((clip_counts / clip_counts.sum() * 100).round(2))