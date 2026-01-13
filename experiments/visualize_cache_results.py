# experiments/visualize_cache_results.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os
import json

# -----------------------------
# CONFIG
# -----------------------------
RESULTS_FILE = os.path.join("results", "cache_test_results.json")  # path to JSON output of load test
OUTPUT_DIR = os.path.join("results", "visualizations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
with open(RESULTS_FILE, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Convert timestamp to seconds if needed
if "timestamp" in df.columns:
    df["time_sec"] = df["timestamp"] - df["timestamp"].min()

print("Loaded data:")
print(df.head())

# -----------------------------
# 1. Boxplot: Latency by cache hit
# -----------------------------
plt.figure(figsize=(8,6))
sns.boxplot(x="cache_hit", y="latency_ms", data=df)
plt.title("Latency by Cache Hit")
plt.xlabel("Cache Hit")
plt.ylabel("Latency (ms)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "latency_by_cache_hit.png"))
plt.close()
print("Saved boxplot: latency_by_cache_hit.png")

# -----------------------------
# 2. Line chart: Cache hit rate over time
# -----------------------------
df["hit_numeric"] = df["cache_hit"].apply(lambda x: 1 if x else 0)
# Compute rolling hit rate (window=10 requests)
df["hit_rate"] = df["hit_numeric"].rolling(10, min_periods=1).mean()

plt.figure(figsize=(10,6))
plt.plot(df["time_sec"], df["hit_rate"], marker='o', linestyle='-', alpha=0.7)
plt.title("Cache Hit Rate Over Time (rolling 10 requests)")
plt.xlabel("Time (s)")
plt.ylabel("Cache Hit Rate")
plt.ylim(0,1.05)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cache_hit_rate_over_time.png"))
plt.close()
print("Saved line chart: cache_hit_rate_over_time.png")

# -----------------------------
# 3. Interactive boxplot with Plotly (optional)
# -----------------------------
fig = px.box(df, x="cache_hit", y="latency_ms", points="all", title="Latency by Cache Hit (Interactive)")
fig.write_html(os.path.join(OUTPUT_DIR, "latency_by_cache_hit_interactive.html"))
print("Saved interactive chart: latency_by_cache_hit_interactive.html")

print(f"All visualizations saved in {OUTPUT_DIR}")
