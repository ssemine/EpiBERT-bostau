import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

counts_dir = "/home/sstyopa/data/bovineATAC/data_processing/counts"
files = sorted([f for f in os.listdir(counts_dir) if f.endswith(".txt")])
counts = []
sample_names = []

for f in files:
    path = os.path.join(counts_dir, f)
    if os.path.getsize(path) == 0:
        print(f"Skipping empty file: {f}")
        continue
    df = pd.read_csv(path, header=None)
    counts.append(df)
    sample_names.append(f.replace(".txt", ""))

# Combine into single matrix
matrix = pd.concat(counts, axis=1)
matrix.columns = sample_names

# Compute Pearson correlation
corr = matrix.corr(method="pearson")

# Plot
sns.set(style="white", font_scale=0.8)
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="vlag", square=True, annot=False, linewidths=0.1)
plt.title("ATAC-seq sample correlation")
plt.tight_layout()
plt.savefig("../plots/corr_heatmap.png", dpi=300)
plt.show()
