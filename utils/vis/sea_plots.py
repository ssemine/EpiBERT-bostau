import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sample = "ADP.6.1M"

path_to_plots = f"../plots/consensus/{sample}"
os.makedirs(path_to_plots, exist_ok=True)

sea = f"/home/sstyopa/data/bovineATAC/data_processing/sea_out/consensus/{sample}/sea.tsv"

df = pd.read_csv(sea, sep='\t', comment='#')
top = df.sort_values("LOG_PVALUE", ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x="LOG_PVALUE", y="ALT_ID", data=top, palette="viridis")
plt.xlabel("log10(p-value)")
plt.ylabel("Motif")
plt.title("Top Enriched Motifs by -log10(p-value)")
plt.tight_layout()
plt.savefig(f"{path_to_plots}/motif_enrichment_barplot.png", dpi=300)
plt.close()

df["log2_enrichment"] = np.log2(df["ENR_RATIO"])

plt.figure(figsize=(8, 6))
sns.scatterplot(x="log2_enrichment", y="LOG_PVALUE", data=df, hue="ALT_ID", legend=False, s=50)
plt.axhline(y=-np.log10(0.05), color='red', linestyle='--')  # significance threshold
plt.xlabel("log2(Enrichment Ratio)")
plt.ylabel("log10(p-value)")
plt.title("Motif Enrichment Volcano Plot")
plt.tight_layout()
plt.savefig(f"{path_to_plots}/motif_volcano_plot.png", dpi=300)
plt.close()


plt.figure(figsize=(10, 5))
sns.scatterplot(x="RANK", y="ENR_RATIO", data=df)
plt.xlabel("Rank")
plt.ylabel("Enrichment Ratio")
plt.title("Enrichment Ratio by Motif Rank")
plt.tight_layout()
plt.savefig(f"{path_to_plots}/enrichment_ratio_by_rank.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
sns.histplot(df["LOG_PVALUE"], bins=30, kde=True)
plt.xlabel("P-value")
plt.ylabel("Frequency")
plt.title("Distribution of P-values")
plt.tight_layout()
plt.savefig(f"{path_to_plots}/pvalue_distribution.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
sns.scatterplot(x="PVALUE", y="QVALUE", data=df)
plt.xlabel("P-value")
plt.ylabel("Q-value")
plt.title("Q-value vs P-value")
plt.tight_layout()
plt.savefig(f"{path_to_plots}/qvalue_vs_pvalue.png", dpi=300)
plt.close()