#!/usr/bin/env python3
"""
Genereaza figura `cap5-scores.pdf` pentru capitolul 5 (sectiunea Proiectare).

Distributia ilustrativa a scorurilor de normalitate produse de
IsolationForest, cu cele doua praguri marcate vertical:
  - global_low  = -0.022  (severity 1; percentila 2.0%)
  - global_high = -0.102  (severity 2; percentila 0.2%)

Pragurile sunt citite din meta.json al modelului antrenat
(`masina_invata/isolation_forest/beth_iforest_model_host2tier/meta.json`).
Distributia este generata sintetic dintr-un amestec calibrat sa
respecte percentilele empirice ale modelului (188.967 scoruri).

Reproducere:
  python3 figuri/cap5-scores.py
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
META = REPO / "masina_invata" / "isolation_forest" / "beth_iforest_model_host2tier" / "meta.json"
OUT  = HERE / "cap5-scores.pdf"

with open(META) as f:
    meta = json.load(f)

low_thr  = meta["global_thresholds"]["low"]
high_thr = meta["global_thresholds"]["high"]
p_low    = meta["low_percentile"]
p_high   = meta["high_percentile"]
n_train  = meta["global_thresholds"]["n_scores"]

# Genereaza scoruri ilustrative: amestec de doua Gaussiene
# care reproduc empiric forma scorurilor de IsolationForest pe BETH.
rng = np.random.default_rng(seed=42)
n = 200_000
benign = rng.normal(loc=0.08, scale=0.045, size=int(n * 0.985))
anomaly = rng.normal(loc=-0.08, scale=0.045, size=int(n * 0.015))
scores = np.concatenate([benign, anomaly])

# Calibreaza scorurile sintetice la percentilele reale ale modelului.
# Ajustam locatia astfel incat P2.0 si P0.2 sa coincida cu pragurile reale.
p2_sample  = float(np.percentile(scores, p_low))
p02_sample = float(np.percentile(scores, p_high))
# Mapare liniara: (p02_sample, p2_sample) -> (high_thr, low_thr)
m = (low_thr - high_thr) / (p2_sample - p02_sample) if p2_sample != p02_sample else 1.0
b = low_thr - m * p2_sample
scores = m * scores + b

fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.hist(scores, bins=140, density=True, color="#3b6fb6", alpha=0.85,
        edgecolor="white", linewidth=0.3)
ax.axvline(low_thr,  color="#d97706", linestyle="--", linewidth=1.4,
           label=f"prag severity 1 ({low_thr:.3f}, P{p_low}%)")
ax.axvline(high_thr, color="#b91c1c", linestyle="--", linewidth=1.4,
           label=f"prag severity 2 ({high_thr:.3f}, P{p_high}%)")

# Anotari pentru cele trei zone
ymax = ax.get_ylim()[1]
ax.text(low_thr + 0.05, ymax * 0.85, "benign", ha="left", va="top",
        fontsize=10, color="#1f2937")
ax.text((low_thr + high_thr) / 2, ymax * 0.5, "minor", ha="center",
        va="top", fontsize=10, color="#92400e")
ax.text(high_thr - 0.02, ymax * 0.85, "major", ha="right", va="top",
        fontsize=10, color="#991b1b")

ax.set_xlabel("scor de normalitate (IsolationForest.decision_function)")
ax.set_ylabel("densitate")
ax.set_xlim(-0.30, 0.30)
ax.legend(loc="upper left", fontsize=9, frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.25, linewidth=0.4)

fig.tight_layout()
fig.savefig(OUT, format="pdf", bbox_inches="tight")
print(f"[OK] {OUT}  (n={n:,}, P{p_low}%={low_thr:.4f}, P{p_high}%={high_thr:.4f})")
