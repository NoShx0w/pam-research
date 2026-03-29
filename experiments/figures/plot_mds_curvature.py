import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm

coords = pd.read_csv("outputs/fim_mds/mds_coords.csv")
curv = pd.read_csv("outputs/fim_curvature/curvature_surface.csv")

df = coords.merge(curv, on=["r", "alpha"], how="left")

fig, ax = plt.subplots(figsize=(7,5.5))

sc = ax.scatter(
    df.mds1,
    df.mds2,
    c=df.scalar_curvature,
    cmap="coolwarm",
    norm=SymLogNorm(linthresh=1),
    s=70
)

for _,row in df.iterrows():
    ax.text(row.mds1,row.mds2,f"({row.r:.2f},{row.alpha:.3f})",fontsize=7,alpha=0.7)

fig.colorbar(sc,label="scalar curvature")

ax.set_xlabel("MDS 1")
ax.set_ylabel("MDS 2")
ax.set_title("PAM Fisher manifold colored by scalar curvature")

fig.tight_layout()

fig.savefig("outputs/fim_mds/mds_curvature.png",dpi=180)

plt.close()
