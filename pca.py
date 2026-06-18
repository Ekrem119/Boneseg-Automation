import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 18

def get8DData(filename):
    """
    Reads a text file and returns an 8D numpy array.
    """
    DATA = np.empty((0,8))
    with open(filename, "r") as dataFile:
        header = next(dataFile)
        for line in dataFile:
            line = line.replace(",", ".")
            xx, yy, zz, aa, bb, cc, dd, ee = line.split()
            DATA = np.vstack((DATA, [[
                float(xx), float(yy), float(zz), float(aa),
                float(bb), float(cc), float(dd), float(ee)
            ]]))
    return DATA

# --- 1) Load and Process Data ---
DATAin = get8DData('results.txt')

# Standardisation
mu = np.mean(DATAin, axis=0)
sig = np.std(DATAin, axis=0)
DATA_std = (DATAin - mu) / sig

# Calcul PCA
pca_obj = PCA(n_components=2)
DATA_pca = pca_obj.fit_transform(DATA_std)

explained_var = pca_obj.explained_variance_ratio_ * 100

# --- 2) Plot ---
fig, ax = plt.subplots(figsize=(8, 6))

# Plot Anat samples
ax.scatter(DATA_pca[:8, 0], DATA_pca[:8, 1],
           c='red', label='anat', alpha=0.8, edgecolors='k', s=80)

# Plot Koek samples
ax.scatter(DATA_pca[8:, 0], DATA_pca[8:, 1],
           c='blue', label='koek', alpha=0.8, edgecolors='k', s=80)

ax.set_xlabel(f"PC1 ({explained_var[0]:.1f}%)")
ax.set_ylabel(f"PC2 ({explained_var[1]:.1f}%)")
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.grid(alpha=0.3, linestyle='--')
fig.tight_layout()


plt.show()
