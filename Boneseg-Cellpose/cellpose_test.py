import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nd2reader import ND2Reader
from cellpose import models, plot
from scipy.ndimage import binary_dilation
import os

# --- 1. SETUP ---
# CHANGE THIS to the KOEK file you are testing
FILE_PATH = r"C:\Users\ekrem\Desktop\Internship\Bone-Sample\11-05-2026\Koek-H-TI_A2\alteration.nd2"
print(f"Loading image: {os.path.basename(FILE_PATH)}...")

# --- 2. EXTRACT RAW DAPI TENSOR ---
with ND2Reader(FILE_PATH) as images:
    channels = images.metadata.get('channels', [])
    c_idx = next((i for i, c in enumerate(channels) if "DAPI" in c.upper() or "405" in str(c)), 0)
    
    img = np.max(np.stack([images.get_frame_2D(c=c_idx, z=z) for z in range(images.sizes.get('z', 1))]), axis=0).astype(np.float32)

img_8bit = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)

# --- 3. RUN CELLPOSE AI ---
print("Running Cellpose topological segmentation...")
model = models.CellposeModel(gpu=True, model_type='cyto2')
outputs = model.eval(img_8bit, diameter=None, channels=[0,0])
masks, flows = outputs[0], outputs[1]

cell_ids = np.unique(masks)[1:] 
print(f"✅ AI isolated {len(cell_ids)} true biological structures.\n")

# --- 4. BONESEG MATHEMATICAL ENGINE ---
print("Executing IO/IB Boundary Subtraction Mathematics...")
results = []

for cell_id in cell_ids:
    cell_mask = (masks == cell_id)
    area = np.sum(cell_mask)
    
    border_mask = binary_dilation(cell_mask, iterations=5) ^ cell_mask
    
    mean_in = np.mean(img[cell_mask])
    mean_out = np.mean(img[border_mask])
    ratio = mean_in / (mean_out + 1e-8)
    
    results.append({
        "Cell_ID": cell_id,
        "Area": area,
        "Inside_Signal": round(mean_in, 2),
        "Border_Signal": round(mean_out, 2),
        "DAPI_Ratio": round(ratio, 3)
    })

# --- 5. DATA OUTPUT ---
df = pd.DataFrame(results)
print("\n--- AUTOMATED BATCH EXTRACTION RESULTS ---")
print(df.head(15).to_string(index=False))
print("\n" + "="*50)
print(f"FINAL AUTOMATED KOEK SCORE: {df['DAPI_Ratio'].mean():.3f}")
print("="*50)

# Save to CSV
desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
df.to_csv(os.path.join(desktop, "Cellpose_Proof_of_Concept.csv"), index=False)

# --- 6. VISUAL VERIFICATION (THE "SANITY CHECK") ---
print("Launching Visualizer... Please close the pop-up window to exit.")
fig = plt.figure(figsize=(12, 6))

# Show the original image and the segmented masks
plot.show_segmentation(fig, img_8bit, masks, flows[0], channels=[0,0])

# Add ID numbers on top of the masks so you can check the table
ax = fig.axes[2] # Target the "predicted masks" panel
for cell_id in cell_ids:
    y, x = np.where(masks == cell_id)
    if len(y) > 0:
        ax.text(x.mean(), y.mean(), str(cell_id), color='white', fontsize=8, ha='center', va='center')

plt.suptitle(f"Cellpose Validation - Mean DAPI: {df['DAPI_Ratio'].mean():.3f}", fontsize=14)
plt.tight_layout()
plt.show()