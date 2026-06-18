import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import resize
from nd2reader import ND2Reader
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

from hydra import initialize
from hydra.core.global_hydra import GlobalHydra

import sys
sys.path.append(r"C:\Users\Bicel service\Desktop\USER\Maria\sam2")

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- Hydra init ---
GlobalHydra.instance().clear()
initialize(config_path="sam2/configs", version_base=None)

# --- Device selection ---
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

np.random.seed(3)

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    overlay = np.ones((sorted_anns[0]['segmentation'].shape[0],
                       sorted_anns[0]['segmentation'].shape[1], 4), dtype=np.float32)
    overlay[:, :, 3] = 0.0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        overlay[m] = color_mask
    ax.imshow(overlay)

# --- Choix du dossier ---
root = tk.Tk()
root.withdraw()
dir_path = filedialog.askdirectory(title="Choisir un dossier jour_mois_annee")
if not dir_path:
    messagebox.showinfo("Info", "Aucun dossier sélectionné.")
    raise SystemExit

# --- Paramètres ---
anat_vs_koek = simpledialog.askstring("Conditions expérimentales", "Lame récente (anat) ou archéologique (koek)?", initialvalue="anat")
num_lame = simpledialog.askinteger("Conditions expérimentales", "Numéro de la lame", initialvalue=0)
imagerie = simpledialog.askstring("Conditions expérimentales", "Imagerie (stitch-bf/fluo-spinning)", initialvalue="fluo-spinning")
if imagerie == "fluo-spinning":
    obj = simpledialog.askstring("Conditions expérimentales", "Objet imagé", initialvalue="blanc")
    channel_choice = simpledialog.askstring("Select channel", "Choisir un canal (DAPI, CFP, Brightfield, Cy3, YFP, GFP)", initialvalue="DAPI")

parent = os.path.dirname(os.path.dirname(dir_path))

# --- Charger ND2 ---
if imagerie == "fluo-spinning":
    nd2_path = f"{dir_path}/{anat_vs_koek}-{num_lame}/{anat_vs_koek}-{num_lame}-{imagerie}-{obj}.nd2"
else:
    nd2_path = f"{dir_path}/{anat_vs_koek}-{num_lame}/{anat_vs_koek}-{num_lame}-{imagerie}.nd2"
print("Chemin ND2 construit:", nd2_path)

if imagerie == "fluo-spinning":
    with ND2Reader(nd2_path) as images:
        Z = images.sizes.get('z', 1)
        C = images.sizes.get('c', len(images.metadata.get('channels', [])))
        channel_map = {"DAPI": 0, "CFP": 1, "BrightField": 2, "Cy3": 3, "YFP": 4, "GFP": 5}
        c_idx = channel_map[channel_choice]
        frames = []
        for z in range(Z):
            idx = z * C + c_idx
            if idx < len(images):
                frames.append(images[idx])
        channel_stack = np.stack(frames, axis=0)
        img2d = channel_stack.max(axis=0)
else:
    with ND2Reader(nd2_path) as images:
        img2d = images.get_frame(0)

# Optionnel: limiter la taille disque (SAM2 gère ses propres transforms)
if img2d.shape[0] > 4096 or img2d.shape[1] > 4096:
    img2d = resize(img2d, (4096, 4096), preserve_range=True, anti_aliasing=True)

# --- Sauvegarde PNG via Matplotlib (comme à l’entraînement) ---
dataset_validation_dir = os.path.join(parent, "dataset", "validation_blanc")
dataset_images_dir = os.path.join(dataset_validation_dir, "images_validation")
dataset_seg_dir = os.path.join(dataset_validation_dir, "seg_validation")
os.makedirs(dataset_images_dir, exist_ok=True)
os.makedirs(dataset_seg_dir, exist_ok=True)

filename = f"pretraitee-{anat_vs_koek}-{num_lame}-{imagerie}-{obj}.png" if imagerie == "fluo-spinning" \
           else f"pretraitee-{anat_vs_koek}-{num_lame}-{imagerie}.png"
png_path = os.path.join(dataset_images_dir, filename)

plt.figure()
plt.imshow(img2d, cmap="gray", aspect="equal")
plt.axis("off")
plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0)
plt.close()


from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

model_cfg=r"C:\Users\Bicel service\anaconda3\envs\sam_env\Lib\site-packages\sam2\configs\sam2.1\sam2.1_hiera_b+.yaml"
sam2_checkpoint=r"C:\Users\Bicel service\Desktop\USER\Maria\sam2\checkpoints\sam2.1_hiera_base_plus.pt"

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)


mask_generator = SAM2AutomaticMaskGenerator(sam2)
# Convertir img2d (grayscale) en RGB
img_rgb = np.stack([img2d]*3, axis=-1).astype(np.uint8)

# Générer les masques
masks = mask_generator.generate(img_rgb)

plt.figure()
plt.imshow(img_rgb)
show_anns(masks)
plt.axis('off')
plt.show()

