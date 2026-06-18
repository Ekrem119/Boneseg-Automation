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


# --- Cellpose imports ---
from cellpose import models, io, plot
from cellpose.utils import masks_to_outlines

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
else:
    obj = "bf"
    channel_choice = "Brightfield"

parent = os.path.dirname(os.path.dirname(dir_path))

# --- Construire chemin ND2 ---
if imagerie == "fluo-spinning":
    nd2_path = f"{dir_path}/{anat_vs_koek}-{num_lame}/{anat_vs_koek}-{num_lame}-{imagerie}-{obj}.nd2"
else:
    nd2_path = f"{dir_path}/{anat_vs_koek}-{num_lame}/{anat_vs_koek}-{num_lame}-{imagerie}.nd2"
print("Chemin ND2 construit:", nd2_path)

# --- Charger ND2 et extraire canal ---
if imagerie == "fluo-spinning":
    with ND2Reader(nd2_path) as images:
        Z = images.sizes.get('z', 1)
        channels = images.metadata.get('channels', [])
        print("Canaux disponibles:", channels)

        channel_map = {name: idx for idx, name in enumerate(channels)}
        channel_choice = channel_choice.strip()
        if channel_choice not in channel_map:
            print(f"Canal {channel_choice} non trouvé, utilisation de DAPI par défaut")
            c_idx = channel_map.get("DAPI", 0)
        else:
            c_idx = channel_map[channel_choice]

        frames = [images.get_frame_2D(c=c_idx, z=z) for z in range(Z)]
        channel_stack = np.stack(frames, axis=0)
        img2d = channel_stack.max(axis=0)  # projection max
else:
    with ND2Reader(nd2_path) as images:
        img2d = images.get_frame(0)

# Limiter taille si très grande
if img2d.shape[0] > 4096 or img2d.shape[1] > 4096:
    img2d = resize(img2d, (4096, 4096), preserve_range=True, anti_aliasing=True)

# Construire dossier de sortie (image + masque)
images_dir = os.path.join(parent, "dataset", f"images_{obj}_{channel_choice}")
masks_dir = os.path.join(parent, "dataset", f"masks_{obj}_{channel_choice}")
image_folder = f"pretraitee-{anat_vs_koek}-{num_lame}-{imagerie}-{obj}"
image_subdir = os.path.join(images_dir, image_folder)
mask_subdir = os.path.join(masks_dir, image_folder)
os.makedirs(image_subdir, exist_ok=True)
os.makedirs(mask_subdir, exist_ok=True)

filename = f"pretraitee-{anat_vs_koek}-{num_lame}-{imagerie}-{obj}.png"
png_path = os.path.join(image_subdir, filename)
mask_path = os.path.join(mask_subdir, filename)

# Sauvegarde de l’image (grayscale)
plt.figure(figsize=(6,6))
plt.imshow(img2d, cmap="gray", aspect="equal")
plt.axis("off")
plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0)
plt.close()
print("Image prétraitée enregistrée sous :", png_path)

# --- Segmentation Cellpose ---
# Mapping des channels pour Cellpose:
# channels = [channel for cytoplasm, channel for nuclei]
# Pour fluorescence nucleaire DAPI: channels=[0,3] (nuclei=DAPI)
# Pour brightfield: channels=[0,0]
def cellpose_channels_for(channel_choice):
    ch = channel_choice.lower()
    if ch in ["dapi"]:
        return [0, 3]  # nuclei = DAPI
    elif ch in ["brightfield", "bf"]:
        return [0, 0]
    else:
        # canal unique fluorescence (sans nuclei explicite)
        return [0, 0]

def run_cellpose(img_float):
    # img_float doit être float32
    if img_float.dtype != np.float32:
        img_float = img_float.astype(np.float32)
    vmax = np.percentile(img_float, 99.5)
    img_float = np.clip(img_float / (vmax + 1e-8), 0.0, 1.0)

    channels = cellpose_channels_for(channel_choice)
    print(f"Cellpose-SAM, channels={channels}")

    use_gpu = (device.type == "cuda")
    # Charger le modèle SAM
    model = models.CellposeModel(gpu=use_gpu, pretrained_model='sam')

    results = model.eval(img_float, diameter=None, channels=channels,
                         flow_threshold=0.4, cellprob_threshold=0.0)

    # Cellpose v4 renvoie 3 valeurs
    masks, flows, styles = results
    return masks



# Préparation image pour Cellpose
img_float = img2d.astype(np.float32)
masks = run_cellpose(img_float)  # Cellpose-SAM


# Masque binaire strict (0 fond, 1 objet)
mask_bin = (masks > 0).astype(np.uint8)

# Sauvegarde du masque
Image.fromarray(mask_bin * 255).save(mask_path)
print("Masque Cellpose enregistré sous :", mask_path)

# --- Visualisation: image + contours + masque ---
outlines = masks_to_outlines(masks)
plt.figure(figsize=(8,8))
plt.imshow(img2d, cmap='gray')
plt.imshow(np.ma.masked_where(masks==0, masks), alpha=0.35, cmap='Reds')
plt.contour(outlines, colors='yellow', linewidths=0.8)
plt.title(f"Cellpose ({channel_choice}) - {anat_vs_koek}-{num_lame}-{imagerie}-{obj}")
plt.axis('off')
plt.show()
