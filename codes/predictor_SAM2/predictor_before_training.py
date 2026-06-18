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

# --- Charger ND2 ---
if imagerie == "fluo-spinning":
    nd2_path = f"{dir_path}/{anat_vs_koek}-{num_lame}/{anat_vs_koek}-{num_lame}-{imagerie}-{obj}.nd2"
else:
    nd2_path = f"{dir_path}/{anat_vs_koek}-{num_lame}/{anat_vs_koek}-{num_lame}-{imagerie}.nd2"
print("Chemin ND2 construit:", nd2_path)

if imagerie == "fluo-spinning":
    with ND2Reader(nd2_path) as images:
        Z = images.sizes.get('z', 1)
        channels = images.metadata.get('channels', [])
        print("Canaux disponibles:", channels)

        # dictionnaire dynamique basé sur l'ordre réel
        channel_map = {name: idx for idx, name in enumerate(channels)}

        # normaliser la saisie utilisateur
        channel_choice = channel_choice.strip()
        if channel_choice not in channel_map:
            print(f"Canal {channel_choice} non trouvé, utilisation de DAPI par défaut")
            c_idx = channel_map.get("DAPI", 0)
        else:
            c_idx = channel_map[channel_choice]

        # Extraire toutes les frames du canal choisi
        frames = [images.get_frame_2D(c=c_idx, z=z) for z in range(Z)]
        channel_stack = np.stack(frames, axis=0)
        img2d = channel_stack.max(axis=0)
else:
    with ND2Reader(nd2_path) as images:
        img2d = images.get_frame(0)


# Optionnel: limiter la taille disque (SAM2 gère ses propres transforms)
if img2d.shape[0] > 4096 or img2d.shape[1] > 4096:
    img2d = resize(img2d, (4096, 4096), preserve_range=True, anti_aliasing=True)


# Construire le dossier principal en fonction de l'objet et du canal
images_dir = os.path.join(parent, "dataset", f"images_{obj}_{channel_choice}")

# Construire le sous-dossier avec le nom de l'image (sans extension)

image_folder = f"pretraitee-{anat_vs_koek}-{num_lame}-{imagerie}-{obj}"
image_subdir = os.path.join(images_dir, image_folder)

# Créer les dossiers
os.makedirs(image_subdir, exist_ok=True)

# Nom du fichier PNG
filename = f"pretraitee-{anat_vs_koek}-{num_lame}-{imagerie}-{obj}.png"
png_path = os.path.join(image_subdir, filename)

# Sauvegarde de l'image
plt.figure()
plt.imshow(img2d, cmap="gray", aspect="equal")
plt.axis("off")
plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0)
plt.close()

print("Image prétraitée enregistrée sous :", png_path)


r'''
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor
# --- Config et checkpoint SAM2 officiel ---
model_cfg = r"C:\Users\Bicel service\Desktop\USER\Maria\sam2\configs\sam2.1\sam2.1_hiera_b+.yaml"
sam2_checkpoint = r"C:\Users\Bicel service\Desktop\USER\Maria\sam2\checkpoints\sam2.1_hiera_base_plus.pt"

# Charger directement SAM2 pré-entraîné
model = build_sam2(model_cfg, sam2_checkpoint, device=device)
model.eval()

predictor = SAM2ImagePredictor(model)


# Charger l'image RGB et sanity check
image = np.array(Image.open(png_path).convert("RGB"))
print("Image dtype/shape/min/max:", image.dtype, image.shape, image.min(), image.max())




import numpy as np
from skimage.measure import regionprops, label, perimeter as sk_perimeter
import pandas as pd

import numpy as np
import pandas as pd
from skimage.measure import regionprops, label, perimeter

def compute_mask_metrics(masks, min_area=0):
    """
    Calcule les métriques pour chaque masque détecté par le predictor.
    masks: tableau numpy (N, H, W) avec des masques binaires
    min_area: filtre les petits masques
    """
    rows = []
    for i in range(masks.shape[0]):
        mask = masks[i].astype(np.uint8)
        if mask.sum() < min_area:
            continue

        # Aire
        area = mask.sum()

        # Périmètre
        perim = perimeter(mask)

        # Circularité
        circ = (4.0 * np.pi * area / (perim ** 2)) if perim > 0 else 0.0

        # Centroid
        coords = np.column_stack(np.where(mask > 0))
        centroid_row, centroid_col = coords.mean(axis=0)

        # BBox
        minr, minc = coords.min(axis=0)
        maxr, maxc = coords.max(axis=0)
        bbox_h = maxr - minr
        bbox_w = maxc - minc
        aspect = (bbox_w / bbox_h) if bbox_h > 0 else np.nan

        rows.append({
            "mask_index": i,
            "area_px": int(area),
            "perimeter_px": float(perim),
            "circularity": float(circ),
            "centroid_row": float(centroid_row),
            "centroid_col": float(centroid_col),
            "bbox_height": int(bbox_h),
            "bbox_width": int(bbox_w),
            "aspect_ratio": float(aspect),
        })

    df = pd.DataFrame(rows)
    summary = {
        "num_masks": len(df),
        "mean_area_px": float(df["area_px"].mean()) if len(df) else 0.0,
        "mean_circularity": float(df["circularity"].mean()) if len(df) else 0.0,
    }
    return df, summary


def onclick(event):
    if event.xdata is None or event.ydata is None:
        return
    x, y = int(event.xdata), int(event.ydata)
    plt.close()
    run_predict(x, y)

def run_predict(x, y):
    H, W = image.shape[:2]
    point_coords = np.array([[[x, y]]], dtype=np.int32)
    point_labels = np.array([[1]], dtype=np.int32)

    with torch.inference_mode():
        predictor.set_image(image)
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )

    print("Scores:", scores)
    best_idx = int(np.argmax(scores))
    best_mask = masks[best_idx]

    # --- calcul des métriques uniquement sur le meilleur masque ---
    df_stats, summary = compute_mask_metrics(np.expand_dims(best_mask, axis=0), min_area=0)
    print("Résumé:", summary)
    print(df_stats.head())

    # --- affichage ---
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.imshow(best_mask, alpha=0.5)
    plt.scatter([x],[y], c='red', s=40)
    plt.axis("off")
    plt.show()


plt.figure(figsize=(10,10))
plt.imshow(image)
plt.title("Clique pour poser un point (prompt foreground)")
plt.axis("off")
cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
plt.show()
'''