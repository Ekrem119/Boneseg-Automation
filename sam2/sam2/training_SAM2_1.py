import os
import cv2
import numpy as np
import torch

from hydra import initialize
from hydra.core.global_hydra import GlobalHydra

# Réinitialiser Hydra si déjà utilisée
GlobalHydra.instance().clear()

# Initialiser Hydra avec le dossier configs relatif au package SAM2
initialize(config_path="sam2/configs", version_base=None)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# -------------------------
# Chemins du dataset (à adapter)
# -------------------------
data_root = r"C:/Users/Bicel service/Desktop/USER/Maria/dataset"
#data_root = r"C:/Users/maria/stage_M2LST/dataset"
images_dir = os.path.join(data_root, "images")
masks_dir = os.path.join(data_root, "masks")

# -------------------------
# Collecte des paires image/masque
# -------------------------
data = []
for name in os.listdir(images_dir):
    if name.lower().endswith(".png"):
        img_path = os.path.join(images_dir, name)
        mask_path = os.path.join(masks_dir, "mask-" + name)
        if os.path.exists(mask_path):
            data.append({"image": img_path, "mask": mask_path})

if len(data) == 0:
    raise RuntimeError("Aucune paire image/masque trouvée. Vérifie les noms dans images/ et masks/.")

# -------------------------
# Lecture d'un échantillon (multiclasses -> one-vs-all)
# -------------------------
def read_single(data, target_classes=(1, 2, 3)):
    ent = data[np.random.randint(len(data))]

    # Image RGB
    Img = cv2.imread(ent["image"])
    if Img is None:
        return read_single(data, target_classes)
    Img = Img[..., ::-1]  # BGR -> RGB

    # Masque multiclasses en niveaux de gris (valeurs 0,1,2,3)
    ann_map = cv2.imread(ent["mask"], cv2.IMREAD_GRAYSCALE)
    if ann_map is None:
        return read_single(data, target_classes)

    # Redimensionner pour que le max côté <= 1024
    r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])
    new_w, new_h = int(Img.shape[1] * r), int(Img.shape[0] * r)
    Img = cv2.resize(Img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    ann_map = cv2.resize(ann_map, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Padding à 1024x1024
    pad_img = np.zeros((2048, 2048, 3), dtype=np.uint8)
    pad_mask = np.zeros((2048, 2048), dtype=np.uint8)
    pad_img[:new_h, :new_w] = Img
    pad_mask[:new_h, :new_w] = ann_map

    # Classes présentes (exclure fond=0)
    present = [c for c in target_classes if (pad_mask == c).any()]
    if len(present) == 0:
        return read_single(data, target_classes)

    # Choisir une classe au hasard parmi celles présentes
    chosen_class = int(np.random.choice(present))

    # Masque binaire pour la classe choisie (one-vs-all)
    mask = (pad_mask == chosen_class).astype(np.uint8)

    # Choisir un point prompt dans l'objet
    coords = np.argwhere(mask > 0)
    yx = coords[np.random.randint(len(coords))]
    input_point = [[int(yx[1]), int(yx[0])]]  # [x, y]

    return pad_img, mask, input_point, 1  # label 1 = foreground

# -------------------------
# Lecture d'un batch
# -------------------------
def read_batch(data, batch_size=4):
    limage, lmask, linput_point, linput_label = [], [], [], []
    for _ in range(batch_size):
        image, mask, input_point, input_label = read_single(data)
        limage.append(image)
        lmask.append(mask)
        linput_point.append(input_point)
        linput_label.append([input_label])  # shape (1,)
    return limage, np.array(lmask), np.array(linput_point), np.array(linput_label)

# -------------------------
# Chargement du modèle SAM2
# -------------------------
# Adapte ces chemins à ton environnement
model_cfg = r"C:\Users\Bicel service\anaconda3\envs\sam_env\Lib\site-packages\sam2\configs\sam2\sam2_hiera_b+.yaml"
sam2_checkpoint = r"C:\Users\Bicel service\Desktop\USER\Maria\sam2\checkpoints\sam2_hiera_base_plus.pt"

#model_cfg = r"C:\Users\maria\sam2_env\Lib\site-packages\sam2\configs\sam2\sam2_hiera_b+"
#model_cfg = r"C:\Users\maria\stage_M2LST\codes\sam2\configs\sam2_hiera_b+"
#sam2_checkpoint = r"C:\Users\maria\sam2_env\Lib\site-packages\sam2\checkpoints\sam2_hiera_base_plus.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)

# -------------------------
# Paramètres d'entraînement
# -------------------------
# Activer l'entraînement des modules pertinents
predictor.model.sam_mask_decoder.train(True)
predictor.model.sam_prompt_encoder.train(True)
predictor.model.image_encoder.train(True)  # nécessite que toute section "no_grad" soit retirée côté lib si présent

optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=1e-5, weight_decay=4e-5)
scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

# -------------------------
# Boucle d'entraînement (one-vs-all par itération)
# -------------------------
mean_iou = 0.0

for itr in range(5000):
    with torch.cuda.amp.autocast(enabled=(device == "cuda")):
        images, masks, input_points, input_labels = read_batch(data, batch_size=4)
        if masks.shape[0] == 0:
            continue

        # Encode image batch
        predictor.set_image_batch(images)

        # Prompt encoding (points)
        mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
            input_points, input_labels, box=None, mask_logits=None, normalize_coords=True
        )
        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
            points=(unnorm_coords, labels), boxes=None, masks=None
        )

        # Mask decoder
        high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
            image_embeddings=predictor._features["image_embed"],
            image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        # Upscale to original resolution
        prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])  # [B, K, H, W]
        # On prend le premier logit (K=0) pour la supervision binaire
        prd_mask = torch.sigmoid(prd_masks[:, 0])  # [B, H, W]

        # GT binaire (one-vs-all pour la classe choisie)
        gt_mask = torch.tensor(masks.astype(np.float32), device=device)  # [B, H, W]

        # Binary cross-entropy
        seg_loss = (-gt_mask * torch.log(prd_mask + 1e-5) - (1 - gt_mask) * torch.log((1 - prd_mask) + 1e-5)).mean()

        # Score loss via IoU
        inter = (gt_mask * (prd_mask > 0.5)).sum(dim=(1, 2))
        union = gt_mask.sum(dim=(1, 2)) + (prd_mask > 0.5).sum(dim=(1, 2)) - inter
        iou = torch.where(union > 0, inter / union, torch.zeros_like(union))
        score_loss = torch.abs(prd_scores[:, 0] - iou).mean()

        loss = seg_loss + 0.05 * score_loss

    # Backprop + update
    predictor.model.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    # Logging
    mean_iou = float(0.99 * mean_iou + 0.01 * iou.mean().detach().cpu().numpy())
    if itr % 100 == 0:
        print(f"step {itr} | loss {float(loss.detach().cpu().numpy()):.4f} | mean IoU {mean_iou:.4f}")

    # Sauvegarde périodique
    if itr % 1000 == 0 and itr > 0:
        torch.save(predictor.model.state_dict(), os.path.join(data_root, f"sam2_finetune_step_{itr}.torch"))
