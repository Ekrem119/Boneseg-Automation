import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hydra import initialize
from hydra.core.global_hydra import GlobalHydra

GlobalHydra.instance().clear()
initialize(config_path="configs/sam2", version_base=None)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# -------------------------
# Dataset
# -------------------------
data_root = r"C:/Users/Bicel service/Desktop/USER/Maria/dataset"
images_dir = os.path.join(data_root, "images")
masks_dir = os.path.join(data_root, "masks")

data = []
for name in os.listdir(images_dir):
    if name.lower().endswith(".png"):
        img_path = os.path.join(images_dir, name)
        mask_path = os.path.join(masks_dir, "mask-" + name)
        if os.path.exists(mask_path):
            data.append({"image": img_path, "mask": mask_path})
if len(data) == 0:
    raise RuntimeError("Aucune paire image/masque trouvée.")

# -------------------------
# Lecture d'un échantillon multi‑classe
# -------------------------
def read_single(data):
    ent = data[np.random.randint(len(data))]
    Img = cv2.imread(ent["image"])[..., ::-1]
    ann_map = cv2.imread(ent["mask"], cv2.IMREAD_GRAYSCALE)

    r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])
    new_w, new_h = int(Img.shape[1] * r), int(Img.shape[0] * r)
    Img = cv2.resize(Img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    ann_map = cv2.resize(ann_map, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    pad_img = np.zeros((2048, 2048, 3), dtype=np.uint8)
    pad_mask = np.zeros((2048, 2048), dtype=np.uint8)
    pad_img[:new_h, :new_w] = Img
    pad_mask[:new_h, :new_w] = ann_map

    return pad_img, pad_mask

def read_batch(data, batch_size=4):
    limage, lmask = [], []
    for _ in range(batch_size):
        image, mask = read_single(data)
        limage.append(image)
        lmask.append(mask)
    return limage, np.array(lmask)

# -------------------------
# SAM2 + tête multi‑classe
# -------------------------
model_cfg = r"C:\Users\Bicel service\anaconda3\envs\sam_env\Lib\site-packages\sam2\configs\sam2\sam2_hiera_b+.yaml"
sam2_checkpoint = r"C:\Users\Bicel service\Desktop\USER\Maria\sam2\checkpoints\sam2_hiera_base_plus.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)

num_classes = 4  # fond=0 + 3 classes

class MultiClassSegHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels//2, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels//2)
        self.conv2 = nn.Conv2d(in_channels//2, num_classes, 1)
    def forward(self, x, out_size):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)

dummy_img = np.zeros((512,512,3), dtype=np.uint8)
predictor.set_image(dummy_img)
in_channels = predictor._features["high_res_feats"][-1].shape[1]
mc_head = MultiClassSegHead(in_channels, num_classes).to(device)

optimizer = torch.optim.AdamW(list(predictor.model.image_encoder.parameters())+list(mc_head.parameters()), lr=2e-4)
criterion = nn.CrossEntropyLoss()

# -------------------------
# Boucle d'entraînement avec sauvegarde
# -------------------------
mean_iou = 0.0
max_iters = 2000

for itr in range(max_iters):
    images, masks = read_batch(data, batch_size=4)
    predictor.set_image_batch(images)
    high_res = predictor._features["high_res_feats"][-1]
    out_size = predictor._orig_hw[-1]

    logits = mc_head(high_res, out_size)  # [B,C,H,W]
    gt_mask = torch.tensor(masks, device=device, dtype=torch.long)

    seg_loss = criterion(logits, gt_mask)

    preds = logits.argmax(1)
    ious = []
    for c in range(1,num_classes):
        inter = ((gt_mask==c)&(preds==c)).sum()
        union = ((gt_mask==c)|(preds==c)).sum()
        if union>0:
            ious.append(inter.float()/union.float())
    mean_iou_batch = torch.mean(torch.stack(ious)) if ious else torch.tensor(0.0, device=device)

    optimizer.zero_grad()
    seg_loss.backward()
    optimizer.step()

    mean_iou = 0.99*mean_iou + 0.01*mean_iou_batch.item()
    if itr%100==0:
        print(f"step {itr} | loss {seg_loss.item():.4f} | mean IoU {mean_iou:.4f}")

    # Sauvegarde périodique
    if itr % 500 == 0 and itr > 0:
        torch.save({
            "image_encoder": predictor.model.image_encoder.state_dict(),
            "mc_head": mc_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": itr
        }, os.path.join(data_root, f"sam2_multiclass_step_{itr}.pth"))

# Sauvegarde finale
torch.save({
    "image_encoder": predictor.model.image_encoder.state_dict(),
    "mc_head": mc_head.state_dict(),
    "optimizer": optimizer.state_dict(),
    "iteration": max_iters
}, os.path.join(data_root, "sam2_multiclass_final.pth"))
