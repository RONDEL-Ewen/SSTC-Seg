"""
Compare visuellement la frame RGB, la Ground Truth et les prédictions de :
U-Net, PSPNet, CFPNet-M, MFCPNet et SSTC-Seg
sur le test split de CholecSeg8k.
"""

import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# --- Imports de ton repo ---
from datasets.CholecSeg8kDataset import CholecSeg8kDataset
from model.SSTCSeg import SSTCSeg
from baselines.UNet2 import UNet
from baselines.CFPNetM import CFPNetM
from baselines.MFCPNet import MFCPNet
from baselines.PSPNet import PSPNet

warnings.simplefilter("ignore", category=FutureWarning)
torch.manual_seed(1)
cudnn.benchmark = True


# ---------------- CONFIGURATION EN DUR ----------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 13
MEMORY_LENGTH = 4
BATCH_SIZE = 2

# Chemin du dataset (à modifier selon ton arborescence)
CHOLEC_DIR = "./datasets/tensors/cholecseg8k_tensors"

# Chemins vers les poids
WEIGHTS = {
    "U-Net":     "./model/weights/cholec_unet.pth",
    "PSPNet":    "./model/weights/cholec_pspnet.pth",
    "CFPNet-M":  "./model/weights/cholec_cfpnetm.pth",
    "MFCPNet":   "./model/weights/cholec_mfcpnet.pth",
    "SSTC-Seg":  "./model/weights/cholec_fulldef_multihead.pth",
}

# Hyperparamètres spécifiques à SSTC-Seg
SSTC_CFG = dict(
    num_encoder_blocks=3,
    num_heads=8,
    k=5,
    memory_frames=MEMORY_LENGTH,
    num_memory_blocks=3,
    deformable=True,
    use_crf=True,
)

SAVE_DIR = "./comparisons"   # Dossier de sortie
SHOW = False             # Mettre False sur un serveur sans affichage
MAX_FRAMES = None         # Limiter le nombre d’échantillons (None = tout)


# ------------------- UTILS -------------------

def build_model(name: str) -> nn.Module:
    if name == "SSTC-Seg":
        return SSTCSeg(num_classes=NUM_CLASSES, **SSTC_CFG)
    elif name == "U-Net":
        return UNet(in_channels=3, num_classes=NUM_CLASSES)
    elif name == "PSPNet":
        return PSPNet(in_channels=3, num_classes=NUM_CLASSES, pool_sizes=(1, 2, 3, 6),
                      ppm_reduction_dim=512, backbone_pretrained=True, dropout=0.1)
    elif name == "CFPNet-M":
        return CFPNetM(in_channels=3, num_classes=NUM_CLASSES)
    elif name == "MFCPNet":
        return MFCPNet(input_dim=3, num_classes=NUM_CLASSES)
    else:
        raise ValueError(f"Modèle inconnu : {name}")


def load_weights(model, path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location=DEVICE)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)


def one_hot_to_indices(t: torch.Tensor) -> torch.Tensor:
    return t.argmax(dim=1) if t.dim() == 4 else t


# ---------------- VISUALISATION ----------------

CHOLEC_LABELS = [
    "Background", "Abdominal Wall", "Liver", "Gastrointestinal Tract", "Fat",
    "Grasper", "Connective Tissue", "Blood", "Cystic Duct",
    "L-hook Electrocautery", "Gallbladder", "Hepatic Vein", "Liver Ligament",
]


def get_custom_cmap(num_classes):
    """Crée une colormap où la classe 0 (Background) est noire."""
    base_cmap = plt.cm.get_cmap("jet", num_classes)
    colors = base_cmap(np.arange(num_classes))
    colors[0] = [0, 0, 0, 1.0]  # Background = noir opaque
    return ListedColormap(colors)


def viz_side_by_side(rgb, gt, preds_ordered, save_path=None, show=True, title=None):
    """
    Affiche 7 panneaux : [RGB, GT, U-Net, PSPNet, CFPNet-M, MFCPNet, SSTC-Seg]
    """
    names = ["RGB", "Ground Truth", "U-Net", "PSPNet", "CFPNet-M", "MFCPNet", "SSTC-Seg"]
    num_classes = len(CHOLEC_LABELS)
    fig, axes = plt.subplots(1, 7, figsize=(21, 3.6), gridspec_kw={'wspace': 0.05})

    cmap = get_custom_cmap(num_classes)
    norm = plt.Normalize(vmin=0, vmax=num_classes - 1)

    # Le RGB est le premier panneau, puis GT + 5 prédictions
    panels = [rgb, gt] + preds_ordered
    for ax, panel, name in zip(axes, panels, names):
        if name == "RGB":
            im = ax.imshow(panel)
        else:
            im = ax.imshow(panel, cmap=cmap, norm=norm)
        ax.set_title(name, fontsize=10)
        ax.axis("off")

    # Barre de couleurs commune (sauf pour RGB)
    cbar = fig.colorbar(im, ax=axes[1:].ravel().tolist(),
                        ticks=range(num_classes), fraction=0.025, pad=0.01)
    cbar.ax.set_yticklabels(CHOLEC_LABELS, fontsize=8)
    cbar.ax.set_title("Classes", fontsize=10)

    if title:
        fig.suptitle(title, y=1.02, fontsize=12)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------- MAIN LOOP ----------------

def main():
    dataset = CholecSeg8kDataset(
        data_dir=CHOLEC_DIR,
        history_length=MEMORY_LENGTH,
        augment=False,
    )

    train_vids = {1, 9, 12, 17, 18, 20, 24, 25, 26, 27, 28}
    val_vids = {35, 37, 43}
    test_vids = {48, 52, 55}

    test_indices = []
    for idx, seq in enumerate(dataset.sequences):
        files = seq[0] if isinstance(seq, tuple) else seq
        video_id = int(os.path.basename(files[0]).split("_")[0])
        if video_id in test_vids:
            test_indices.append(idx)

    loader = DataLoader(Subset(dataset, test_indices),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model_names = ["U-Net", "PSPNet", "CFPNet-M", "MFCPNet", "SSTC-Seg"]
    models = {}
    for n in model_names:
        print(f"Chargement {n} ...")
        m = build_model(n).to(DEVICE)
        load_weights(m, WEIGHTS[n])
        m.eval()
        models[n] = m

    shown = 0
    with torch.no_grad():
        for b_idx, (imgs, masks) in enumerate(loader):
            imgs = [i.squeeze(1).to(DEVICE) for i in imgs]
            masks = [m.squeeze(1).to(DEVICE) for m in masks]

            current = imgs[-1]
            current_mask = masks[-1]
            mem_imgs, mem_masks = imgs[:-1], masks[:-1]

            preds = {n: m(current, training=True,
                          memory_frames=mem_imgs,
                          memory_masks=mem_masks)
                     for n, m in models.items()}

            gt = one_hot_to_indices(current_mask)
            preds_idx = {n: p.argmax(1) for n, p in preds.items()}

            for i in range(gt.shape[0]):
                # Frame RGB normalisée pour affichage
                rgb_np = current[i].permute(1, 2, 0).cpu().numpy()
                rgb_np = np.clip(rgb_np, 0, 1)

                gt_np = gt[i].cpu().numpy()
                preds_np = [preds_idx[n][i].cpu().numpy() for n in model_names]

                save_path = os.path.join(SAVE_DIR, f"frame_{b_idx:04d}_{i:02d}.png")
                viz_side_by_side(rgb_np, gt_np, preds_np, save_path=save_path,
                                 show=SHOW, title=f"Batch {b_idx} / Sample {i}")
                shown += 1
                if MAX_FRAMES and shown >= MAX_FRAMES:
                    return


if __name__ == "__main__":
    main()
