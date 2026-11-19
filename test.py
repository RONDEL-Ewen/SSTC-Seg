from scipy.ndimage import distance_transform_edt, binary_erosion
import matplotlib.pyplot as plt
import numpy as np
import warnings
import argparse
import yaml
import os

# PyTorch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, DataLoader, Subset

# Datasets
from datasets.DresdenDataset import DresdenDataset
from datasets.HemoSetDataset import HemoSetDataset
from datasets.CholecSeg8kDataset import CholecSeg8kDataset

# Models
from model.SSTCSeg import SSTCSeg
from baselines.UNet2 import UNet
from baselines.CFPNetM import CFPNetM
from baselines.MFCPNet import MFCPNet
from baselines.PSPNet import PSPNet

torch.manual_seed(1)
cudnn.benchmark = True
warnings.simplefilter("ignore", category=FutureWarning)

eps = 1e-6



# ===============================[ Arguments ]===============================

def parse_args():

    parser = argparse.ArgumentParser(
        description = "Training PyTorch script for SSTC-Seg"
    )

    # Dataset arguments

    parser.add_argument("--dresden_dir",        type = str,  default = None,        help = "Tensors directory for the Dresden dataset.")
    parser.add_argument("--hemoset_dir",        type = str,  default = None,        help = "Tensors directory for the HemoSet dataset.")
    parser.add_argument("--cholec_dir",         type = str,  default = None,        help = "Tensors directory for the CholecSeg8k dataset.")

    parser.add_argument("--num_classes",        type = int,  default = 8,           help = "Number of classes to segment.")


    # Model arguments

    parser.add_argument("--selected_model",     type = str,  default = "SSTC-Seg",  help = "Model to train ([SSTC-Seg, U-Net, CFPNet-M, MFCPNet]).")
    parser.add_argument("--model_dir",          type = str,  default = None,        help = "Weights directory for the model.")

    parser.add_argument("--num_encoder_blocks", type = int,  default = 3,           help = "Number of encoder blocks in FrameEncoder (last one does not perform downsampling).")
    parser.add_argument("--num_heads",          type = int,  default = 8,           help = "Number of heads in attention mechanisms.")
    parser.add_argument("--k",                  type = int,  default = 5,           help = "")
    parser.add_argument("--memory_length",      type = int,  default = 4,           help = "Number of past frames to use as memory in the attention mechanisms.")
    parser.add_argument("--num_memory_blocks",  type = int,  default = 3,           help = "Number of attention blocks in MemoryAttention.")
    parser.add_argument("--deformable",         type = bool, default = True,        help = "Replace standard convolutions with deformable convolutions within FrameEncoder.")
    parser.add_argument("--use_crf",            type = bool, default = True,        help = "Use a Hierarchical Dense CRF.")

    # Testing arguments

    parser.add_argument("--batch_size",         type = int,  default = 2,           help = "Batch size.")
    parser.add_argument("--device",             type = str,  default = "cuda",      help = "Device to run the training on.")

def load_config(
    config_path = "config.yaml"
):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config



# ===============================[ Visualization ]===============================

def visualize_predictions(
    images,
    masks,
    predictions,
    num_classes
):
    class_labels = [
        "Abdominal Wall",
        "Colon",
        "Liver",
        "Pancreas",
        "Small Intestine",
        "Spleen",
        "Stomach",
        "Background",
        "Blood"
    ]
    current_frame = images[-1].detach().cpu().squeeze(0).permute(1, 2, 0)
    current_mask = masks[-1].detach().cpu().squeeze(0)
    predicted_mask = predictions.detach().cpu().squeeze(0)

    if len(current_mask.shape) == 3:
        current_mask = current_mask.argmax(dim=0)
    if len(predicted_mask.shape) == 3:
        predicted_mask = predicted_mask.argmax(dim=0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={'width_ratios': [1, 1, 1]})
    axes[0].imshow(current_frame)
    axes[0].set_title("Frame", fontsize=10)
    axes[0].axis("off")

    cmap = plt.get_cmap("jet", num_classes)
    norm = plt.Normalize(vmin=0, vmax=num_classes - 1)

    img1 = axes[1].imshow(current_mask, cmap=cmap, norm=norm)
    axes[1].set_title("Ground Truth", fontsize=10)
    axes[1].axis("off")

    img2 = axes[2].imshow(predicted_mask, cmap=cmap, norm=norm)
    axes[2].set_title("Predicted Mask", fontsize=10)
    axes[2].axis("off")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(img2, cax=cbar_ax, ticks=range(num_classes))
    cbar.ax.set_yticklabels(class_labels, fontsize=8)
    cbar.ax.set_title("Classes", fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()



# ===============================[ Global Metrics ]===============================

def accuracy_score(
    pred,
    gt
):
    return np.sum(pred == gt) / pred.size

def dice_score_per_sample(
    pred,
    gt,
    num_classes
):
    eps = 1e-6
    scores = []
    for c in range(num_classes):
        pred_c = (pred == c).astype(np.float32)
        gt_c = (gt == c).astype(np.float32)
        intersection = np.sum(pred_c * gt_c)
        denominator = np.sum(pred_c) + np.sum(gt_c)
        if denominator == 0:
            dice = 1.0
        else:
            dice = (2. * intersection + eps) / (denominator + eps)
        scores.append(dice)
    return np.mean(scores)

def jaccard_score_per_sample(
    pred,
    gt,
    num_classes
):
    eps = 1e-6
    scores = []
    for c in range(num_classes):
        pred_c = (pred == c).astype(np.float32)
        gt_c = (gt == c).astype(np.float32)
        intersection = np.sum(pred_c * gt_c)
        union = np.sum(pred_c) + np.sum(gt_c) - intersection
        if union == 0:
            score = 1.0
        else:
            score = (intersection + eps) / (union + eps)
        scores.append(score)
    return np.mean(scores)

def compute_hd95(
    binary1,
    binary2
):
    struct = np.ones((3, 3), dtype=bool)
    edge1 = binary1 ^ binary_erosion(binary1, structure=struct)
    edge2 = binary2 ^ binary_erosion(binary2, structure=struct)
    if not edge1.any() or not edge2.any():
        return np.nan
    dt_edge2 = distance_transform_edt(~edge2)
    distances1 = dt_edge2[edge1]
    dt_edge1 = distance_transform_edt(~edge1)
    distances2 = dt_edge1[edge2]
    all_distances = np.concatenate([distances1, distances2])
    hd95 = np.percentile(all_distances, 95)
    return hd95

def hausdorff95_score_per_sample(
    pred,
    gt,
    num_classes
):
    scores = []
    for c in range(num_classes):
        pred_c = (pred == c).astype(np.bool_)
        gt_c = (gt == c).astype(np.bool_)
        if not pred_c.any() and not gt_c.any():
            hd95 = 0.0
        elif not pred_c.any() or not gt_c.any():
            hd95 = np.inf
        else:
            hd95 = compute_hd95(pred_c, gt_c)
        scores.append(hd95)
    finite_scores = [s for s in scores if np.isfinite(s)]
    if len(finite_scores) == 0:
        return np.inf
    return np.mean(finite_scores)



# ===============================[ Per-class Metrics ]===============================

def accuracy_score_per_class(pred, gt, num_classes):
    scores = []
    for c in range(num_classes):
        pred_mask = (pred == c)
        gt_mask   = (gt   == c)
        if not pred_mask.any() and not gt_mask.any():
            # ni GT ni préd → nan
            scores.append(np.nan)
        elif not gt_mask.any():
            # GT vide, préd non vide → faux positifs = 0%
            scores.append(0.0)
        else:
            correct = np.sum(pred_mask & gt_mask)
            scores.append(correct / np.sum(gt_mask))
    return scores

def dice_score_per_class(pred, gt, num_classes):
    eps = 1e-6
    scores = []
    for c in range(num_classes):
        pred_c = (pred == c).astype(np.float32)
        gt_c   = (gt   == c).astype(np.float32)
        if pred_c.sum() == 0 and gt_c.sum() == 0:
            scores.append(np.nan)
        else:
            inter = np.sum(pred_c * gt_c)
            denom = np.sum(pred_c) + np.sum(gt_c)
            scores.append((2.*inter + eps) / (denom + eps))
    return scores

def jaccard_score_per_class(pred, gt, num_classes):
    eps = 1e-6
    scores = []
    for c in range(num_classes):
        pred_c = (pred == c).astype(np.float32)
        gt_c   = (gt   == c).astype(np.float32)
        if pred_c.sum() == 0 and gt_c.sum() == 0:
            scores.append(np.nan)
        else:
            inter = np.sum(pred_c * gt_c)
            union = np.sum(pred_c) + np.sum(gt_c) - inter
            scores.append((inter + eps) / (union + eps))
    return scores

def hausdorff95_score_per_class(pred, gt, num_classes):
    scores = []
    for c in range(num_classes):
        pred_c = (pred == c).astype(bool)
        gt_c   = (gt   == c).astype(bool)
        if not pred_c.any() and not gt_c.any():
            scores.append(np.nan)
        elif not pred_c.any() or not gt_c.any():
            scores.append(np.inf)
        else:
            scores.append(compute_hd95(pred_c, gt_c))
    return scores



# ===============================[ Testing ]===============================

def test_model(
    model,
    test_loader,
    num_classes = 9,
    device      = "cuda"
):
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    model.eval()

    test_loss = 0
    accuracies = []
    dices = []
    jaccards = []
    hd95s = []

    per_class_accuracy = [[] for _ in range(num_classes)]
    per_class_dice = [[] for _ in range(num_classes)]
    per_class_jaccard = [[] for _ in range(num_classes)]
    per_class_hd95 = [[] for _ in range(num_classes)]

    images_counter = 0

    with torch.no_grad():
        for images, masks in test_loader:

            images_counter += 1

            images = [img.squeeze(1).to(device) for img in images]
            masks  = [mask.squeeze(1).to(device) for mask in masks]
            current_frame = images[-1]
            current_mask  = masks[-1]
            memory_frames = images[:-1]
            memory_masks  = masks[:-1]

            prediction = model(current_frame, training=True, memory_frames=memory_frames, memory_masks=memory_masks)
            loss = criterion(prediction, current_mask.argmax(dim=1).long())
            print(f"[{images_counter}/{len(test_loader)}] Testing loss: {loss}")
            test_loss += loss.item()

            pred_masks = prediction.argmax(dim=1).detach().cpu().numpy()
            target_masks = current_mask.argmax(dim=1).detach().cpu().numpy()

            batch_acc = []
            batch_dice = []
            batch_jaccard = []
            batch_hd95 = []

            for i in range(pred_masks.shape[0]):
                acc = accuracy_score(pred_masks[i], target_masks[i])
                dice = dice_score_per_sample(pred_masks[i], target_masks[i], num_classes)
                jaccard = jaccard_score_per_sample(pred_masks[i], target_masks[i], num_classes)
                hd95 = hausdorff95_score_per_sample(pred_masks[i], target_masks[i], num_classes)
                batch_acc.append(acc)
                batch_dice.append(dice)
                batch_jaccard.append(jaccard)
                batch_hd95.append(hd95)

                acc_pc = accuracy_score_per_class(pred_masks[i], target_masks[i], num_classes)
                dice_pc = dice_score_per_class(pred_masks[i], target_masks[i], num_classes)
                jaccard_pc = jaccard_score_per_class(pred_masks[i], target_masks[i], num_classes)
                hd95_pc = hausdorff95_score_per_class(pred_masks[i], target_masks[i], num_classes)

                for c in range(num_classes):
                    per_class_accuracy[c].append(acc_pc[c])
                    per_class_dice[c].append(dice_pc[c])
                    per_class_jaccard[c].append(jaccard_pc[c])
                    per_class_hd95[c].append(hd95_pc[c])

            accuracies.append(sum(batch_acc) / len(batch_acc))
            dices.append(sum(batch_dice) / len(batch_dice))
            jaccards.append(sum(batch_jaccard) / len(batch_jaccard))
            hd95s.append(sum(batch_hd95) / len(batch_hd95))

            #visualize_predictions(current_frame, current_mask, prediction, num_classes)

            del current_frame, current_mask, memory_frames, memory_masks, prediction, loss
            torch.cuda.empty_cache()

    avg_test_loss = test_loss / len(test_loader)
    avg_accuracy = np.mean(accuracies)
    avg_dice = np.mean(dices)
    avg_jaccard = np.mean(jaccards)
    avg_hd95 = np.mean([v for v in hd95s if np.isfinite(v)])

    print("------------------------------")
    print(f"Global Metrics:")
    print(f"   Test Loss:             {avg_test_loss:.4f}")
    print(f"   Accuracy:              {avg_accuracy:.4f}")
    print(f"   Dice Coefficient:      {avg_dice:.4f}")
    print(f"   Jaccard Index:         {avg_jaccard:.4f}")
    print(f"   Hausdorff Distance 95: {avg_hd95:.4f}")
    print("------------------------------")
    print("")

    avg_per_class_accuracy = [np.nanmean(vals) for vals in per_class_accuracy]
    avg_per_class_dice     = [np.nanmean(vals) for vals in per_class_dice]
    avg_per_class_jaccard  = [np.nanmean(vals) for vals in per_class_jaccard]
    # Pour le HD95, on ignore nan et inf
    avg_per_class_hd95    = [
        np.nanmean([v for v in vals if np.isfinite(v)])
        for vals in per_class_hd95
    ]

    print("Per-class Metrics:")

    # Dresden
    """
    class_labels = [
        "Abdominal Wall",
        "Colon",
        #"Inferior Mesenteric Artery",
        #"Intestinal Veins",
        "Liver",
        "Pancreas",
        #"Small Intestine",
        "Spleen",
        "Stomach",
        "Ureter",
        #"Vesicular Glands",
        "Background",
        #"Blood"
    ]
    """

    # HemoSet
    #class_labels = [
    #    "Other",
    #    "Blood"
    #]

    # CholecSeg8k
    class_labels = [
        "Background",
        "Abdominal Wall",
        "Liver",
        "Gastrointestinal Tract",
        "Fat",
        "Grasper",
        "Connective Tissue",
        "Blood",
        "Cystic Duct",
        "L-hook Electrocautery",
        "Gallbladder",
        "Hepatic Vein",
        "Liver Ligament"
    ]

    for c in range(num_classes):
        acc = avg_per_class_accuracy[c]
        dice = avg_per_class_dice[c]
        jac = avg_per_class_jaccard[c]
        hd = avg_per_class_hd95[c]
        print(f"{class_labels[c]}:")
        print(f"   Accuracy:              {acc:.4f}" if not np.isnan(acc) else "   Accuracy:              N/A")
        print(f"   Dice Coefficient:      {dice:.4f}" if not np.isnan(dice) else "   Dice Coefficient:      N/A")
        print(f"   Jaccard Index:         {jac:.4f}" if not np.isnan(jac) else "   Jaccard Index:         N/A")
        print(f"   Hausdorff Distance 95: {hd:.4f}" if not np.isnan(hd) else "   Hausdorff Distance 95: N/A")
        print("")
    print("------------------------------")



# ===============================[ Main ]===============================

if __name__ == '__main__':

    args = parse_args()
    config = load_config()


    # Dataset arguments

    dresden_dir = config.get("dresden_dir")
    hemoset_dir = config.get("hemoset_dir")
    cholec_dir  = config.get("cholec_dir")

    num_classes = config.get("num_classes", 8)


    # Model arguments

    selected_model     = config.get("selected_model", "SSTC-Seg")
    model_dir          = config.get("model_dir")

    num_encoder_blocks = config.get("num_encoder_blocks", 3)
    num_heads          = config.get("num_heads", 8)
    k                  = config.get("k", 5)
    memory_length      = config.get("memory_length", 3)
    num_memory_blocks  = config.get("num_memory_blocks", 3)
    deformable         = config.get("deformable", True)
    use_crf            = config.get("use_crf", True)


    # Testing arguments

    batch_size = config.get("batch_size", 2)
    device     = config.get("device", "cuda")


    # Dresden dataset

    if dresden_dir:

        dataset = DresdenDataset(
            data_dir       = dresden_dir,
            history_length = memory_length,
            augment        = False
        )

        #train_vids = {4, 5, 8, 10, 12, 16, 17, 22, 23, 24, 25, 27, 29, 30, 31}
        #val_vids   = {2, 7, 11, 18, 20}
        #test_vids  = {3, 21, 26}
        train_vids = {2, 5, 7, 8, 12, 16, 17, 21, 23, 24, 25, 27, 31}
        val_vids   = {3, 11, 18, 20, 29}
        test_vids  = {4, 10, 22, 26, 30}

    # HemoSet dataset

    elif hemoset_dir:

        dataset = HemoSetDataset(
            data_dir       = hemoset_dir,
            history_length = memory_length,
            augment        = False
        )

        train_vids = {1, 2, 3, 4, 5, 6}
        val_vids   = {7, 9}
        test_vids  = {10, 11}

    # CholecSeg8k dataset

    elif cholec_dir:

        dataset = CholecSeg8kDataset(
            data_dir       = cholec_dir,
            history_length = memory_length,
            augment        = False
        )

        train_vids = {1, 9, 12, 17, 18, 20, 24, 25, 26, 27, 28}
        val_vids   = {35, 37, 43}
        test_vids  = {48, 52, 55}

    else:
        print("Please select a correct dataset.")


    train_indices = []
    val_indices   = []
    test_indices  = []

    for idx, seq in enumerate(dataset.sequences):
        if isinstance(seq, tuple):
            files = seq[0]
        else:
            files = seq
        video_id = int(os.path.basename(files[0]).split('_')[0])
        if video_id in train_vids:
            train_indices.append(idx)
        elif video_id in val_vids:
            val_indices.append(idx)
        elif video_id in test_vids:
            test_indices.append(idx)

    train_set = Subset(dataset, train_indices)
    val_set   = Subset(dataset, val_indices)
    test_set  = Subset(dataset, test_indices)

    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 4)
    val_loader   = DataLoader(val_set,   batch_size = batch_size, shuffle = True, num_workers = 4)
    test_loader  = DataLoader(test_set,  batch_size = batch_size, shuffle = True, num_workers = 4)


    # SSTC-Seg

    if selected_model == "SSTC-Seg":

        model = SSTCSeg(
            num_classes        = num_classes,
            num_encoder_blocks = num_encoder_blocks,
            num_heads          = num_heads,
            k                  = k,
            memory_frames      = memory_length,
            num_memory_blocks  = num_memory_blocks,
            deformable         = deformable,
            use_crf            = use_crf
        )

    # U-Net

    elif selected_model == "U-Net":

        model = UNet(
            in_channels = 3,
            num_classes = num_classes
        )

    # CFPNet-M

    elif selected_model == "CFPNet-M":

        model = CFPNetM(
            in_channels = 3,
            num_classes = num_classes
        )

    # MFCPNet

    elif selected_model == "MFCPNet":

        model = MFCPNet(
            input_dim   = 3,
            num_classes = num_classes
        )

    elif selected_model == "PSPNet":

        model = PSPNet(
            in_channels         = 3,
            num_classes         = num_classes,
            pool_sizes          = (1, 2, 3, 6),
            ppm_reduction_dim   = 512,
            backbone_pretrained = True,
            dropout             = 0.1
        )
    
    else:
        print("Please select a correct model.")


    checkpoint = torch.load(model_dir)
    model.load_state_dict(checkpoint["model_state_dict"])


    test_model(
        model       = model,
        test_loader = test_loader,
        num_classes = num_classes,
        device      = device
    )