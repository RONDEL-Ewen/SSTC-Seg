from scipy.ndimage import distance_transform_edt, binary_erosion
import numpy as np
import warnings
import argparse
import yaml
import os

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
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



# ===============================[ Arguments ]===============================

def load_config(
    config_path = "config.yaml"
):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config



# ===============================[ Metrics ]===============================

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



# ===============================[ Training ]===============================

def train_model(
    model,
    model_dir,
    train_loader,
    val_loader,
    test_loader,
    num_classes = 8,
    num_epochs  = 1000,
    lr          = 1e-4,
    patience    = 4,
    device      = "cuda"
):
    
    model = model.to(device).to(memory_format=torch.channels_last) #TEST

    optimizer = optim.AdamW(model.parameters(), lr = lr)

    criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr          = lr,
        steps_per_epoch = len(train_loader),
        epochs          = num_epochs,
        anneal_strategy = 'cos'
    )

    best_val_loss = np.inf
    patience_counter = 0
    best_model_state = None
    nan_detected = False

    #train_loss_history = []
    #val_loss_history = []

    # TEST
    start_epoch      = 0
    best_val_loss    = np.inf
    patience_counter = 0
    train_loss_history, val_loss_history = [], []
    # TEST
    if model_dir is not None and os.path.isfile(model_dir):
        print(f"🔄️ Resuming from checkpoint {model_dir} …")
        ckpt = torch.load(model_dir, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch        = ckpt["epoch"] + 1
        best_val_loss      = ckpt["best_val_loss"]
        patience_counter   = ckpt["patience_counter"]
        train_loss_history = ckpt["train_loss_history"]
        val_loss_history   = ckpt["val_loss_history"]
        print(f"✅ Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}, patience_counter={patience_counter}")

    epochs_counter = 0
    #for epoch in range(num_epochs):
    # TEST
    for epoch in range(start_epoch, num_epochs):
        epochs_counter += 1
        model.train()
        total_loss = 0
        images_counter = 0

        for images, masks in train_loader:
            images_counter += 1
            images = [img.squeeze(1).to(device) for img in images]
            masks  = [mask.squeeze(1).to(device) for mask in masks]
            optimizer.zero_grad()
            
            current_frame = images[-1]
            current_mask  = masks[-1]
            memory_frames = images[:-1]
            memory_masks  = masks[:-1]

            prediction = model(current_frame, training=True, memory_frames=memory_frames, memory_masks=memory_masks)
            loss = criterion(prediction, current_mask.argmax(dim=1).long())
            print(f"[{epochs_counter}/{num_epochs}] [{images_counter}/{len(train_loader)}] Training loss: {loss}")

            if torch.isnan(loss).any():
                print(f"⚠️ NaN loss detected during training at epoch n°{epoch+1}, batch n°{images_counter}.")
                nan_detected = True
                break

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            del current_frame, current_mask, memory_frames, memory_masks, prediction, loss
            torch.cuda.empty_cache()

        avg_train_loss = total_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            images_counter = 0
            for images, masks in val_loader:
                images_counter += 1
                images = [img.squeeze(1).to(device) for img in images]
                masks  = [mask.squeeze(1).to(device) for mask in masks]
                current_frame = images[-1]
                current_mask  = masks[-1]
                memory_frames = images[:-1]
                memory_masks  = masks[:-1]

                prediction = model(current_frame, training=True, memory_frames=memory_frames, memory_masks=memory_masks)
                loss = criterion(prediction, current_mask.argmax(dim=1).long())
                print(f"[{epochs_counter}/{num_epochs}] [{images_counter}/{len(val_loader)}] Validation loss: {loss}")

                if torch.isnan(loss).any():
                    print(f"⚠️ NaN loss detected during validation at epoch n°{epoch+1}, batch n°{images_counter}.")
                    nan_detected = True
                    break

                val_loss += loss.item()

                del current_frame, current_mask, memory_frames, memory_masks, prediction, loss
                torch.cuda.empty_cache()

        if nan_detected:
            break

        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        print("------------------------------")
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print("------------------------------")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
            #torch.save({
            #    "epoch": epoch,
            #    "model_state_dict": model.state_dict(),
            #    "optimizer_state_dict": optimizer.state_dict(),
            #    "val_loss": best_val_loss
            #}, r"model\weights\best_model_checkpoint.pth")

            #TEST 
            state = {
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss":        best_val_loss,
                "patience_counter":     patience_counter,
                "train_loss_history":   train_loss_history,
                "val_loss_history":     val_loss_history
            }
            torch.save(state, r"model\weights\best_model_checkpoint.pth")

            print("🔹 New best model found and saved !")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement since {patience_counter} epoch(s)...")

        if patience_counter >= patience:
            print(f"⏹️ Early stopping activated after {epoch+1} epochs. Best val loss = {best_val_loss:.4f}")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)
        print("✅ Best model reloaded !")

        model.eval()

        test_loss = 0
        accuracies = []
        dices = []
        jaccards = []
        hd95s = []

        with torch.no_grad():
            for images, masks in test_loader:
                images = [img.squeeze(1).to(device) for img in images]
                masks  = [mask.squeeze(1).to(device) for mask in masks]
                current_frame = images[-1]
                current_mask  = masks[-1]
                memory_frames = images[:-1]
                memory_masks  = masks[:-1]

                prediction = model(current_frame, training=True, memory_frames=memory_frames, memory_masks=memory_masks)
                loss = criterion(prediction, current_mask.argmax(dim=1).long())
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

                accuracies.append(sum(batch_acc) / len(batch_acc))
                dices.append(sum(batch_dice) / len(batch_dice))
                jaccards.append(sum(batch_jaccard) / len(batch_jaccard))
                hd95s.append(sum(batch_hd95) / len(batch_hd95))

                del current_frame, current_mask, memory_frames, memory_masks, prediction, loss
                torch.cuda.empty_cache()

        avg_test_loss = test_loss / len(test_loader)
        avg_accuracy = np.mean(accuracies)
        avg_dice = np.mean(dices)
        avg_jaccard = np.mean(jaccards)
        avg_hd95 = np.mean([v for v in hd95s if np.isfinite(v)])

        print("------------------------------")
        print(f"Test Loss:             {avg_test_loss:.5f}")
        print(f"Accuracy:              {avg_accuracy:.5f}")
        print(f"Dice Coefficient:      {avg_dice:.5f}")
        print(f"Jaccard Index:         {avg_jaccard:.5f}")
        print(f"Hausdorff Distance 95: {avg_hd95:.5f}")
        print("------------------------------")

    print("🏁 Training finished !")



# ===============================[ Main ]===============================

if __name__ == '__main__':

    config = load_config()


    # Datasets arguments

    dresden_dir = config.get("dresden_dir")
    hemoset_dir = config.get("hemoset_dir")
    cholec_dir  = config.get("cholec_dir")

    num_classes = config.get("num_classes", 8)


    # Model arguments

    selected_model     = config.get("selected_model", "SSTC-Seg")
    model_dir          = config.get("model_dir", None)

    num_encoder_blocks = config.get("num_encoder_blocks", 3)
    num_heads          = config.get("num_heads", 8)
    k                  = config.get("k", 5)
    memory_length      = config.get("memory_length", 3)
    num_memory_blocks  = config.get("num_memory_blocks", 3)
    deformable         = config.get("deformable", True)
    use_crf            = config.get("use_crf", True)


    # Training arguments

    augmentation = config.get("augmentation", False)
    batch_size   = config.get("batch_size", 2)
    epochs       = config.get("epochs", 1000)
    lr           = config.get("lr", 1e-4)
    patience     = config.get("patience", 5)
    device       = config.get("device", "cuda")


    # Dresden dataset

    if dresden_dir:

        print("🔄️ Loading Dresden dataset...")

        dataset = DresdenDataset(
            data_dir       = dresden_dir,
            history_length = memory_length,
            augment        = augmentation
        )

        #train_vids = {4, 5, 8, 10, 12, 16, 17, 22, 23, 24, 25, 27, 29, 30, 31}
        #val_vids   = {2, 7, 11, 18, 20}
        #test_vids  = {3, 21, 26}
        train_vids = {2, 5, 7, 8, 12, 16, 17, 21, 23, 24, 25, 27, 31}
        val_vids   = {3, 11, 18, 20, 29}
        test_vids  = {4, 10, 22, 26, 30}

    # HemoSet dataset

    elif hemoset_dir:

        print("🔄️ Loading HemoSet dataset...")

        dataset = HemoSetDataset(
            data_dir       = hemoset_dir,
            history_length = memory_length,
            augment        = augmentation
        )

        train_vids = {1, 2, 3, 4, 5, 6}
        val_vids   = {7, 9}
        test_vids  = {10, 11}

    # CholecSeg8k dataset

    elif cholec_dir:
        
        print("🔄️ Loading CholecSeg8k dataset...")

        dataset = CholecSeg8kDataset(
            data_dir       = cholec_dir,
            history_length = memory_length,
            augment        = augmentation
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

        print("🔄️ Creating SSTC-Seg...")

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

        print("🔄️ Creating U-Net...")

        model = UNet(
            in_channels = 3,
            num_classes = num_classes
        )

    # CFPNet-M

    elif selected_model == "CFPNet-M":

        print("🔄️ Creating CFPNet-M...")

        model = CFPNetM(
            in_channels = 3,
            num_classes = num_classes
        )

    # MFCPNet

    elif selected_model == "MFCPNet":

        print("🔄️ Creating MFCPNet...")

        model = MFCPNet(
            input_dim   = 3,
            num_classes = num_classes
        )

    elif selected_model == "PSPNet":

        print("🔄️ Creating PSPNet...")

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


    #if model_dir:
        #print("🔄️ Loading model...")
        #checkpoint = torch.load(model_dir)
        #model.load_state_dict(checkpoint["model_state_dict"])
        #print("✅ Model loaded !")
        

    print("🔄️ Training model...")

    train_model(
        model        = model,
        model_dir    = model_dir,
        train_loader = train_loader,
        val_loader   = val_loader,
        test_loader  = test_loader,
        num_classes  = num_classes,
        num_epochs   = epochs,
        lr           = lr,
        patience     = patience,
        device       = device
    )