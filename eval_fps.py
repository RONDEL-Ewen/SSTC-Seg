#!/usr/bin/env python3
"""
eval_fps.py

Measure inference speed (FPS) of a multi-organ segmentation model.

Usage:
    python eval_fps.py \
        --config config.yaml \
        --checkpoint model/weights/best_model_checkpoint.pth \
        [--device cuda] \
        [--batch_size 2] \
        [--num_workers 4] \
        [--warmup_batches 10] \
        [--num_batches 100]
"""

import argparse
import yaml
import os
import time
import warnings

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset, DataLoader

# Datasets
from datasets.DresdenDataset import DresdenDataset
from datasets.HemoSetDataset import HemoSetDataset
from datasets.CholecSeg8kDataset import CholecSeg8kDataset

# Models
from model.SSTCSeg import SSTCSeg
from baselines.UNet import UNet
from baselines.CFPNetM import CFPNetM
from baselines.MFCPNet import MFCPNet
from baselines.PSPNet import PSPNet

warnings.simplefilter("ignore", category=FutureWarning)

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

def build_model(selected_model, num_classes, memory_length, num_encoder_blocks, num_heads, k, num_memory_blocks, deformable, use_crf, device):

    if selected_model == "SSTC-Seg":
        m = SSTCSeg(
            num_classes        = num_classes,
            num_encoder_blocks = num_encoder_blocks,
            num_heads          = num_heads,
            k                  = k,
            memory_frames      = memory_length,
            num_memory_blocks  = num_memory_blocks,
            deformable         = deformable,
            use_crf            = use_crf
        )
    elif selected_model == "U-Net":
        m = UNet(in_channels=3, num_classes=num_classes)
    elif selected_model == "CFPNet-M":
        m = CFPNetM(in_channels=3, num_classes=num_classes)
    elif selected_model == "MFCPNet":
        m = MFCPNet(input_dim=3, num_classes=num_classes)
    elif selected_model == "PSPNet":
        m = PSPNet(
            in_channels=3,
            num_classes=num_classes,
            pool_sizes=(1,2,3,6),
            ppm_reduction_dim=512,
            backbone_pretrained=True,
            dropout=0.1
        )
    else:
        raise ValueError(f"Unknown model: {selected_model}")
    return m.to(device).to(memory_format=torch.channels_last)

def main():
    #args = parse_args()
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

    # set up cudnn
    cudnn.benchmark = True

    # build data
    nb_frames = 2000
    frames = [
        torch.randn(1, 3, 80, 120, device=device).to(memory_format=torch.channels_last)
        for _ in range(nb_frames)
    ]

    # build and load model
    #device = torch.device(args.device)
    model = build_model(
        selected_model=selected_model,
        num_classes=num_classes,
        memory_length=memory_length,
        num_encoder_blocks=num_encoder_blocks,
        num_heads=num_heads,
        k=k,
        num_memory_blocks=num_memory_blocks,
        deformable=deformable,
        use_crf=use_crf,
        device=device
    )
    #ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt = torch.load(model_dir)
    # handle state dict inside checkpoint
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd)
    model.eval()
    #print(f"Loaded checkpoint: {args.checkpoint}")

    # inference speed test
    t0 = time.time()
    with torch.no_grad():
        for i in range(nb_frames):
            _ = model(frames[i], False)
    if device == "cuda": torch.cuda.synchronize()
    t1 = time.time()

    elapsed = t1 - t0
    fps = len(frames) / elapsed if elapsed > 0 else float("inf")
    print("----------------------------")
    print(f"Frames processed: {len(frames)}")
    print(f"Elapsed time   : {elapsed:.3f}s")
    print(f"Inference speed: {fps:.2f} FPS")
    print("----------------------------")

if __name__ == "__main__":
    main()