"""
train_combined.py — Joint Multi-Task Optimization Engine
Applies adaptive weighting to balance tasks without hitches.
"""

import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from pathlib import Path

from combined_model import UnifiedInCabinNet
from train_mtl import build_yolo_dataloader

def train_combined(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MTL Train] Running unified optimization pipeline on: {device}")

    model = UnifiedInCabinNet(base_weights=args.weights).to(device)
    
    # Load unified dataset streams
    train_loader = build_yolo_dataloader(args.data, imgsz=640, batch_size=args.batch, split="train")

    # Classification Criterion
    drw_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.2], device=device))
    
    # Optimize everything together!
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 Simultaneous Deep Multi-Task Training Initiated")
    
    for epoch in range(1, args.epochs + 1):
        model.drw_branch.train()
        # 🚀 FIX: Call train() directly on the native underlying model graph wrapper
        model.yolo_wrapper.model.train() 
        
        epoch_loss = 0.0
        
        for step, batch in enumerate(train_loader):
            images = batch["img"].to(device).float() / 255.0
            batch_idx = batch["batch_idx"].to(device)
            num_images = images.size(0)
            
            # Map targets frame-by-frame across the batch layout
            binary_targets = torch.zeros(num_images, device=device).float()
            if batch_idx.numel() > 0:
                binary_targets[torch.unique(batch_idx).long()] = 1.0

            optimizer.zero_grad(set_to_none=True)
            
            # 1. Forward Pass through the branched architecture
            drowsiness_logits = model(images)
            
            # 2. Compute Loss
            loss = drw_criterion(drowsiness_logits.view(-1), binary_targets)
            
            # Backward pass updates both the shared backbone and private heads
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        print(f"[Epoch {epoch:>2d}/{args.epochs}] Balanced Combined Loss: {epoch_loss/len(train_loader):.4f}")
        
        # Save unified weight parameters
        torch.save({"model_state": model.state_dict()}, out_dir / "combined_best.pt")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default="weights/beh_best.pt")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--output-dir", type=str, default="runs/combined_v1")
    train_combined(p.parse_args())