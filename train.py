"""
Unified Training Script with W&B logging.
Supports Task 1 (classification), Task 2 (localization),
Task 3 (segmentation), Task 4 (multi-task).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import wandb
import numpy as np
from sklearn.metrics import f1_score

from dataset import PetDataset
from models.vgg11 import VGG11
from models.multitask import MultiTaskVGG, MultiTaskLoss
from losses import IoULoss
from models.segmentation import SegmentationLoss


# ── Helper Metrics ────────────────────────────────────────────────────────
def compute_dice(pred_logits, targets, num_classes=3, eps=1e-6):
    """Compute mean Dice Score over classes."""
    preds = pred_logits.argmax(dim=1)   # [B, H, W]
    dice  = 0.0
    for c in range(num_classes):
        p = (preds == c).float()
        g = (targets == c).float()
        intersection = (p * g).sum()
        dice += (2 * intersection + eps) / (p.sum() + g.sum() + eps)
    return (dice / num_classes).item()


def compute_pixel_accuracy(pred_logits, targets):
    preds   = pred_logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total   = targets.numel()
    return correct / total


def compute_iou_metric(pred_bbox, gt_bbox, eps=1e-6):
    """Batch IoU metric (not loss)."""
    px1 = pred_bbox[:, 0] - pred_bbox[:, 2] / 2
    py1 = pred_bbox[:, 1] - pred_bbox[:, 3] / 2
    px2 = pred_bbox[:, 0] + pred_bbox[:, 2] / 2
    py2 = pred_bbox[:, 1] + pred_bbox[:, 3] / 2

    gx1 = gt_bbox[:, 0] - gt_bbox[:, 2] / 2
    gy1 = gt_bbox[:, 1] - gt_bbox[:, 3] / 2
    gx2 = gt_bbox[:, 0] + gt_bbox[:, 2] / 2
    gy2 = gt_bbox[:, 1] + gt_bbox[:, 3] / 2

    ix1 = torch.max(px1, gx1); iy1 = torch.max(py1, gy1)
    ix2 = torch.min(px2, gx2); iy2 = torch.min(py2, gy2)
    inter = (ix2-ix1).clamp(0) * (iy2-iy1).clamp(0)
    pred_a = (px2-px1).clamp(0) * (py2-py1).clamp(0)
    gt_a   = (gx2-gx1).clamp(0) * (gy2-gy1).clamp(0)
    union  = pred_a + gt_a - inter
    return (inter / (union + eps)).mean().item()


# ── Task 1: Train VGG11 Classifier ────────────────────────────────────────
def train_vgg11(config=None):
    with wandb.init(config=config, project="da6401_assignment2", name="task1_vgg11"):
        cfg = wandb.config

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Dataset
        dataset    = PetDataset(root=cfg.data_root, split='train')
        val_size   = int(0.1 * len(dataset))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size,
                                  shuffle=False, num_workers=4)

        # Model
        model     = VGG11(num_classes=37, dropout_p=cfg.dropout_p).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

        wandb.watch(model, log='all', log_freq=100)

        best_val_acc = 0.0

        for epoch in range(cfg.epochs):
            # ── Train ──
            model.train()
            train_loss, train_correct, train_total = 0, 0, 0

            for batch in train_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)

                optimizer.zero_grad()
                logits = model(images)
                loss   = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_loss    += loss.item()
                preds          = logits.argmax(dim=1)
                train_correct += (preds == labels).sum().item()
                train_total   += labels.size(0)

            scheduler.step()

            # ── Validate ──
            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            all_preds, all_labels = [], []

            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(device)
                    labels = batch['label'].to(device)
                    logits = model(images)
                    loss   = criterion(logits, labels)

                    val_loss    += loss.item()
                    preds        = logits.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total   += labels.size(0)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            train_acc = train_correct / train_total
            val_acc   = val_correct   / val_total
            macro_f1  = f1_score(all_labels, all_preds, average='macro')

            wandb.log({
                'epoch'         : epoch + 1,
                'train/loss'    : train_loss / len(train_loader),
                'train/acc'     : train_acc,
                'val/loss'      : val_loss / len(val_loader),
                'val/acc'       : val_acc,
                'val/macro_f1'  : macro_f1,
                'lr'            : scheduler.get_last_lr()[0],
            })

            print(f"Epoch {epoch+1}/{cfg.epochs} | "
                  f"Train Loss: {train_loss/len(train_loader):.4f} | "
                  f"Val Acc: {val_acc:.4f} | F1: {macro_f1:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'checkpoints/vgg11_best.pth')
                wandb.save('checkpoints/vgg11_best.pth')

        print(f"Best Val Acc: {best_val_acc:.4f}")
        return model


# ── Task 4: Train Unified Multi-Task Model ─────────────────────────────────
def train_multitask(config=None):
    with wandb.init(config=config, project="da6401_assignment2", name="task4_multitask"):
        cfg = wandb.config
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Dataset
        dataset  = PetDataset(root=cfg.data_root, split='train')
        val_size = int(0.1 * len(dataset))
        train_ds, val_ds = random_split(dataset, [len(dataset)-val_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=4)

        # Model: load pretrained VGG11 backbone
        model = MultiTaskVGG(num_classes=37, seg_classes=3).to(device)
        vgg   = VGG11(num_classes=37)
        vgg.load_state_dict(torch.load('checkpoints/vgg11_best.pth', map_location=device))
        model.load_from_vgg11(vgg)

        criterion = MultiTaskLoss(lambda_cls=1.0, lambda_bbox=1.0, lambda_seg=1.0)
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

        wandb.watch(model, log='gradients', log_freq=50)

        for epoch in range(cfg.epochs):
            model.train()
            total_loss = 0
            task_losses = {'cls': 0, 'bbox': 0, 'seg': 0}

            for batch in train_loader:
                images  = batch['image'].to(device)
                labels  = batch['label'].to(device)
                bboxes  = batch['bbox'].to(device)
                masks   = batch['mask'].to(device)

                optimizer.zero_grad()
                cls_pred, bbox_pred, seg_pred = model(images)
                loss, sub_losses = criterion(
                    cls_pred, labels, bbox_pred, bboxes, seg_pred, masks
                )
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                for k in task_losses: task_losses[k] += sub_losses[k]

            scheduler.step()

            # Validation
            model.eval()
            val_cls_correct = 0; val_cls_total = 0
            val_iou = 0; val_dice = 0; val_steps = 0
            all_preds = []; all_labels = []

            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(device)
                    labels = batch['label'].to(device)
                    bboxes = batch['bbox'].to(device)
                    masks  = batch['mask'].to(device)

                    cls_pred, bbox_pred, seg_pred = model(images)

                    preds = cls_pred.argmax(1)
                    val_cls_correct += (preds == labels).sum().item()
                    val_cls_total   += labels.size(0)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                    val_iou  += compute_iou_metric(bbox_pred, bboxes)
                    val_dice += compute_dice(seg_pred, masks)
                    val_steps += 1

            f1   = f1_score(all_labels, all_preds, average='macro')
            n    = len(train_loader)

            wandb.log({
                'epoch'           : epoch + 1,
                'train/total_loss': total_loss / n,
                'train/cls_loss'  : task_losses['cls'] / n,
                'train/bbox_loss' : task_losses['bbox'] / n,
                'train/seg_loss'  : task_losses['seg'] / n,
                'val/cls_acc'     : val_cls_correct / val_cls_total,
                'val/macro_f1'    : f1,
                'val/mean_iou'    : val_iou / val_steps,
                'val/dice_score'  : val_dice / val_steps,
            })

            print(f"Epoch {epoch+1} | Loss: {total_loss/n:.4f} | "
                  f"F1: {f1:.3f} | IoU: {val_iou/val_steps:.3f} | "
                  f"Dice: {val_dice/val_steps:.3f}")

        torch.save(model.state_dict(), 'checkpoints/multitask_best.pth')


# ── Main entry ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import os
    os.makedirs('checkpoints', exist_ok=True)

    # Task 1 config
    config_task1 = {
        'data_root' : '/path/to/oxford-iiit-pet',
        'batch_size': 32,
        'lr'        : 1e-3,
        'epochs'    : 50,
        'dropout_p' : 0.5,
    }
    train_vgg11(config=config_task1)

    # Task 4 config
    config_task4 = {
        'data_root' : '/path/to/oxford-iiit-pet',
        'batch_size': 16,
        'lr'        : 5e-4,
        'epochs'    : 30,
    }
    train_multitask(config=config_task4)
