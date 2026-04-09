"""Training entrypoint for DA6401 Assignment 2."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import wandb
from sklearn.metrics import f1_score
import numpy as np

from data.pets_dataset import PetDataset
from models.classification import VGG11Classifier
from models.vgg11          import VGG11Encoder
from models.multitask      import MultiTaskVGG, MultiTaskLoss


# ── Metrics ────────────────────────────────────────────────────────────────
def compute_dice(pred_logits, targets, num_classes=3, eps=1e-6):
    preds = pred_logits.argmax(dim=1)
    dice  = 0.0
    for c in range(num_classes):
        p = (preds == c).float()
        g = (targets == c).float()
        inter = (p * g).sum()
        dice += (2 * inter + eps) / (p.sum() + g.sum() + eps)
    return (dice / num_classes).item()


def compute_iou(pred_bbox, gt_bbox, eps=1e-6):
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
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    pa = (px2-px1).clamp(0) * (py2-py1).clamp(0)
    ga = (gx2-gx1).clamp(0) * (gy2-gy1).clamp(0)
    return (inter / (pa + ga - inter + eps)).mean().item()


# ── Task 1: Train VGG11 Classifier ─────────────────────────────────────────
def train_classifier(data_root, epochs=50, batch_size=32,
                     lr=1e-3, dropout_p=0.5):
    wandb.init(project='da6401_assignment2', name='task1_vgg11_classifier')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset  = PetDataset(root=data_root, split='train')
    val_n    = int(0.1 * len(dataset))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_n, val_n])
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, num_workers=4)

    model     = VGG11Classifier(num_classes=37, dropout_p=dropout_p).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    wandb.watch(model, log='all', log_freq=100)
    best_val_acc = 0.0

    for epoch in range(epochs):
        # Train
        model.train()
        t_loss, t_correct, t_total = 0, 0, 0
        for batch in train_loader:
            imgs   = batch['image'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            t_loss    += loss.item()
            t_correct += (logits.argmax(1) == labels).sum().item()
            t_total   += labels.size(0)
        scheduler.step()

        # Validate
        model.eval()
        v_loss, v_correct, v_total = 0, 0, 0
        all_p, all_l = [], []
        with torch.no_grad():
            for batch in val_loader:
                imgs   = batch['image'].to(device)
                labels = batch['label'].to(device)
                logits = model(imgs)
                v_loss    += criterion(logits, labels).item()
                preds      = logits.argmax(1)
                v_correct += (preds == labels).sum().item()
                v_total   += labels.size(0)
                all_p.extend(preds.cpu().numpy())
                all_l.extend(labels.cpu().numpy())

        val_acc = v_correct / v_total
        f1      = f1_score(all_l, all_p, average='macro')

        wandb.log({'epoch': epoch+1,
                   'train/loss': t_loss/len(train_loader),
                   'train/acc' : t_correct/t_total,
                   'val/loss'  : v_loss/len(val_loader),
                   'val/acc'   : val_acc,
                   'val/f1'    : f1})

        print(f'Epoch {epoch+1}/{epochs} | '
              f'Train Loss: {t_loss/len(train_loader):.4f} | '
              f'Val Acc: {val_acc:.4f} | F1: {f1:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/vgg11_classifier.pth')

    wandb.finish()
    return model


# ── Task 4: Train Multi-Task Model ─────────────────────────────────────────
def train_multitask(data_root, epochs=30, batch_size=16, lr=5e-4):
    wandb.init(project='da6401_assignment2', name='task4_multitask')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset  = PetDataset(root=data_root, split='train')
    val_n    = int(0.1 * len(dataset))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_n, val_n])
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, num_workers=4)

    model     = MultiTaskVGG(num_classes=37, seg_classes=3).to(device)
    criterion = MultiTaskLoss(lambda_cls=1.0, lambda_bbox=1.0, lambda_seg=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    wandb.watch(model, log='gradients', log_freq=50)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        sub = {'cls': 0, 'bbox': 0, 'seg': 0}

        for batch in train_loader:
            imgs   = batch['image'].to(device)
            labels = batch['label'].to(device)
            bboxes = batch['bbox'].to(device)
            masks  = batch['mask'].to(device)

            optimizer.zero_grad()
            cls_p, bbox_p, seg_p = model(imgs)
            loss, losses = criterion(cls_p, labels, bbox_p, bboxes, seg_p, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            for k in sub: sub[k] += losses[k]

        scheduler.step()

        # Validate
        model.eval()
        v_correct = 0; v_total = 0
        v_iou = 0; v_dice = 0; v_steps = 0
        all_p = []; all_l = []
        with torch.no_grad():
            for batch in val_loader:
                imgs   = batch['image'].to(device)
                labels = batch['label'].to(device)
                bboxes = batch['bbox'].to(device)
                masks  = batch['mask'].to(device)
                cls_p, bbox_p, seg_p = model(imgs)
                preds      = cls_p.argmax(1)
                v_correct += (preds == labels).sum().item()
                v_total   += labels.size(0)
                all_p.extend(preds.cpu().numpy())
                all_l.extend(labels.cpu().numpy())
                v_iou  += compute_iou(bbox_p, bboxes)
                v_dice += compute_dice(seg_p, masks)
                v_steps += 1

        n  = len(train_loader)
        f1 = f1_score(all_l, all_p, average='macro')

        wandb.log({'epoch'           : epoch+1,
                   'train/total_loss': total_loss/n,
                   'train/cls_loss'  : sub['cls']/n,
                   'train/bbox_loss' : sub['bbox']/n,
                   'train/seg_loss'  : sub['seg']/n,
                   'val/cls_acc'     : v_correct/v_total,
                   'val/f1'          : f1,
                   'val/iou'         : v_iou/v_steps,
                   'val/dice'        : v_dice/v_steps})

        print(f'Epoch {epoch+1}/{epochs} | Loss: {total_loss/n:.4f} | '
              f'F1: {f1:.3f} | IoU: {v_iou/v_steps:.3f} | '
              f'Dice: {v_dice/v_steps:.3f}')

    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/multitask.pth')
    wandb.finish()


# ── Entry point ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    DATA_ROOT = '/path/to/oxford-iiit-pet'   # ← CHANGE THIS to your dataset path

    train_classifier(DATA_ROOT)
    train_multitask(DATA_ROOT)
