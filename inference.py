"""Inference and evaluation for DA6401 Assignment 2."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import wandb
from data.pets_dataset import PetDataset
from models.multitask  import MultiTaskVGG
from torch.utils.data  import DataLoader


def run_inference(data_root, checkpoint_path, num_samples=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MultiTaskVGG(num_classes=37, seg_classes=3).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    dataset = PetDataset(root=data_root, split='test')
    loader  = DataLoader(dataset, batch_size=1, shuffle=True)

    CLASS_COLORS = {0: [0, 128, 0], 1: [0, 0, 128], 2: [128, 0, 0]}

    wandb.init(project='da6401_assignment2', name='inference')

    count = 0
    with torch.no_grad():
        for batch in loader:
            if count >= num_samples:
                break

            imgs   = batch['image'].to(device)
            labels = batch['label']
            bboxes = batch['bbox']
            masks  = batch['mask']

            cls_p, bbox_p, seg_p = model(imgs)

            pred_class = cls_p.argmax(1).item()
            pred_bbox  = bbox_p[0].cpu().numpy()
            pred_mask  = seg_p[0].argmax(0).cpu().numpy()

            # Denormalize image
            img_np = imgs[0].cpu().permute(1, 2, 0).numpy()
            mean   = np.array([0.485, 0.456, 0.406])
            std    = np.array([0.229, 0.224, 0.225])
            img_np = np.clip(img_np * std + mean, 0, 1)
            H, W   = img_np.shape[:2]

            # Colorize mask
            rgb_mask = np.zeros((H, W, 3), dtype=np.uint8)
            for c, col in CLASS_COLORS.items():
                rgb_mask[pred_mask == c] = col

            # Plot
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Original + bbox
            axes[0].imshow(img_np)
            cx, cy, bw, bh = pred_bbox
            x1 = (cx - bw/2) * W; y1 = (cy - bh/2) * H
            rect = patches.Rectangle((x1, y1), bw*W, bh*H,
                                      linewidth=2, edgecolor='red',
                                      facecolor='none', label='Pred')
            # GT bbox
            gcx, gcy, gbw, gbh = bboxes[0].numpy()
            gx1 = (gcx - gbw/2)*W; gy1 = (gcy - gbh/2)*H
            gt_rect = patches.Rectangle((gx1, gy1), gbw*W, gbh*H,
                                         linewidth=2, edgecolor='green',
                                         facecolor='none', label='GT')
            axes[0].add_patch(rect)
            axes[0].add_patch(gt_rect)
            axes[0].set_title(f'Class: {pred_class} | GT: {labels[0].item()}')
            axes[0].legend()
            axes[0].axis('off')

            # GT mask
            gt_mask_rgb = np.zeros((H, W, 3), dtype=np.uint8)
            for c, col in CLASS_COLORS.items():
                gt_mask_rgb[masks[0].numpy() == c] = col
            axes[1].imshow(gt_mask_rgb)
            axes[1].set_title('GT Mask')
            axes[1].axis('off')

            # Pred mask
            axes[2].imshow(rgb_mask)
            axes[2].set_title('Predicted Mask')
            axes[2].axis('off')

            plt.tight_layout()
            wandb.log({f'inference/sample_{count}': wandb.Image(fig)})
            plt.close(fig)
            count += 1

    wandb.finish()
    print(f'Logged {count} inference samples to W&B.')


if __name__ == '__main__':
    DATA_ROOT   = '/path/to/oxford-iiit-pet'      # ← CHANGE THIS
    CHECKPOINT  = 'checkpoints/multitask.pth'
    run_inference(DATA_ROOT, CHECKPOINT)
