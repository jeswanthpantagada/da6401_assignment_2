import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import xml.etree.ElementTree as ET


def get_transforms(split='train', img_size=224):
    if split == 'train':
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.3),
            A.Rotate(limit=15, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='albumentations',
                                    label_fields=['class_labels'],
                                    min_visibility=0.1))
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='albumentations',
                                    label_fields=['class_labels'],
                                    min_visibility=0.1))


class PetDataset(Dataset):
    """
    Oxford-IIIT Pet Dataset.
    Provides: class label (37 breeds), bounding box, segmentation trimap.

    Trimap pixel values:
        1 = Foreground  → remapped to 0
        2 = Background  → remapped to 1
        3 = Boundary    → remapped to 2
    """
    def __init__(self, root: str, split: str = 'train',
                 img_size: int = 224, transform=None):
        self.root      = root
        self.split     = split
        self.img_size  = img_size
        self.transform = transform if transform else get_transforms(split, img_size)
        self.samples   = self._load_split()

    def _load_split(self):
        fname = 'trainval.txt' if self.split == 'train' else 'test.txt'
        fpath = os.path.join(self.root, 'annotations', fname)
        samples = []
        with open(fpath) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts     = line.strip().split()
                img_name  = parts[0]
                class_id  = int(parts[1]) - 1   # 0-indexed (0 to 36)
                samples.append({'name': img_name, 'class': class_id})
        return samples

    def _load_bbox(self, img_name, img_w, img_h):
        """Returns [x1, y1, x2, y2] normalized to [0,1]."""
        xml_path = os.path.join(self.root, 'annotations', 'xmls',
                                f'{img_name}.xml')
        if not os.path.exists(xml_path):
            return [0.0, 0.0, 1.0, 1.0]   # full image fallback

        tree = ET.parse(xml_path)
        root = tree.getroot()
        obj  = root.find('object')
        bnd  = obj.find('bndbox')
        x1 = float(bnd.find('xmin').text) / img_w
        y1 = float(bnd.find('ymin').text) / img_h
        x2 = float(bnd.find('xmax').text) / img_w
        y2 = float(bnd.find('ymax').text) / img_h
        return [
            max(0.0, min(x1, 1.0)),
            max(0.0, min(y1, 1.0)),
            max(0.0, min(x2, 1.0)),
            max(0.0, min(y2, 1.0)),
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample   = self.samples[idx]
        img_name = sample['name']
        class_id = sample['class']

        # Load image
        img_path = os.path.join(self.root, 'images', f'{img_name}.jpg')
        image    = np.array(Image.open(img_path).convert('RGB'))
        H, W     = image.shape[:2]

        # Load bbox [x1,y1,x2,y2] normalized
        bbox = self._load_bbox(img_name, W, H)

        # Load trimap mask (values 1,2,3 → remap to 0,1,2)
        mask_path = os.path.join(self.root, 'annotations', 'trimaps',
                                 f'{img_name}.png')
        mask = np.array(Image.open(mask_path)).astype(np.int64) - 1
        mask = np.clip(mask, 0, 2)

        # Apply albumentations transforms
        transformed = self.transform(
            image=image,
            mask=mask,
            bboxes=[bbox],
            class_labels=[class_id]
        )

        image = transformed['image']   # [3, H, W] float tensor
        mask  = transformed['mask']    # [H, W] int tensor

        # Convert bbox back to [cx, cy, w, h] normalized
        if transformed['bboxes']:
            x1, y1, x2, y2 = transformed['bboxes'][0]
        else:
            x1, y1, x2, y2 = 0.0, 0.0, 1.0, 1.0

        bbox_tensor = torch.tensor([
            (x1 + x2) / 2,   # cx
            (y1 + y2) / 2,   # cy
            (x2 - x1),       # w
            (y2 - y1),       # h
        ], dtype=torch.float32)

        return {
            'image' : image,
            'label' : torch.tensor(class_id, dtype=torch.long),
            'bbox'  : bbox_tensor,
            'mask'  : mask.long(),
        }
