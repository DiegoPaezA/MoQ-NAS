# moq-nas/core/fairness/data.py

import json
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- 1. Dataset Classes (These are unchanged) ---

class BinaryFolderDataset(Dataset):
    """Dataset for binary classification from a folder structure."""
    # ... (This class remains exactly the same) ...
    def __init__(self, root: str, split: str = "train", tfm: Optional[transforms.Compose] = None,
                pos_name: Optional[str] = None, neg_name: Optional[str] = None):
        
        self.root = Path(root) / split
        if not self.root.is_dir():
            raise FileNotFoundError(f"Directory for split '{split}' not found at: {self.root}")
            
        self.tfm = tfm
        self.samples: List[Tuple[Path, int]] = []
        
        if pos_name is None or neg_name is None:
            subdirs = {d.name for d in self.root.iterdir() if d.is_dir()}
            if {"person", "non_person"}.issubset(subdirs):
                pos_name, neg_name = "person", "non_person"
            elif {"face", "non_face"}.issubset(subdirs):
                pos_name, neg_name = "face", "non_face"
            else:
                raise ValueError(f"Could not auto-detect class folders in {self.root}.")
        
        exts = {".jpg", ".jpeg", ".png"}
        for label, class_name in [(1, pos_name), (0, neg_name)]:
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Class folder not found: {class_dir}")
            
            for p in class_dir.glob("**/*"):
                if p.suffix.lower() in exts:
                    self.samples.append((p, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.tfm:
            img = self.tfm(img)
        return img, label


class FacetEvalDataset(Dataset):
    """
    Dataset for fairness evaluation on FACET (skin tone).
    It loads images, crops faces, and can use an on-disk cache to speed up re-runs.
    """
    def __init__(self, csv_path: str, tfm: Optional[transforms.Compose] = None, cache_dir: Optional[str] = None):
        self.df = pd.read_csv(csv_path)
        self.tfm = tfm
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        required_cols = {"image_path", "x", "y", "width", "height", "skin_tone_probs"}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(f"FACET CSV must contain columns: {required_cols}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        
        if self.cache_dir:
            # Create a unique, stable filename for the crop
            p = Path(row["image_path"])
            cached_filename = f"{p.stem}_{int(row['x'])}_{int(row['y'])}_{int(row['width'])}_{int(row['height'])}.jpg"
            cached_path = self.cache_dir / cached_filename
            
            if cached_path.exists():
                img = Image.open(cached_path).convert("RGB")
            else:
                # If not cached, load, crop, and save to cache
                img = Image.open(row["image_path"]).convert("RGB")
                box = (row["x"], row["y"], row["x"] + row["width"], row["y"] + row["height"])
                img = img.crop(box)
                try:
                    img.save(cached_path, quality=95)
                except Exception:
                    pass # Non-critical if cache write fails
        else:
            # Original behavior if no cache is provided
            img_path = row["image_path"]
            box = (row["x"], row["y"], row["x"] + row["width"], row["y"] + row["height"])
            img = Image.open(img_path).convert("RGB").crop(box)

        if self.tfm:
            img = self.tfm(img)
            
        soft_labels = torch.tensor(json.loads(row["skin_tone_probs"]), dtype=torch.float32)
        return img, soft_labels

class FairFaceEvalDataset(Dataset):
    """Dataset for fairness evaluation on FairFace (race)."""
    # ... (This class remains exactly the same) ...
    def __init__(self, csv_path: str, tfm: Optional[transforms.Compose] = None):
        self.df = pd.read_csv(csv_path)
        self.tfm = tfm
        required_cols = {"image_path", "race"}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(f"FairFace CSV must contain columns: {required_cols}")
            
        self.race_labels = sorted(self.df['race'].unique())
        self.race_to_idx = {race: i for i, race in enumerate(self.race_labels)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        race = row["race"]
        
        img = Image.open(img_path).convert("RGB")
        if self.tfm:
            img = self.tfm(img)
            
        label_idx = self.race_to_idx[race]
        return img, label_idx


# --- 2. Factory Functions for DataLoaders and Transforms -------------------

def get_default_transforms(img_size: int = 224) -> dict:
    """Returns a dictionary with standard train and validation transforms."""
    # ... (This function remains exactly the same) ...
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])
    
    return {'train': train_transforms, 'val': val_transforms}

def create_binary_loaders(data_root: str, batch_size: int, num_workers: int, 
                        tf_train: transforms.Compose, tf_val: transforms.Compose,
                          pos_name: str = None, neg_name: str = None, **kwargs
                        ) -> Tuple[DataLoader, DataLoader]:
    """
    Creates training and validation DataLoaders for a binary dataset.
    This version now correctly accepts pre-made transforms as arguments.
    """
    # It no longer creates its own transforms, it uses the ones passed in.
    
    train_dataset = BinaryFolderDataset(
        root=data_root, split='train', tfm=tf_train, 
        pos_name=pos_name, neg_name=neg_name
    )
    val_dataset = BinaryFolderDataset(
        root=data_root, split='val', tfm=tf_val,
        pos_name=pos_name, neg_name=neg_name
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader


def create_eval_loader(dataset_name: str, csv_path: str, batch_size: int, img_size: int = 224, 
                    cache_dir: Optional[str] = ".cache/facet_crops", phase: Optional[str] = None) -> DataLoader:
    """Creates a DataLoader for a fairness evaluation dataset, optimizing for the execution phase."""
    
    val_transforms = get_default_transforms(img_size)['val']

    if dataset_name.lower() == 'facet':
        dataset = FacetEvalDataset(csv_path, tfm=val_transforms, cache_dir=cache_dir)
    elif dataset_name.lower() == 'fairface':
        dataset = FairFaceEvalDataset(csv_path, tfm=val_transforms)
    else:
        raise ValueError(f"Evaluation dataset '{dataset_name}' is not supported. Use 'facet' or 'fairface'.")

    is_evolution = (phase == 'evolutionx')

    num_workers = 0 if is_evolution else 4
    pin_memory = False if is_evolution else True

    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=pin_memory
    )