# moq-nas/core/fairness/data.py

import json
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Allows loading of potentially truncated image files, common in large datasets
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- 1. Dataset Classes ----------------------------------------------------

class BinaryFolderDataset(Dataset):
    """
    Dataset for binary classification from a folder structure.
    Expects the following structure: root/{train,val}/{pos_name,neg_name}/*
    
    This class is flexible enough to handle both 'person'/'non_person' and
    'face'/'non_face' datasets by auto-detecting folder names if not specified.
    """
    def __init__(self, root: str, split: str = "train", tfm: Optional[transforms.Compose] = None,
                 pos_name: Optional[str] = None, neg_name: Optional[str] = None):
        
        self.root = Path(root) / split
        if not self.root.is_dir():
            raise FileNotFoundError(f"Directory for split '{split}' not found at: {self.root}")
            
        self.tfm = tfm
        self.samples: List[Tuple[Path, int]] = []
        
        # --- Auto-detection of class folder names ---
        if pos_name is None or neg_name is None:
            subdirs = {d.name for d in self.root.iterdir() if d.is_dir()}
            if {"person", "non_person"}.issubset(subdirs):
                pos_name, neg_name = "person", "non_person"
            elif {"face", "non_face"}.issubset(subdirs):
                pos_name, neg_name = "face", "non_face"
            else:
                raise ValueError(f"Could not auto-detect class folders in {self.root}. "
                                 f"Expected ('person', 'non_person') or ('face', 'non_face'), but found: {subdirs}")
        
        exts = {".jpg", ".jpeg", ".png"}
        # The positive class (person/face) will always have the label 1
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
    It loads images, crops faces according to the CSV coordinates, and returns
    the soft skin tone probabilities as a tensor.
    """
    def __init__(self, csv_path: str, tfm: Optional[transforms.Compose] = None):
        self.df = pd.read_csv(csv_path)
        self.tfm = tfm
        # Verification of essential columns
        required_cols = {"image_path", "x", "y", "width", "height", "skin_tone_probs"}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(f"FACET CSV must contain the following columns: {required_cols}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        box = (row["x"], row["y"], row["x"] + row["width"], row["y"] + row["height"])
        
        img = Image.open(img_path).convert("RGB").crop(box)
        if self.tfm:
            img = self.tfm(img)
            
        # Parse the JSON string of probabilities into a tensor
        soft_labels = torch.tensor(json.loads(row["skin_tone_probs"]), dtype=torch.float32)
        return img, soft_labels


# --- 2. Factory Functions for DataLoaders and Transforms -------------------

def get_default_transforms(img_size: int = 224) -> dict:
    """
    Returns a dictionary with standard train and validation transforms.
    The training pipeline includes TrivialAugmentWide, a state-of-the-art
    automatic augmentation policy, to align with moq-nas retraining practices.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    # --- Training Transforms ---
    # This pipeline applies strong, automatic augmentation.
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    
    # --- Validation/Evaluation Transforms ---
    # This pipeline is deterministic and used for validation and testing.
    val_transforms = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])
    
    return {'train': train_transforms, 'val': val_transforms}

def create_binary_loaders(data_root: str, batch_size: int, num_workers: int = 4, 
                        img_size: int = 224, pos_name: str = None, neg_name: str = None
                        ) -> Tuple[DataLoader, DataLoader]:
    """
    Creates training and validation DataLoaders for a binary dataset.
    """
    transforms_dict = get_default_transforms(img_size)
    
    train_dataset = BinaryFolderDataset(
        root=data_root, split='train', tfm=transforms_dict['train'], 
        pos_name=pos_name, neg_name=neg_name
    )
    val_dataset = BinaryFolderDataset(
        root=data_root, split='val', tfm=transforms_dict['val'],
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


def create_eval_loader(dataset_name: str, csv_path: str, batch_size: int, 
                    num_workers: int = 4, img_size: int = 224) -> DataLoader:
    """
    Creates a DataLoader for a fairness evaluation dataset (e.g., FACET).
    """
    # For evaluation, we always use the validation transforms
    val_transforms = get_default_transforms(img_size)['val']

    if dataset_name.lower() == 'facet':
        dataset = FacetEvalDataset(csv_path, tfm=val_transforms)
    # You could add 'fairface' here in the future
    # elif dataset_name.lower() == 'fairface':
    #     dataset = FairFaceEvalDataset(csv_path, tfm=val_transforms)
    else:
        raise ValueError(f"Evaluation dataset '{dataset_name}' is not supported.")
        
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )