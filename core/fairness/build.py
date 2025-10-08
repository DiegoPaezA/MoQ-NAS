# moq-nas/core/fairness/build.py

import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

# Ensure pycocotools is installed for COCO processing
try:
    from pycocotools.coco import COCO
except ImportError:
    print("Warning: pycocotools not found. Please install it (`pip install pycocotools`) to use COCO building functions.")
    COCO = None

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- General Helper Functions ---

def _ensure_dir(p: Path):
    """Creates a directory if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)

def _save_crop(img: Image.Image, box: Tuple[int, int, int, int], out_path: Path) -> bool:
    """Saves a cropped region of an image, returns True on success."""
    try:
        crop = img.crop(box)
        crop.save(out_path, format="JPEG", quality=90)
        return True
    except Exception:
        return False

# --- 1. FACET CSV Builder Functions -------------------------------------------

def build_facet_csv(ann_csv: str, img_dirs: List[str], out_csv: str):
    """
    Processes raw FACET annotations to create a standardized evaluation CSV.
    """
    print("Building standardized FACET evaluation CSV...")
    df = pd.read_csv(ann_csv)
    path_map = {p.name: str(p) for dir_path in img_dirs for p in Path(dir_path).glob("*") if p.is_file()}
    df["image_path"] = df["filename"].map(path_map)
    df.dropna(subset=["image_path"], inplace=True)

    hard_labels, soft_probs = [], []
    skin_tone_cols = [f"skin_tone_{i}" for i in range(1, 11)]
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing FACET annotations"):
        counts = [int(row.get(col, 0)) for col in skin_tone_cols]
        total_votes = sum(counts)
        probs = [c / total_votes for c in counts] if total_votes > 0 else [0.0] * 10
        # Simple hard label: highest vote, ties broken by highest tone index
        hard_label = np.argmax(counts) + 1 if total_votes > 0 else None
        soft_probs.append(json.dumps(probs))
        hard_labels.append(hard_label)

    bbox_data = df["bounding_box"].apply(json.loads)
    df_out = pd.DataFrame({
        "image_path": df["image_path"],
        "x": bbox_data.apply(lambda b: b.get("x")), "y": bbox_data.apply(lambda b: b.get("y")),
        "width": bbox_data.apply(lambda b: b.get("width")), "height": bbox_data.apply(lambda b: b.get("height")),
        "skin_tone_final": hard_labels, "skin_tone_probs": soft_probs,
    })
    df_out.dropna(inplace=True)
    df_out["skin_tone_final"] = df_out["skin_tone_final"].astype(int)
    _ensure_dir(Path(out_csv).parent)
    df_out.to_csv(out_csv, index=False)
    print(f"✅ Successfully created FACET CSV with {len(df_out)} rows at: {out_csv}")

# --- 2. Face Binary Dataset Builder (from WIDER) ------------------------------

def _read_wider_annotations(txt_path: str) -> List[Tuple[str, List[Tuple[int, int, int, int]]]]:
    """Robustly parses WIDER Face annotation files."""
    items = []
    with open(txt_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    i = 0
    while i < len(lines):
        if not lines[i].lower().endswith(".jpg"): i += 1; continue
        path = lines[i]; i += 1
        num_boxes_line = lines[i]; i += 1
        try:
            num_boxes = int(num_boxes_line)
            box_data = lines[i:i+num_boxes]; i += num_boxes
        except ValueError: # Handles format without explicit box count
            box_data = [num_boxes_line]
            while i < len(lines) and not lines[i].lower().endswith(".jpg"):
                box_data.append(lines[i]); i += 1

        boxes = []
        for line in box_data:
            parts = [int(p) for p in line.split()[:4]]
            if len(parts) == 4 and parts[2] > 0 and parts[3] > 0:
                boxes.append(tuple(parts))
        if boxes: items.append((path, boxes))
    return items

def _build_face_binary_split(split: str, wider_root: Path, neg_pool: List[Path], out_dir: Path, **kwargs):
    """Processes one split (train/val) for the face binary dataset."""
    ann_file = wider_root / "wider_face_split" / f"wider_face_{split}_bbx_gt.txt"
    img_dir = wider_root / f"WIDER_{split}" / "images"
    annotations = _read_wider_annotations(str(ann_file))

    out_pos = out_dir / split / "face"; _ensure_dir(out_pos)
    out_neg = out_dir / split / "non_face"; _ensure_dir(out_neg)
    
    pos_count = 0
    for img_path_str, boxes in tqdm(annotations, desc=f"[{split}] Building face dataset"):
        img_path = img_dir / img_path_str
        if not img_path.exists(): continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception: continue
        
        for i, (x, y, w, h) in enumerate(boxes):
            if w < kwargs.get('min_face', 20) or h < kwargs.get('min_face', 20): continue
            
            # Save positive crop
            box = (x, y, x + w, y + h)
            out_pos_path = out_pos / f"{img_path.stem}_{i}.jpg"
            if _save_crop(img, box, out_pos_path):
                pos_count += 1
    
    # Generate balanced negatives
    print(f"[{split}] Found {pos_count} positive face samples. Generating balanced negatives...")
    neg_count = 0
    pbar_neg = tqdm(total=pos_count, desc=f"[{split}] Sampling non-face negatives")
    while neg_count < pos_count:
        neg_src_path = random.choice(neg_pool)
        try:
            neg_img = Image.open(neg_src_path).convert("RGB")
            nW, nH = neg_img.size
            # Use a random size for more diversity
            tw, th = random.randint(32, 256), random.randint(32, 256)
            if nW <= tw or nH <= th: continue
            
            nx0, ny0 = random.randint(0, nW - tw), random.randint(0, nH - th)
            box = (nx0, ny0, nx0 + tw, ny0 + th)
            out_neg_path = out_neg / f"{neg_src_path.stem}_{random.randint(0,99999)}.jpg"
            if _save_crop(neg_img, box, out_neg_path):
                neg_count += 1
                pbar_neg.update(1)
        except Exception: continue
    pbar_neg.close()
    print(f"✅ [{split}] split built: {pos_count} faces, {neg_count} non-faces.")

def build_face_binary_dataset(wider_root: str, neg_root: str, out_dir: str, **kwargs):
    """Builds a FACE vs NON_FACE dataset from WIDER Face and a negative source."""
    wider_root, neg_root, out_dir = Path(wider_root), Path(neg_root), Path(out_dir)
    print(f"Building FACE/NON_FACE dataset at: {out_dir}")
    random.seed(kwargs.get('seed', 42))
    
    print("Collecting negative image pool...")
    neg_pool = list(neg_root.glob("**/*.jpg")) + list(neg_root.glob("**/*.png"))
    if not neg_pool: raise FileNotFoundError(f"No negative images (.jpg, .png) found in {neg_root}")

    for split in ['train', 'val']:
        _build_face_binary_split(split, wider_root, neg_pool, out_dir, **kwargs)

# --- 3. Person Binary Dataset Builder (from COCO) -----------------------------

def _build_person_binary_split(split: str, coco_root: Path, out_dir: Path, **kwargs):
    """Builds one split (train/val) for the person binary dataset."""
    random.seed(kwargs.get('seed', 42))
    ann_file = coco_root / "annotations" / f"instances_{split}2017.json"
    img_dir = coco_root / f"{split}2017"
    coco = COCO(str(ann_file))
    person_cat_id = coco.getCatIds(catNms=['person'])[0]
    
    img_ids = coco.getImgIds()
    person_img_ids = coco.getImgIds(catIds=[person_cat_id])
    non_person_img_ids = list(set(img_ids) - set(person_img_ids))
    random.shuffle(non_person_img_ids)

    out_pos = out_dir / split / "person"; _ensure_dir(out_pos)
    out_neg = out_dir / split / "non_person"; _ensure_dir(out_neg)

    pos_count = 0
    for img_id in tqdm(person_img_ids, desc=f"[{split}] Cropping person positives"):
        ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=[person_cat_id], iscrowd=False)
        if not ann_ids: continue
        
        img_info = coco.loadImgs([img_id])[0]
        img_path = img_dir / img_info['file_name']
        if not img_path.exists(): continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception: continue

        anns = coco.loadAnns(ann_ids)
        for i, ann in enumerate(anns):
            x, y, w, h = ann['bbox']
            if w < kwargs.get('min_person_w', 32) or h < kwargs.get('min_person_h', 32): continue
            
            box = (int(x), int(y), int(x + w), int(y + h))
            out_pos_path = out_pos / f"{img_path.stem}_{i}.jpg"
            if _save_crop(img, box, out_pos_path):
                pos_count += 1

    # Generate balanced negatives from non-person images
    neg_count = 0
    pbar_neg = tqdm(total=pos_count, desc=f"[{split}] Sampling non-person negatives")
    img_ptr = 0
    while neg_count < pos_count and img_ptr < len(non_person_img_ids):
        img_id = non_person_img_ids[img_ptr]; img_ptr += 1
        img_info = coco.loadImgs([img_id])[0]
        img_path = img_dir / img_info['file_name']
        if not img_path.exists(): continue
        
        # We can just copy the whole image as a negative sample
        out_neg_path = out_neg / img_path.name
        try:
            shutil.copy(str(img_path), str(out_neg_path))
            neg_count += 1
            pbar_neg.update(1)
        except Exception:
            continue
    pbar_neg.close()
    print(f"✅ [{split}] split built: {pos_count} persons, {neg_count} non-persons.")


def build_person_binary_dataset(coco_root: str, out_dir: str, **kwargs):
    """Builds a PERSON vs NON_PERSON binary dataset from COCO."""
    if COCO is None:
        raise ImportError("Please install pycocotools (`pip install pycocotools`) to build datasets from COCO.")
    
    print(f"Building PERSON/NON_PERSON dataset at: {out_dir}")
    for split in ['train', 'val']:
        _build_person_binary_split(split, Path(coco_root), Path(out_dir), **kwargs)