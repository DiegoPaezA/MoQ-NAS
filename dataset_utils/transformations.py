# dataset_utils/transformations.py
import torch
from torchvision.transforms import (
    Compose, ToTensor, Normalize, Resize,
    TrivialAugmentWide, RandomResizedCrop, RandomHorizontalFlip, CenterCrop, RandomCrop
)

# Train-time augmentation policies for generic (CIFAR-like) datasets:
#   'ta'              TrivialAugmentWide only (F13 protocol, default).
#   'standard'        RandomCrop(pad 4) + HorizontalFlip + TrivialAugmentWide.
#   'standard_cutout' 'standard' + Cutout (1 hole, 16 px) on the normalised tensor.
AUGMENTATION_POLICIES = ('ta', 'standard', 'standard_cutout')


class Cutout:
    """Cutout (DeVries & Taylor, 2017): zero one square patch of a CHW tensor.

    The patch centre is uniform over the image and the patch is clipped at the
    borders, as in the reference implementation. Uses the torch RNG, so it is
    reproducible under the DataLoader worker seeding.
    """
    def __init__(self, length: int = 16, n_holes: int = 1):
        self.length = int(length)
        self.n_holes = int(n_holes)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        _, h, w = img.shape
        mask = torch.ones((h, w), dtype=img.dtype)
        for _ in range(self.n_holes):
            y = int(torch.randint(h, (1,)))
            x = int(torch.randint(w, (1,)))
            y1, y2 = max(0, y - self.length // 2), min(h, y + self.length // 2)
            x1, x2 = max(0, x - self.length // 2), min(w, x + self.length // 2)
            mask[y1:y2, x1:x2] = 0
        return img * mask.unsqueeze(0)

    def __repr__(self):
        return f"Cutout(length={self.length}, n_holes={self.n_holes})"


def build_transforms(spec, data_augmentation: bool, augmentation_policy: str = 'ta'):
    """
    spec: DatasetSpec or object with .name .shape .mean .std
    augmentation_policy: one of AUGMENTATION_POLICIES; only used for generic
        (CIFAR-like) datasets. ATLETA, person/face and MedMNIST behaviour is unchanged.
    Return: (train_transform, eval_transform)
    """
    if augmentation_policy not in AUGMENTATION_POLICIES:
        raise ValueError(f"augmentation_policy must be one of {AUGMENTATION_POLICIES}, "
                         f"got {augmentation_policy!r}")
    ds_name_check = (spec.name or "").lower()
    if augmentation_policy != 'ta' and any(k in ds_name_check for k in ('mnist', 'atleta', 'person', 'face')):
        raise ValueError(f"augmentation_policy={augmentation_policy!r} is only defined for CIFAR-like datasets; "
                         f"'{spec.name}' keeps its own augmentation (use 'ta').")
    ds_name = (spec.name or "").lower()

    # --- Heads (PIL-space ops) ---
    train_head, eval_head = [], []

    if "atleta" in ds_name:
        # spec.shape expected as (C, H, W)
        _, h, w = spec.shape
        resize = Resize((h, w))
        train_head.append(resize)
        eval_head.append(resize)

        if data_augmentation:
            train_head.append(TrivialAugmentWide(num_magnitude_bins=31))

    elif "person" in ds_name or "face" in ds_name:
        t = getattr(spec, "transform", None)
        img_size = int(t.get("img_size", spec.shape[1])) if isinstance(t, dict) else int(spec.shape[1])

        # Eval (and non-aug train) use standard resize + center crop
        eval_head.extend([Resize(int(img_size * 256 / 224)), CenterCrop(img_size)])

        if data_augmentation:
            # Stronger train-time aug for face/person
            train_head.extend([
                RandomResizedCrop(img_size, scale=(0.08, 1.0)),
                RandomHorizontalFlip(),
                TrivialAugmentWide(num_magnitude_bins=31),
            ])
        else:
            # Mirror eval geometry if no aug
            train_head.extend(eval_head)

    else:
        # Generic datasets: optionally add light augmentation to train
        if data_augmentation:
            if augmentation_policy in ('standard', 'standard_cutout'):
                _, h, w = spec.shape
                train_head.extend([RandomCrop((h, w), padding=4), RandomHorizontalFlip()])
            train_head.append(TrivialAugmentWide(num_magnitude_bins=31))
        # eval_head stays empty (identity) unless you add dataset-specific geometry

    # --- Shared tail (tensor-space ops) ---
    tail = [ToTensor()]
    if getattr(spec, "mean", None) is not None and getattr(spec, "std", None) is not None:
        tail.append(Normalize(mean=spec.mean, std=spec.std))

    train_tail = list(tail)
    if data_augmentation and augmentation_policy == 'standard_cutout':
        train_tail.append(Cutout(length=16, n_holes=1))

    train_tf = Compose([*train_head, *train_tail])
    eval_tf  = Compose([*eval_head,  *tail])
    return train_tf, eval_tf