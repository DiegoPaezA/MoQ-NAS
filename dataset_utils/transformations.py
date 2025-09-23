# dataset_utils/transformations.py
from torchvision.transforms import TrivialAugmentWide
from torchvision.transforms import Compose, ToTensor, Normalize, Resize


def build_transforms(spec, data_augmentation: bool):
    """
    spec: DatasetSpec or a simple object with .name .shape .mean .std
    Return: train_transform, eval_transform
    """
    # Start with transforms that expect a PIL image
    train_transforms_list = []
    eval_transforms_list = []

    # 1. Handle resizing first for both pipelines
    if "atleta" in spec.name.lower():
        _, h, w = spec.shape
        resize_transform = Resize((h, w))
        train_transforms_list.append(resize_transform)
        eval_transforms_list.append(resize_transform)

    # 2. Add data augmentation ONLY to the training pipeline
    if data_augmentation:
        # TrivialAugmentWide operates on PIL images
        train_transforms_list.append(TrivialAugmentWide(num_magnitude_bins=31))

    # 3. Add ToTensor and Normalize to BOTH pipelines at the end
    train_transforms_list.append(ToTensor())
    eval_transforms_list.append(ToTensor())

    if spec.mean is not None and spec.std is not None:
        normalize_transform = Normalize(mean=spec.mean, std=spec.std)
        train_transforms_list.append(normalize_transform)
        eval_transforms_list.append(normalize_transform)

    # 4. Compose the final pipelines
    train_tf = Compose(train_transforms_list)
    eval_tf = Compose(eval_transforms_list)

    return train_tf, eval_tf