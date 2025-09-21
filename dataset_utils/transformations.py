# dataset_utils/transformations.py
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

def build_transforms(spec, data_augmentation: bool):
    """
    spec: DatasetSpec or a simple object with .name .shape .mean .std
    Return: train_transform, eval_transform
    """
    base = []
    # Example resize for custom datasets (ATLETA variants)
    if "atleta" in spec.name.lower():
        _, h, w = spec.shape
        base.append(Resize((h, w)))
    base.append(ToTensor())
    if spec.mean is not None and spec.std is not None:
        base.append(Normalize(mean=spec.mean, std=spec.std))
    eval_tf = Compose(base)

    if data_augmentation:
        from torchvision.transforms import TrivialAugmentWide
        train_tf = Compose([*base[:1], TrivialAugmentWide(num_magnitude_bins=31), *base[1:]])
    else:
        train_tf = eval_tf
    return train_tf, eval_tf
