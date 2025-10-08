# moq-nas/scripts/fairness_baseline/train.py
import csv
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import random
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# --- Step 1: Import the shared components from your core library ---
from core.fairness.data import create_binary_loaders, get_default_transforms
from core.fairness.models import make_baseline_model, REGISTRY

def set_seed(seed: int = 42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_one_model(
    arch: str,
    data_root: str,
    device: torch.device,
    args: argparse.Namespace
):
    """
    This function contains the logic to train a single model architecture.
    It's called in a loop by the main function.
    """
    print(f"\n--- Starting Training for [{arch}] ---")
    
    # --- Get components from core modules ---
    transforms = get_default_transforms(img_size=args.img_size)
    train_loader, val_loader = create_binary_loaders(
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        tf_train=transforms['train'],
        tf_val=transforms['val']
    )
    model = make_baseline_model(arch, num_classes=2).to(device)
    
    # --- Logic to handle freezing the backbone ---
    if args.freeze_backbone:
        print(f"[{arch}] Freezing backbone and training only the head.")
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze the final layer (head)
        if hasattr(model, 'fc'): # For ResNets
            for param in model.fc.parameters():
                param.requires_grad = True
        elif hasattr(model, 'classifier'): # For others like MobileNet, EffNet, ConvNeXt
            for param in model.classifier.parameters():
                param.requires_grad = True
    
    # The optimizer will only receive parameters that require gradients
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.wd)

    criterion = nn.CrossEntropyLoss()
    
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{Path(data_root).name}_{arch}.pt"
    best_val_acc = 0.0

    # --- Training Loop ---
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [{arch} Train]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f"Epoch {epoch} [{arch}]: Train Loss: {train_loss/train_total:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  -> New best val acc: {best_val_acc:.2f}%. Saving model to {checkpoint_path}")
            torch.save(model.state_dict(), checkpoint_path)

    print(f"--- Finished Training for [{arch}]. Best model saved to {checkpoint_path} ---")
    if args.results_csv:
        results_path = Path(args.results_csv)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        # Check if file exists to write header only once
        write_header = not results_path.exists()
        
        with open(results_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['arch', 'dataset', 'best_val_acc', 'checkpoint_path'])
            
            writer.writerow([
                arch,
                Path(data_root).name,
                f"{best_val_acc:.4f}",
                str(checkpoint_path.resolve())
            ])
        print(f"Saved best accuracy result to {results_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Train multiple baseline models for fairness evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- Command-Line Arguments ---
    parser.add_argument('--data_root', required=True, help="Root with train/ and val/ subfolders")
    parser.add_argument('--archs', type=str,
                        default="resnet18,resnet50,efficientnet_v2_s",
                        help="Comma-separated torchvision archs to train.")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-4, help="Weight decay")
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default="checkpoints/baselines")
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default=None, help="e.g., 'cuda:0'. Default picks best available.")
    parser.add_argument('--freeze_backbone', action='store_true',
                        help="If set, train only the final classification head.")
    
    parser.add_argument('--results_csv', type=str, default="checkpoints/baselines/baseline_results.csv",
                        help="Path to CSV file to save the best validation accuracy for each model.")
    args = parser.parse_args()

    set_seed(args.seed)
    
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Logic to handle multiple architectures ---
    requested_archs = [a.strip() for a in args.archs.split(',') if a.strip()]
    
    # Check which of the requested models are available in your models.py registry
    available_archs = [arch for arch in requested_archs if arch in REGISTRY]
    unavailable = set(requested_archs) - set(available_archs)
    if unavailable:
        print(f"[Warning] Skipping unavailable architectures: {sorted(list(unavailable))}")

    if not available_archs:
        raise SystemExit("No requested architectures are available in core/fairness/models.py.")

    print(f"Device: {device}")
    print(f"Architectures to train: {available_archs}")

    # --- Loop through and train each model ---
    for arch in available_archs:
        train_one_model(
            arch=arch,
            data_root=args.data_root,
            device=device,
            args=args
        )

if __name__ == "__main__":
    main()