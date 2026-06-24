import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import numpy as np
import sys
from sklearn.metrics import f1_score
from collections import Counter

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from models.damage_model import create_damage_model
from data_processing.damage_data import get_damage_data_loaders


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_probs = nn.functional.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        target_probs = probs[torch.arange(probs.shape[0]), targets]
        focal_weight = (1 - target_probs) ** self.gamma

        if self.alpha is not None:
            alpha_weight = self.alpha[targets]
            focal_weight = focal_weight * alpha_weight

        loss = -focal_weight * log_probs[torch.arange(probs.shape[0]), targets]

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def calculate_f1_score(outputs, targets):
    with torch.no_grad():
        pred_classes = torch.argmax(outputs, dim=1).cpu().numpy()
        target_classes = targets.cpu().numpy()
        f1_macro = f1_score(target_classes, pred_classes, average='macro', zero_division=0)
        f1_weighted = f1_score(target_classes, pred_classes, average='weighted', zero_division=0)
        return f1_macro, f1_weighted


def calculate_class_weights(dataset):
    class_counts = torch.zeros(4)
    for label in dataset.labels:
        class_counts[label] += 1

    class_counts = torch.clamp(class_counts, min=1.0)
    weights = 1.0 / torch.log1p(class_counts)
    weights = weights / weights.min()

    print("Class distribution and weights:")
    class_names = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']
    for i, (name, count, weight) in enumerate(zip(class_names, class_counts, weights)):
        print(f"  {name}: {int(count)} samples (weight: {weight:.3f})")

    return weights


def run_optimized_training():
    print('OPTIMIZED DAMAGE CLASSIFICATION TRAINING')
    print('=' * 60)

    batch_size = 32
    learning_rate = 1e-3
    epochs = 40
    early_stop_patience = 3
    patch_size = 64
    weight_decay = 5e-4
    dropout_rate = 0.4

    print(f'Batch size: {batch_size}')
    print(f'Learning rate: {learning_rate:.0e}')
    print(f'Epochs: {epochs}')
    print(f'Early stopping patience: {early_stop_patience}')
    print(f'Patch size: {patch_size}x{patch_size}')
    print(f'Weight decay: {weight_decay}')
    print(f'Dropout rate: {dropout_rate}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    torch.cuda.empty_cache()

    data_dir = os.path.join(project_root, 'Data')
    train_loader, val_loader = get_damage_data_loaders(
        data_dir,
        batch_size=batch_size,
        patch_size=patch_size,
        num_workers=0,  
        prefetch_factor=2
    )

    if train_loader is None:
        print('Failed to create data loaders!')
        return

    print(f'Training samples: {len(train_loader.dataset)}')
    print(f'Validation samples: {len(val_loader.dataset)}')

    print('\nCalculating class weights...')
    class_weights = calculate_class_weights(train_loader.dataset)
    class_weights = class_weights.to(device)

    model = create_damage_model(dropout_rate=dropout_rate).to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        eps=1e-8,
        betas=(0.9, 0.999)
    )

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[10, 20, 30],
        gamma=0.5
    )

    best_f1_weighted = 0.0
    best_f1_macro = 0.0
    epochs_without_improvement = 0
    best_combined_score = 0.0

    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}/{epochs}')
        print(f'Learning rate: {optimizer.param_groups[0]["lr"]:.6f}')

        model.train()
        train_loss = 0.0
        train_f1_macro = 0.0
        train_f1_weighted = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc='Training')
        for batch_idx, (patches, labels) in enumerate(progress_bar):
            patches = patches.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(patches)
            loss = criterion(outputs, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss at batch {batch_idx}")
                continue

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            if torch.isnan(grad_norm) or grad_norm > 50.0:
                print(f"Warning: Large gradient norm {grad_norm:.2f} at batch {batch_idx}")
                continue

            optimizer.step()

            train_loss += loss.item()
            f1_macro, f1_weighted = calculate_f1_score(outputs, labels)
            train_f1_macro += f1_macro
            train_f1_weighted += f1_weighted
            num_batches += 1

            if batch_idx % 50 == 0:
                progress_bar.set_postfix({
                    'Loss': f'{train_loss/num_batches:.4f}',
                    'F1-M': f'{train_f1_macro/num_batches:.3f}',
                    'F1-W': f'{train_f1_weighted/num_batches:.3f}'
                })

        if num_batches == 0:
            print(f"ERROR: No successful batches in epoch {epoch+1}")
            break

        avg_train_loss = train_loss / num_batches
        avg_train_f1_macro = train_f1_macro / num_batches
        avg_train_f1_weighted = train_f1_weighted / num_batches

        model.eval()
        val_loss = 0.0
        val_f1_macro = 0.0
        val_f1_weighted = 0.0
        val_batches = 0

        with torch.no_grad():
            for patches, labels in tqdm(val_loader, desc='Validation'):
                patches = patches.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(patches)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                f1_macro, f1_weighted = calculate_f1_score(outputs, labels)
                val_f1_macro += f1_macro
                val_f1_weighted += f1_weighted
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        avg_val_f1_macro = val_f1_macro / val_batches
        avg_val_f1_weighted = val_f1_weighted / val_batches

        print(f'Train Loss: {avg_train_loss:.4f} | F1-Macro: {avg_train_f1_macro:.3f} | F1-Weighted: {avg_train_f1_weighted:.3f}')
        print(f'Val   Loss: {avg_val_loss:.4f} | F1-Macro: {avg_val_f1_macro:.3f} | F1-Weighted: {avg_val_f1_weighted:.3f}')

        combined_score = 0.6 * avg_val_f1_weighted + 0.4 * avg_val_f1_macro
        print(f'Combined Score: {combined_score:.3f}')

        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_f1_weighted = avg_val_f1_weighted
            best_f1_macro = avg_val_f1_macro
            epochs_without_improvement = 0

            weights_dir = os.path.join(project_root, 'weights')
            os.makedirs(weights_dir, exist_ok=True)
            model_save_path = os.path.join(weights_dir, 'best_damage_model.pth')

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1_weighted': best_f1_weighted,
                'best_f1_macro': best_f1_macro,
                'best_combined_score': best_combined_score,
                'patch_size': patch_size,
                'class_weights': class_weights.cpu()
            }, model_save_path)

            print(f'Best model saved! Combined: {best_combined_score:.3f} | F1-W: {best_f1_weighted:.3f} | F1-M: {best_f1_macro:.3f}')
        else:
            epochs_without_improvement += 1
            print(f'No improvement: {epochs_without_improvement}/{early_stop_patience}')

            if epochs_without_improvement >= early_stop_patience:
                print(f'\nEarly stopping triggered!')
                break

        scheduler.step()

    print(f'\nTraining complete!')
    print(f'Best Combined Score: {best_combined_score:.3f}')
    print(f'Best F1-Weighted: {best_f1_weighted:.3f} ({best_f1_weighted*100:.1f}%)')
    print(f'Best F1-Macro: {best_f1_macro:.3f} ({best_f1_macro*100:.1f}%)')


if __name__ == '__main__':
    run_optimized_training()