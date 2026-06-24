import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from models.model_multiclass import create_model, calculate_metrics
from data_processing.multiclass_data import get_multiclass_data_loaders


def get_class_weights(device):
    building_pixels = {
        'no-damage':    95259960,
        'minor-damage': 11200180,
        'major-damage': 13845274,
        'destroyed':    6570531,
    }

    total_building = sum(building_pixels.values())

    weights = torch.ones(5)  

    print('Class weights:')
    print(f'  background: 1.000 (fixed)')
    for i, (name, count) in enumerate(building_pixels.items(), start=1):
        freq = count / total_building
        weights[i] = 1.0 / freq
        print(f'  {name}: {weights[i]:.3f} (freq: {freq:.3f})')

    weights[1:] = weights[1:] / weights[1]
    print('\nNormalized weights:')
    class_names = ['background', 'no-damage', 'minor-damage', 'major-damage', 'destroyed']
    for i, name in enumerate(class_names):
        print(f'  {name}: {weights[i]:.3f}')

    return weights.to(device)


def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    torch.cuda.empty_cache()

    data_dir = os.path.join(project_root, '..', 'xView2 - copia - SAM', 'Data')
    data_dir = os.path.abspath(data_dir)
    if not os.path.exists(data_dir):
        data_dir = os.path.join(project_root, 'Data')

    if not os.path.exists(data_dir):
        print(f'Data directory not found: {data_dir}')
        return

    print(f'Data directory: {data_dir}')

    batch_size = 2
    epochs = 25
    learning_rate = 1e-4

    print(f'Batch size: {batch_size}')
    print(f'Epochs: {epochs}')
    print(f'Learning rate: {learning_rate}')

    train_loader, val_loader = get_multiclass_data_loaders(data_dir, batch_size=batch_size)

    if train_loader is None:
        print('Failed to create data loaders!')
        return

    print(f'Training samples: {len(train_loader.dataset)}')
    print(f'Validation samples: {len(val_loader.dataset)}')

    model = create_model().to(device)

    class_weights = get_class_weights(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5
    )

    best_mean_f1 = 0.0
    weights_dir = os.path.join(project_root, 'weights')
    os.makedirs(weights_dir, exist_ok=True)

    class_names = ['background', 'no-damage', 'minor-damage', 'major-damage', 'destroyed']

    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}/{epochs}')

        model.train()
        train_loss = 0.0
        train_iou = 0.0
        train_f1 = 0.0

        for batch_idx, (images, masks) in enumerate(tqdm(train_loader)):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)

            if torch.isnan(loss):
                print(f'NaN loss at batch {batch_idx}')
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            iou, f1, _, _ = calculate_metrics(outputs, masks)
            train_iou += iou
            train_f1 += f1

            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_iou = train_iou / len(train_loader)
        avg_train_f1 = train_f1 / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_f1 = 0.0
        val_f1_per_class = np.zeros(5)

        with torch.no_grad():
            for images, masks in tqdm(val_loader):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, masks)

                val_loss += loss.item()
                iou, f1, _, f1_cls = calculate_metrics(outputs, masks)
                val_iou += iou
                val_f1 += f1
                val_f1_per_class += np.array(f1_cls)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        avg_val_f1 = val_f1 / len(val_loader)
        avg_val_f1_per_class = val_f1_per_class / len(val_loader)

        print(f'Train Loss: {avg_train_loss:.4f} | Train IoU: {avg_train_iou:.4f} | Train F1: {avg_train_f1:.4f}')
        print(f'Val   Loss: {avg_val_loss:.4f} | Val   IoU: {avg_val_iou:.4f} | Val   F1: {avg_val_f1:.4f}')
        print('Val F1 per class:')
        for i, (name, f1_c) in enumerate(zip(class_names, avg_val_f1_per_class)):
            print(f'  {name}: {f1_c:.4f}')

        scheduler.step(avg_val_f1)

        if avg_val_f1 > best_mean_f1:
            best_mean_f1 = avg_val_f1
            model_path = os.path.join(weights_dir, 'best_model_multiclass.pth')
            torch.save(model.state_dict(), model_path)
            print(f'Best F1: {best_mean_f1:.4f} - Model saved!')

        max_mem = torch.cuda.max_memory_allocated(0) // 1024**2
        print(f'Max GPU Memory: {max_mem}MB')
        torch.cuda.empty_cache()

    print(f'\nTraining complete! Best Mean F1: {best_mean_f1:.4f}')


if __name__ == '__main__':
    train_model()