import torch
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import sys
from skimage import measure

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from models.model import create_model
from models.damage_model import create_damage_model



CLASS_NAMES = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']
COLORS_BGR = [
    (0, 255, 0),      
    (0, 255, 255),    
    (0, 165, 255),    
    (0, 0, 255),      
]
COLORS_RGB = [
    (0, 255, 0),
    (255, 255, 0),
    (255, 165, 0),
    (255, 0, 0),
]



def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    loc_model = create_model().to(device)
    loc_path = os.path.join(project_root, 'weights', 'best_model_resnet34_bce_dice_large.pth')
    if not os.path.exists(loc_path):
        print(f'ERROR: Localization model not found: {loc_path}')
        return None, None, device

    checkpoint = torch.load(loc_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        loc_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        loc_model.load_state_dict(checkpoint)
    loc_model.eval()
    print(f'SUCCESS: Localization model loaded from {loc_path}')

    damage_model = create_damage_model().to(device)
    damage_path = os.path.join(project_root, 'weights', 'best_damage_model.pth')
    if not os.path.exists(damage_path):
        print(f'ERROR: Damage model not found: {damage_path}')
        return None, None, device

    damage_checkpoint = torch.load(damage_path, map_location=device)
    damage_model.load_state_dict(damage_checkpoint['model_state_dict'])
    damage_model.eval()
    print(f'SUCCESS: Damage model loaded from {damage_path}')

    return loc_model, damage_model, device



def detect_buildings(loc_model, device, image):
    tensor = torch.from_numpy(
        image.transpose(2, 0, 1)
    ).float().unsqueeze(0).to(device) / 255.0

    with torch.no_grad():
        output = loc_model(tensor)
        prediction = torch.sigmoid(output).cpu().numpy()[0, 0]

    binary = (prediction > 0.5).astype(np.uint8)
    labeled = measure.label(binary)
    regions = measure.regionprops(labeled)

    buildings = []
    for region in regions:
        if region.area > 100:
            minr, minc, maxr, maxc = region.bbox
            buildings.append({
                'bbox': (minr, minc, maxr, maxc),
                'mask': labeled == region.label
            })

    return buildings


def predict_damage(damage_model, device, image, bbox):
    minr, minc, maxr, maxc = bbox
    patch = image[minr:maxr, minc:maxc]

    if patch.shape[0] == 0 or patch.shape[1] == 0:
        return 0, 0.0

    patch_resized = cv2.resize(patch, (64, 64))
    tensor = torch.from_numpy(
        patch_resized.astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        output = damage_model(tensor)
        _, predicted = torch.max(output, 1)
        confidence = torch.softmax(output, dim=1)[0][predicted].item()

    return predicted.item(), confidence


def create_visualization(image, buildings, pred_damages, output_path, patch_name):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image)
    axes[0].set_title('Imagen satelital original', fontsize=13, fontweight='bold')
    axes[0].axis('off')

    mask_img = np.zeros_like(image)
    for building in buildings:
        mask_img[building['mask']] = (200, 200, 200)
    axes[1].imshow(image)
    axes[1].imshow(mask_img, alpha=0.5)
    axes[1].set_title(f'Edificios detectados: {len(buildings)}', fontsize=13, fontweight='bold')
    axes[1].axis('off')

    pred_img = image.copy()
    for building, damage in zip(buildings, pred_damages):
        color = COLORS_RGB[damage]
        pred_img[building['mask']] = color
    axes[2].imshow(pred_img)
    axes[2].set_title('Clasificación de daño', fontsize=13, fontweight='bold')
    axes[2].axis('off')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='No damage'),
        Patch(facecolor='yellow', label='Minor damage'),
        Patch(facecolor='orange', label='Major damage'),
        Patch(facecolor='red', label='Destroyed'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=11)
    fig.suptitle(f'Parche: {patch_name}', fontsize=11, color='gray')

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  SUCCESS: Saved {output_path}')



def run_satellite_test():
    print('PRIMER ENFOQUE — INFERENCIA SOBRE IMÁGENES SATELITALES REALES')
    print('ResNet34 U-Net (localización) + DamageCNN (clasificación)')
    print('=' * 60)

    loc_model, damage_model, device = load_models()
    if loc_model is None or damage_model is None:
        return

    patches_dir = os.path.join(project_root, 'parches_satelite')
    if not os.path.exists(patches_dir):
        print(f'ERROR: Patches directory not found: {patches_dir}')
        return

    patch_files = sorted([
        f for f in os.listdir(patches_dir)
        if f.endswith('.png') or f.endswith('.jpg')
    ])

    print(f'Parches encontrados: {len(patch_files)}')

    output_dir = os.path.join(project_root, 'test_results', 'satelite_primer_enfoque')
    os.makedirs(output_dir, exist_ok=True)

    total_buildings = 0
    damage_counts = [0, 0, 0, 0]

    for i, patch_file in enumerate(patch_files, 1):
        print(f'\nProcessing {i}/{len(patch_files)}: {patch_file}')

        img_path = os.path.join(patches_dir, patch_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f'  ERROR: Could not load {patch_file}')
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        buildings = detect_buildings(loc_model, device, image)
        print(f'  Buildings detected: {len(buildings)}')

        if len(buildings) == 0:
            print(f'  WARNING: No buildings detected')
            continue

        pred_damages = []
        for building in buildings:
            damage, confidence = predict_damage(damage_model, device, image, building['bbox'])
            pred_damages.append(damage)
            damage_counts[damage] += 1

        total_buildings += len(buildings)

        output_path = os.path.join(output_dir, f'satelite_{i:03d}.png')
        create_visualization(image, buildings, pred_damages, output_path, patch_file)

    print('\n' + '=' * 60)
    print('RESUMEN')
    print('=' * 60)
    print(f'Parches procesados : {len(patch_files)}')
    print(f'Edificios detectados: {total_buildings}')
    print(f'\nDistribución de daño predicho:')
    for i, name in enumerate(CLASS_NAMES):
        pct = damage_counts[i] / total_buildings * 100 if total_buildings > 0 else 0
        print(f'  {name}: {damage_counts[i]} ({pct:.1f}%)')
    print(f'\nResultados guardados en: {output_dir}')
    print('TEST COMPLETE')


if __name__ == '__main__':
    run_satellite_test()