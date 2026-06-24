import torch
import numpy as np
import os
import cv2
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from skimage import measure

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from models.model import create_model
from models.damage_model import create_damage_model
from shapely.wkt import loads


def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loc_model = create_model().to(device)
    loc_checkpoint_path = os.path.join(project_root, 'weights', 'best_model_resnet34_bce_dice_large.pth')

    if not os.path.exists(loc_checkpoint_path):
        print(f'ERROR: Localization model not found: {loc_checkpoint_path}')
        return None, None, device

    loc_checkpoint = torch.load(loc_checkpoint_path, map_location=device)
    if isinstance(loc_checkpoint, dict) and 'model_state_dict' in loc_checkpoint:
        loc_model.load_state_dict(loc_checkpoint['model_state_dict'])
    else:
        loc_model.load_state_dict(loc_checkpoint)
    loc_model.eval()
    print(f'SUCCESS: Localization model loaded from {loc_checkpoint_path}')

    damage_model = create_damage_model().to(device)
    damage_checkpoint_path = os.path.join(project_root, 'weights', 'best_damage_model.pth')

    if not os.path.exists(damage_checkpoint_path):
        print(f'ERROR: Damage model not found: {damage_checkpoint_path}')
        return None, None, device

    damage_checkpoint = torch.load(damage_checkpoint_path, map_location=device)
    damage_model.load_state_dict(damage_checkpoint['model_state_dict'])
    damage_model.eval()
    print(f'SUCCESS: Damage model loaded from {damage_checkpoint_path}')

    return loc_model, damage_model, device



def load_test_images(image_id):
    pre_path = os.path.join(project_root, f'Data/test/images/{image_id}_pre_disaster.png')
    post_path = os.path.join(project_root, f'Data/test/images/{image_id}_post_disaster.png')
    labels_path = os.path.join(project_root, f'Data/test/labels/{image_id}_post_disaster.json')

    if not os.path.exists(pre_path) or not os.path.exists(post_path) or not os.path.exists(labels_path):
        return None, None, None

    pre_image = cv2.imread(pre_path)
    pre_image = cv2.cvtColor(pre_image, cv2.COLOR_BGR2RGB)

    post_image = cv2.imread(post_path)
    post_image = cv2.cvtColor(post_image, cv2.COLOR_BGR2RGB)

    with open(labels_path, 'r') as f:
        labels_data = json.load(f)

    return pre_image, post_image, labels_data


def get_ground_truth_buildings(labels_data, image_shape):
    ground_truth_buildings = []
    damage_types = {'no-damage': 0, 'minor-damage': 1, 'major-damage': 2, 'destroyed': 3}

    if 'features' in labels_data and 'xy' in labels_data['features']:
        for feature in labels_data['features']['xy']:
            if feature['properties']['feature_type'] == 'building':
                wkt_str = feature.get('wkt')
                damage_type = feature['properties'].get('subtype', 'no-damage')

                if wkt_str:
                    try:
                        geometry = loads(wkt_str)
                        coords = list(geometry.exterior.coords)
                        pts = np.array([[int(x), int(y)] for x, y in coords])

                        mask = np.zeros(image_shape[:2], dtype=np.uint8)
                        cv2.fillPoly(mask, [pts], 1)
                        mask = mask.astype(bool)

                        x_coords = pts[:, 0]
                        y_coords = pts[:, 1]
                        minc, maxc = int(np.min(x_coords)), int(np.max(x_coords))
                        minr, maxr = int(np.min(y_coords)), int(np.max(y_coords))

                        ground_truth_buildings.append({
                            'bbox': (minr, minc, maxr, maxc),
                            'mask': mask,
                            'damage_type': damage_types.get(damage_type, 0)
                        })
                    except:
                        continue

    return ground_truth_buildings



def detect_buildings(loc_model, device, pre_image):
    image_tensor = torch.from_numpy(
        pre_image.transpose(2, 0, 1)
    ).float().unsqueeze(0).to(device) / 255.0

    with torch.no_grad():
        output = loc_model(image_tensor)
        prediction = torch.sigmoid(output).cpu().numpy()[0, 0]

    binary_prediction = (prediction > 0.5).astype(np.uint8)
    labeled_regions = measure.label(binary_prediction)
    regions = measure.regionprops(labeled_regions)

    building_regions = []
    for region in regions:
        if region.area > 100:
            minr, minc, maxr, maxc = region.bbox
            building_regions.append({
                'bbox': (minr, minc, maxr, maxc),
                'mask': labeled_regions == region.label
            })

    return building_regions



def predict_damage(damage_model, device, post_image, bbox):
    minr, minc, maxr, maxc = bbox
    patch = post_image[minr:maxr, minc:maxc]

    if patch.shape[0] == 0 or patch.shape[1] == 0:
        return 0, 0.0

    patch_resized = cv2.resize(patch, (64, 64))
    patch_tensor = torch.from_numpy(
        patch_resized.astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        output = damage_model(patch_tensor)
        _, predicted = torch.max(output, 1)
        confidence = torch.softmax(output, dim=1)[0][predicted].item()

    return predicted.item(), confidence


def get_gt_damage_for_bbox(labels_data, bbox):
    damage_types = {'no-damage': 0, 'minor-damage': 1, 'major-damage': 2, 'destroyed': 3}
    minr, minc, maxr, maxc = bbox

    if 'features' in labels_data and 'xy' in labels_data['features']:
        for feature in labels_data['features']['xy']:
            if feature['properties']['feature_type'] == 'building':
                wkt_str = feature.get('wkt')
                damage_type = feature['properties'].get('subtype', 'no-damage')

                if wkt_str:
                    try:
                        geometry = loads(wkt_str)
                        gminx, gminy, gmaxx, gmaxy = geometry.bounds
                        if not (maxc < gminx or minc > gmaxx or maxr < gminy or minr > gmaxy):
                            return damage_types.get(damage_type, 0)
                    except:
                        continue

    return 0



def create_visualization(post_image, gt_buildings, pred_regions, pred_damages, output_path):
    colors = [(0, 255, 0), (255, 255, 0), (255, 165, 0), (255, 0, 0)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(post_image)
    axes[0].set_title('Original POST-Disaster Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    gt_image = post_image.copy()
    for building in gt_buildings:
        gt_image[building['mask']] = colors[building['damage_type']]
    axes[1].imshow(gt_image)
    axes[1].set_title('Ground Truth Damage', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    pred_image = post_image.copy()
    for region, pred_damage in zip(pred_regions, pred_damages):
        pred_image[region['mask']] = colors[pred_damage]
    axes[2].imshow(pred_image)
    axes[2].set_title('Predicted Damage', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='No damage'),
        Patch(facecolor='yellow', label='Minor damage'),
        Patch(facecolor='orange', label='Major damage'),
        Patch(facecolor='red', label='Destroyed'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=11)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()



def run_damage_test():
    print('DAMAGE CLASSIFICATION PIPELINE TEST')
    print('ResNet34 (localización) + DamageCNN (clasificación)')
    print('=' * 60)

    loc_model, damage_model, device = load_models()
    if loc_model is None or damage_model is None:
        return

    test_images_dir = os.path.join(project_root, 'Data', 'test', 'images')
    test_ids = sorted(set([
        f.replace('_post_disaster.png', '')
        for f in os.listdir(test_images_dir)
        if f.endswith('_post_disaster.png')
    ]))

    print(f'Processing {len(test_ids)} test images...')

    output_dir = os.path.join(project_root, 'test_results', 'damage_pipeline')
    os.makedirs(output_dir, exist_ok=True)

    all_gt = []
    all_pred = []
    skipped = 0

    class_names = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']

    for i, image_id in enumerate(test_ids, 1):
        print(f'\nProcessing {i}/{len(test_ids)}: {image_id}')

        pre_image, post_image, labels_data = load_test_images(image_id)
        if pre_image is None:
            print(f'  SKIPPED: missing files')
            skipped += 1
            continue

        gt_buildings = get_ground_truth_buildings(labels_data, post_image.shape)
        if len(gt_buildings) == 0:
            print(f'  SKIPPED: no buildings in GT')
            skipped += 1
            continue

        pred_regions = detect_buildings(loc_model, device, pre_image)

        if len(pred_regions) == 0:
            print(f'  WARNING: no buildings detected')
            continue

        pred_damages = []
        gt_damages = []

        for region in pred_regions:
            pred_damage, _ = predict_damage(damage_model, device, post_image, region['bbox'])
            gt_damage = get_gt_damage_for_bbox(labels_data, region['bbox'])
            pred_damages.append(pred_damage)
            gt_damages.append(gt_damage)

        all_gt.extend(gt_damages)
        all_pred.extend(pred_damages)

        print(f'  Buildings detected: {len(pred_regions)} | GT buildings: {len(gt_buildings)}')

        output_path = os.path.join(output_dir, f'damage_{i}.png')
        create_visualization(post_image, gt_buildings, pred_regions, pred_damages, output_path)
        print(f'  SUCCESS: Saved {output_path}')

    print('\n' + '=' * 60)
    print('RESULTS SUMMARY')
    print('=' * 60)
    print(f'Images processed : {len(test_ids) - skipped}')
    print(f'Images skipped   : {skipped}')
    print(f'Total buildings  : {len(all_gt)}')

    if len(all_gt) > 0:
        f1_macro = f1_score(all_gt, all_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(all_gt, all_pred, average='weighted', zero_division=0)

        print(f'\nF1-Macro    : {f1_macro:.4f}')
        print(f'F1-Weighted : {f1_weighted:.4f}')

        print(f'\nPer-class report:')
        print(classification_report(all_gt, all_pred, target_names=class_names, zero_division=0))

    print(f'Results saved to {output_dir}/')
    print('PIPELINE TEST COMPLETE')


if __name__ == '__main__':
    run_damage_test()