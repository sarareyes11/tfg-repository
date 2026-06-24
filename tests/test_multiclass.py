import torch
import numpy as np
import os
import cv2
import json
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import f1_score, classification_report
from shapely.wkt import loads

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from models.model_multiclass import create_model


CLASS_NAMES = ['background', 'no-damage', 'minor-damage', 'major-damage', 'destroyed']
COLORS = [
    (0, 0, 0),      
    (0, 255, 0),    
    (255, 255, 0),    
    (255, 165, 0),    
    (255, 0, 0),     
]

DAMAGE_CLASSES = {
    'no-damage': 1,
    'minor-damage': 2,
    'major-damage': 3,
    'destroyed': 4,
    'un-classified': 1
}




def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model().to(device)

    checkpoint_path = os.path.join(project_root, 'weights', 'best_model_multiclass.pth')
    if not os.path.exists(checkpoint_path):
        print(f'ERROR: Model not found: {checkpoint_path}')
        return None, device

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f'SUCCESS: Model loaded from {checkpoint_path}')
    return model, device



def load_test_sample(data_dir, image_id):
    pre_path = os.path.join(data_dir, 'test', 'images', f'{image_id}_pre_disaster.png')
    post_path = os.path.join(data_dir, 'test', 'images', f'{image_id}_post_disaster.png')
    label_path = os.path.join(data_dir, 'test', 'labels', f'{image_id}_post_disaster.json')

    if not os.path.exists(pre_path) or not os.path.exists(post_path) or not os.path.exists(label_path):
        return None, None, None

    pre_img = cv2.imread(pre_path)
    pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)

    post_img = cv2.imread(post_path)
    post_img = cv2.cvtColor(post_img, cv2.COLOR_BGR2RGB)

    with open(label_path, 'r') as f:
        labels_data = json.load(f)

    return pre_img, post_img, labels_data


def create_gt_mask(labels_data, shape=(1024, 1024)):
    mask = np.zeros(shape, dtype=np.uint8)

    if 'features' in labels_data and 'xy' in labels_data['features']:
        for feature in labels_data['features']['xy']:
            props = feature.get('properties', {})
            if props.get('feature_type') != 'building':
                continue

            wkt_str = feature.get('wkt')
            subtype = props.get('subtype', 'no-damage')
            damage_class = DAMAGE_CLASSES.get(subtype, 1)

            if wkt_str:
                try:
                    geom = loads(wkt_str)
                    coords = np.array(geom.exterior.coords, dtype=np.int32)
                    cv2.fillPoly(mask, [coords], damage_class)
                except:
                    continue

    return mask



def predict(model, device, pre_img, post_img):
    pre = pre_img.astype(np.float32) / 255.0
    post = post_img.astype(np.float32) / 255.0

    pre_t = np.transpose(pre, (2, 0, 1))
    post_t = np.transpose(post, (2, 0, 1))
    image_6ch = np.concatenate([pre_t, post_t], axis=0)

    tensor = torch.from_numpy(image_6ch).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    return pred_mask.astype(np.uint8)


def mask_to_color(mask):
    color_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_idx, color in enumerate(COLORS):
        color_img[mask == class_idx] = color
    return color_img


def create_visualization(post_img, gt_mask, pred_mask, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(post_img)
    axes[0].set_title('Original POST-Disaster Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(mask_to_color(gt_mask))
    axes[1].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(mask_to_color(pred_mask))
    axes[2].set_title('Prediction', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='black', label='Background'),
        Patch(facecolor='green', label='No damage'),
        Patch(facecolor='yellow', label='Minor damage'),
        Patch(facecolor='orange', label='Major damage'),
        Patch(facecolor='red', label='Destroyed'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=11)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()



def run_test():
    print('MULTICLASS U-NET INFERENCE TEST')
    print('Second approach: 6-channel pre+post input, 5-class output')
    print('=' * 60)

    model, device = load_model()
    if model is None:
        return

    data_dir = os.path.join(project_root, '..', 'xView2 - copia - SAM', 'Data')
    data_dir = os.path.abspath(data_dir)
    if not os.path.exists(data_dir):
        data_dir = os.path.join(project_root, 'Data')

    test_images_dir = os.path.join(data_dir, 'test', 'images')
    test_ids = sorted(set([
        f.replace('_pre_disaster.png', '')
        for f in os.listdir(test_images_dir)
        if f.endswith('_pre_disaster.png')
    ]))

    print(f'Processing {len(test_ids)} test images...')

    output_dir = os.path.join(project_root, 'test_results', 'multiclass_v2')
    os.makedirs(output_dir, exist_ok=True)

    all_gt = []
    all_pred = []
    skipped = 0

    for i, image_id in enumerate(test_ids, 1):
        print(f'\nProcessing {i}/{len(test_ids)}: {image_id}')

        pre_img, post_img, labels_data = load_test_sample(data_dir, image_id)
        if pre_img is None:
            print(f'  SKIPPED: missing files')
            skipped += 1
            continue

        gt_mask = create_gt_mask(labels_data)

        if gt_mask.sum() == 0:
            print(f'  SKIPPED: empty ground truth')
            skipped += 1
            continue

        pred_mask = predict(model, device, pre_img, post_img)

        building_pixels = gt_mask > 0
        if building_pixels.sum() == 0:
            skipped += 1
            continue

        all_gt.extend(gt_mask[building_pixels].tolist())
        all_pred.extend(pred_mask[building_pixels].tolist())

        output_path = os.path.join(output_dir, f'multiclass_{i}.png')
        create_visualization(post_img, gt_mask, pred_mask, output_path)
        print(f'  SUCCESS: Saved {output_path}')

    print('\n' + '=' * 60)
    print('RESULTS SUMMARY')
    print('=' * 60)
    print(f'Images processed : {len(test_ids) - skipped}')
    print(f'Images skipped   : {skipped}')
    print(f'Total pixels     : {len(all_gt)}')

    if len(all_gt) > 0:
        f1_macro = f1_score(all_gt, all_pred, average='macro',
                            labels=[1, 2, 3, 4], zero_division=0)
        f1_weighted = f1_score(all_gt, all_pred, average='weighted',
                               labels=[1, 2, 3, 4], zero_division=0)

        print(f'\nF1-Macro    : {f1_macro:.4f}')
        print(f'F1-Weighted : {f1_weighted:.4f}')

        print(f'\nPer-class report (building pixels only):')
        print(classification_report(
            all_gt, all_pred,
            labels=[1, 2, 3, 4],
            target_names=['no-damage', 'minor-damage', 'major-damage', 'destroyed'],
            zero_division=0
        ))

    print(f'Results saved to {output_dir}/')
    print('TEST COMPLETE')


if __name__ == '__main__':
    run_test()