import torch
import numpy as np
import os
import cv2
import json
import matplotlib.pyplot as plt
from shapely.wkt import loads
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from models.model import create_model


def load_localization_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model().to(device)

    checkpoint_path = os.path.join(
        project_root, 'weights', 'best_model_resnet34_bce_dice_large.pth'
    )

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


def load_test_image_and_labels(image_id):
    image_path = os.path.join(
        project_root, f'Data/test/images/{image_id}_pre_disaster.png'
    )
    labels_path = os.path.join(
        project_root, f'Data/test/labels/{image_id}_pre_disaster.json'
    )

    if not os.path.exists(image_path) or not os.path.exists(labels_path):
        return None, None, None

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with open(labels_path, 'r') as f:
        labels_data = json.load(f)

    ground_truth_mask = np.zeros(
        (image.shape[0], image.shape[1]), dtype=np.uint8
    )

    if 'features' in labels_data and 'xy' in labels_data['features']:
        for feature in labels_data['features']['xy']:
            if feature['properties']['feature_type'] == 'building':
                wkt_str = feature.get('wkt')
                if wkt_str:
                    try:
                        geometry = loads(wkt_str)
                        coords = list(geometry.exterior.coords)
                        pts = np.array([[int(x), int(y)] for x, y in coords])
                        cv2.fillPoly(ground_truth_mask, [pts], 255)
                    except:
                        continue

    return image, ground_truth_mask, labels_data


def predict_localization(model, device, image):
    image_tensor = torch.from_numpy(
        image.transpose(2, 0, 1)
    ).float().unsqueeze(0).to(device) / 255.0

    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.sigmoid(output).cpu().numpy()[0, 0]

    prediction_mask = (prediction > 0.5).astype(np.uint8) * 255
    return prediction_mask


def compute_metrics(gt_mask, pred_mask, threshold=127):
    gt_bin = (gt_mask > threshold).astype(np.float32)
    pred_bin = (pred_mask > threshold).astype(np.float32)

    intersection = (gt_bin * pred_bin).sum()
    gt_sum = gt_bin.sum()
    pred_sum = pred_bin.sum()
    union = gt_sum + pred_sum - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)
    f1 = (2 * intersection + 1e-6) / (gt_sum + pred_sum + 1e-6)

    return float(iou), float(f1)


def create_three_panel_visualization(image, ground_truth_mask, prediction_mask,
                                     image_id, output_path, iou, f1):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image)
    axes[0].set_title('Original PRE-Disaster Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(ground_truth_mask, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('Ground Truth Building Masks', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(prediction_mask, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title(
        f'ResNet34 U-Net (Large Dataset)\nIoU: {iou:.4f} | F1: {f1:.4f}',
        fontsize=14, fontweight='bold'
    )
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f'SUCCESS: Saved {output_path}')


def run_localization_test():
    print('RESNET34 U-NET LARGE DATASET LOCALIZATION TEST')
    print('=' * 60)

    model, device = load_localization_model()
    if model is None:
        return

    test_images_dir = os.path.join(project_root, 'Data', 'test', 'images')
    test_images = sorted([
        f.replace('_pre_disaster.png', '')
        for f in os.listdir(test_images_dir)
        if f.endswith('_pre_disaster.png')
    ])

    print(f'Processing {len(test_images)} test images...')

    output_dir = os.path.join(project_root, 'test_results', 'resnet34_large_localization')
    os.makedirs(output_dir, exist_ok=True)

    all_iou = []
    all_f1 = []
    skipped = []

    for i, image_id in enumerate(test_images, 1):
        print(f'\nProcessing image {i}/{len(test_images)}: {image_id}')

        image, ground_truth_mask, labels_data = load_test_image_and_labels(image_id)
        if image is None:
            print(f'ERROR: Could not load {image_id}')
            continue

        if ground_truth_mask.sum() == 0:
            print(f'  SKIPPED: empty ground truth mask')
            skipped.append(image_id)
            continue

        prediction_mask = predict_localization(model, device, image)

        iou, f1 = compute_metrics(ground_truth_mask, prediction_mask)
        all_iou.append(iou)
        all_f1.append(f1)
        print(f'  IoU: {iou:.4f} | F1: {f1:.4f}')

        output_path = os.path.join(output_dir, f'resnet34_large_test_{i}.png')
        create_three_panel_visualization(
            image, ground_truth_mask, prediction_mask,
            image_id, output_path, iou, f1
        )

    print('\n' + '=' * 60)
    print('RESULTS SUMMARY')
    print('=' * 60)
    print(f'Valid images       : {len(all_iou)}')
    print(f'Skipped (empty GT) : {len(skipped)}')
    if all_iou:
        print(f'Mean IoU : {np.mean(all_iou):.4f} ± {np.std(all_iou):.4f}')
        print(f'Mean F1  : {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}')
        print(f'Best  IoU: {np.max(all_iou):.4f} | Worst IoU: {np.min(all_iou):.4f}')
    print(f'Results saved to {output_dir}/')
    print('LOCALIZATION TEST COMPLETE')


if __name__ == '__main__':
    run_localization_test()