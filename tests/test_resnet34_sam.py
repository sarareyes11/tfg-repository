import torch
import numpy as np
import os
import cv2
import json
import matplotlib.pyplot as plt
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from models.model import create_model
from shapely.wkt import loads
from segment_anything import sam_model_registry, SamPredictor



def load_unet_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model().to(device)

    checkpoint_path = os.path.join(project_root, 'weights', 'best_model_resnet34_bce_dice.pth')
    if not os.path.exists(checkpoint_path):
        print(f'ERROR: UNet model not found: {checkpoint_path}')
        return None, device

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f'SUCCESS: UNet loaded from {checkpoint_path}')
    return model, device


def load_sam_model(device):
    sam_checkpoint = os.path.join(project_root, 'weights', 'sam_vit_h_4b8939.pth')
    if not os.path.exists(sam_checkpoint):
        print(f'ERROR: SAM model not found: {sam_checkpoint}')
        return None

    sam = sam_model_registry['vit_h'](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)
    print(f'SUCCESS: SAM loaded from {sam_checkpoint}')
    return predictor



def load_test_image_and_labels(image_id):
    image_path = os.path.join(project_root, f'Data/test/images/{image_id}_pre_disaster.png')
    labels_path = os.path.join(project_root, f'Data/test/labels/{image_id}_pre_disaster.json')

    if not os.path.exists(image_path) or not os.path.exists(labels_path):
        return None, None, None

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with open(labels_path, 'r') as f:
        labels_data = json.load(f)

    ground_truth_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

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



def predict_unet(model, device, image):
    image_tensor = torch.from_numpy(
        image.transpose(2, 0, 1)
    ).float().unsqueeze(0).to(device) / 255.0

    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.sigmoid(output).cpu().numpy()[0, 0]

    prediction_mask = (prediction > 0.5).astype(np.uint8) * 255
    return prediction_mask



def extract_bboxes_from_mask(mask, min_area=100):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    bboxes = []

    for i in range(1, num_labels): 
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        bboxes.append([x, y, x + w, y + h])

    return bboxes



def refine_with_sam(predictor, image, bboxes):

    if len(bboxes) == 0:
        return np.zeros(image.shape[:2], dtype=np.uint8)

    predictor.set_image(image)
    refined_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for bbox in bboxes:
        input_box = np.array(bbox)
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False  
        )
        best_mask = masks[0]
        refined_mask[best_mask] = 255

    return refined_mask


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



def create_visualization(image, gt_mask, unet_mask, sam_mask, output_path, iou_unet, f1_unet, iou_sam, f1_sam):
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    axes[0].imshow(image)
    axes[0].set_title('Original PRE-Disaster Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(gt_mask, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(unet_mask, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title(
        f'ResNet34 U-Net\nIoU: {iou_unet:.4f} | F1: {f1_unet:.4f}',
        fontsize=12, fontweight='bold'
    )
    axes[2].axis('off')

    axes[3].imshow(sam_mask, cmap='gray', vmin=0, vmax=255)
    axes[3].set_title(
        f'U-Net + SAM Refined\nIoU: {iou_sam:.4f} | F1: {f1_sam:.4f}',
        fontsize=12, fontweight='bold'
    )
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f'SUCCESS: Saved {output_path}')


def run_pipeline():
    print('RESNET34 U-NET + SAM REFINEMENT PIPELINE')
    print('=' * 60)

    unet_model, device = load_unet_model()
    if unet_model is None:
        return

    sam_predictor = load_sam_model(device)
    if sam_predictor is None:
        return

    test_images_dir = os.path.join(project_root, 'Data', 'test', 'images')
    test_images = sorted([
        f.replace('_pre_disaster.png', '')
        for f in os.listdir(test_images_dir)
        if f.endswith('_pre_disaster.png')
    ])

    print(f'Processing {len(test_images)} test images...')

    output_dir = os.path.join(project_root, 'test_results', 'resnet34_sam')
    os.makedirs(output_dir, exist_ok=True)

    all_iou_unet = []
    all_f1_unet = []
    all_iou_sam = []
    all_f1_sam = []
    skipped = []

    for i, image_id in enumerate(test_images, 1):
        print(f'\nProcessing image {i}/{len(test_images)}: {image_id}')

        image, gt_mask, _ = load_test_image_and_labels(image_id)
        if image is None:
            print(f'  ERROR: Could not load {image_id}')
            continue

        if gt_mask.sum() == 0:
            print(f'  SKIPPED: empty ground truth mask')
            skipped.append(image_id)
            continue

        unet_mask = predict_unet(unet_model, device, image)
        iou_unet, f1_unet = compute_metrics(gt_mask, unet_mask)

        bboxes = extract_bboxes_from_mask(unet_mask, min_area=100)

        sam_mask = refine_with_sam(sam_predictor, image, bboxes)
        iou_sam, f1_sam = compute_metrics(gt_mask, sam_mask)

        all_iou_unet.append(iou_unet)
        all_f1_unet.append(f1_unet)
        all_iou_sam.append(iou_sam)
        all_f1_sam.append(f1_sam)

        print(f'  U-Net  → IoU: {iou_unet:.4f} | F1: {f1_unet:.4f}')
        print(f'  SAM    → IoU: {iou_sam:.4f} | F1: {f1_sam:.4f}')

        output_path = os.path.join(output_dir, f'resnet34_sam_{i}.png')
        create_visualization(
            image, gt_mask, unet_mask, sam_mask,
            output_path, iou_unet, f1_unet, iou_sam, f1_sam
        )

    print('\n' + '=' * 60)
    print('RESULTS SUMMARY')
    print('=' * 60)
    print(f'Valid images       : {len(all_iou_unet)}')
    print(f'Skipped (empty GT) : {len(skipped)}')
    if all_iou_unet:
        print(f'\nResNet34 U-Net:')
        print(f'  Mean IoU : {np.mean(all_iou_unet):.4f} ± {np.std(all_iou_unet):.4f}')
        print(f'  Mean F1  : {np.mean(all_f1_unet):.4f} ± {np.std(all_f1_unet):.4f}')
        print(f'\nResNet34 U-Net + SAM:')
        print(f'  Mean IoU : {np.mean(all_iou_sam):.4f} ± {np.std(all_iou_sam):.4f}')
        print(f'  Mean F1  : {np.mean(all_f1_sam):.4f} ± {np.std(all_f1_sam):.4f}')
        delta_iou = np.mean(all_iou_sam) - np.mean(all_iou_unet)
        delta_f1 = np.mean(all_f1_sam) - np.mean(all_f1_unet)
        print(f'\nMejora SAM:')
        print(f'  ΔIoU : {delta_iou:+.4f}')
        print(f'  ΔF1  : {delta_f1:+.4f}')
    print(f'\nResults saved to {output_dir}/')
    print('PIPELINE COMPLETE')


if __name__ == '__main__':
    run_pipeline()