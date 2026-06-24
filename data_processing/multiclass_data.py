import os
import torch
import numpy as np
import json
import cv2
from torch.utils.data import Dataset, DataLoader
from shapely.wkt import loads
from shapely.geometry import Polygon
import warnings
warnings.filterwarnings('ignore')


class XBDMulticlassDataset(Dataset):
    """
    Dataset para el segundo enfoque (U-Net multiclase).
    - Entrada: concatenación de imagen pre y post desastre (6 canales)
    - Salida: máscara multiclase con 5 clases:
        0 = fondo
        1 = no-damage
        2 = minor-damage
        3 = major-damage
        4 = destroyed
    """

    DAMAGE_CLASSES = {
        'no-damage': 1,
        'minor-damage': 2,
        'major-damage': 3,
        'destroyed': 4,
        'un-classified': 1  
    }

    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split
        self.image_dir = os.path.join(data_dir, split, 'images')
        self.label_dir = os.path.join(data_dir, split, 'labels')

        self.samples = []
        self._load_samples()

    def _load_samples(self):
        if not os.path.exists(self.image_dir) or not os.path.exists(self.label_dir):
            print(f'Missing directories: {self.image_dir} or {self.label_dir}')
            return

        pre_files = [f for f in os.listdir(self.image_dir) if f.endswith('_pre_disaster.png')]

        for pre_file in pre_files:
            base_id = pre_file.replace('_pre_disaster.png', '')
            post_file = f'{base_id}_post_disaster.png'
            post_label = f'{base_id}_post_disaster.json'

            pre_img_path = os.path.join(self.image_dir, pre_file)
            post_img_path = os.path.join(self.image_dir, post_file)
            post_lbl_path = os.path.join(self.label_dir, post_label)

            if os.path.exists(post_img_path) and os.path.exists(post_lbl_path):
                self.samples.append({
                    'pre_image': pre_img_path,
                    'post_image': post_img_path,
                    'post_label': post_lbl_path
                })

        print(f'{self.split} samples: {len(self.samples)}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        pre_img = cv2.imread(sample['pre_image'])
        if pre_img is None:
            pre_img = np.zeros((1024, 1024, 3), dtype=np.uint8)
        else:
            pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)
            if pre_img.shape[:2] != (1024, 1024):
                pre_img = cv2.resize(pre_img, (1024, 1024))

        post_img = cv2.imread(sample['post_image'])
        if post_img is None:
            post_img = np.zeros((1024, 1024, 3), dtype=np.uint8)
        else:
            post_img = cv2.cvtColor(post_img, cv2.COLOR_BGR2RGB)
            if post_img.shape[:2] != (1024, 1024):
                post_img = cv2.resize(post_img, (1024, 1024))

        mask = self._create_multiclass_mask(sample['post_label'])

        pre_img = pre_img.astype(np.float32) / 255.0
        post_img = post_img.astype(np.float32) / 255.0

        pre_tensor = np.transpose(pre_img, (2, 0, 1))
        post_tensor = np.transpose(post_img, (2, 0, 1))

        image_6ch = np.concatenate([pre_tensor, post_tensor], axis=0)

        return torch.from_numpy(image_6ch), torch.from_numpy(mask).long()

    def _create_multiclass_mask(self, label_path):
        """
        Crea máscara multiclase a partir del JSON post-desastre.
        Cada píxel dentro de un edificio recibe el valor de su clase de daño.
        El fondo es 0.
        """
        mask = np.zeros((1024, 1024), dtype=np.uint8)

        try:
            with open(label_path, 'r') as f:
                data = json.load(f)

            if 'features' in data and 'xy' in data['features']:
                for feature in data['features']['xy']:
                    props = feature.get('properties', {})
                    if props.get('feature_type') != 'building':
                        continue

                    wkt_str = feature.get('wkt')
                    subtype = props.get('subtype', 'no-damage')
                    damage_class = self.DAMAGE_CLASSES.get(subtype, 1)

                    if wkt_str:
                        try:
                            geom = loads(wkt_str)
                            if isinstance(geom, Polygon) and geom.is_valid:
                                coords = np.array(geom.exterior.coords, dtype=np.int32)
                                cv2.fillPoly(mask, [coords], damage_class)
                        except:
                            continue
        except:
            pass

        return mask


def get_multiclass_data_loaders(data_dir, batch_size=2):
    try:
        train_dataset = XBDMulticlassDataset(data_dir, 'train')
        val_dataset = XBDMulticlassDataset(data_dir, 'test')

        if len(train_dataset) == 0:
            print('No training samples found!')
            return None, None

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        return train_loader, val_loader

    except Exception as e:
        print(f'Error creating data loaders: {e}')
        return None, None