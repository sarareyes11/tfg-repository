import random
import shutil
from pathlib import Path

images_dir = Path('/home/jovyan/xView2 - copia - SAM/xbd/tier1/images')
labels_dir = Path('/home/jovyan/xView2 - copia - SAM/xbd/tier1/labels')

output_base = Path('/home/jovyan/xView2 - copia - SAM/Data/train')
output_images_dir = output_base / "images"
output_labels_dir = output_base / "labels"

num_samples = 2000
label_ext = ".json"

output_images_dir.mkdir(parents=True, exist_ok=True)
output_labels_dir.mkdir(parents=True, exist_ok=True)

all_images = list(images_dir.glob("*_pre_disaster.png"))

valid_pairs = []

for img_path in all_images:
    label_path = labels_dir / f"{img_path.stem}{label_ext}"

    if label_path.exists():
        valid_pairs.append((img_path, label_path))

print(f"Total pares PRE-disaster válidos encontrados: {len(valid_pairs)}")

if len(valid_pairs) < num_samples:
    raise ValueError(
        f"No hay suficientes pares válidos ({len(valid_pairs)}) para seleccionar {num_samples} muestras."
    )

random.seed(42)
selected_pairs = random.sample(valid_pairs, num_samples)

for img_path, label_path in selected_pairs:
    shutil.copy2(img_path, output_images_dir / img_path.name)
    shutil.copy2(label_path, output_labels_dir / label_path.name)

print(f"\nSe copiaron {num_samples} imágenes PRE-disaster y sus labels asociados correctamente.")
print(f"Imágenes destino: {output_images_dir}")
print(f"Labels destino: {output_labels_dir}")


test_images_dir = Path('/home/jovyan/xView2 - copia - SAM/xbd/test/images')
test_labels_dir = Path('/home/jovyan/xView2 - copia - SAM/xbd/test/labels')

output_test_images_dir = Path('/home/jovyan/xView2 - copia - SAM/Data/test/images')
output_test_labels_dir = Path('/home/jovyan/xView2 - copia - SAM/Data/test/labels')

num_test_samples = 350
label_ext = ".json"

output_test_images_dir.mkdir(parents=True, exist_ok=True)
output_test_labels_dir.mkdir(parents=True, exist_ok=True)

all_test_images = list(test_images_dir.glob("*_pre_disaster.png"))

valid_test_pairs = []

for img_path in all_test_images:
    label_path = test_labels_dir / f"{img_path.stem}{label_ext}"

    if label_path.exists():
        valid_test_pairs.append((img_path, label_path))

print(f"Total pares TEST PRE-disaster válidos encontrados: {len(valid_test_pairs)}")

if len(valid_test_pairs) < num_test_samples:
    raise ValueError(
        f"No hay suficientes pares válidos ({len(valid_test_pairs)}) para seleccionar {num_test_samples} muestras TEST."
    )

random.seed(42)
selected_test_pairs = random.sample(valid_test_pairs, num_test_samples)

for img_path, label_path in selected_test_pairs:
    shutil.copy2(img_path, output_test_images_dir / img_path.name)
    shutil.copy2(label_path, output_test_labels_dir / label_path.name)

print(f"\nSe copiaron {num_test_samples} imágenes PRE-disaster para TEST correctamente.")
print(f"Imágenes test destino: {output_test_images_dir}")
print(f"Labels test destino: {output_test_labels_dir}")