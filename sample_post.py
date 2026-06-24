import shutil
from pathlib import Path


train_images_src = Path('/home/jovyan/xView2 - copia - SAM/xbd/tier1/images')
train_labels_src = Path('/home/jovyan/xView2 - copia - SAM/xbd/tier1/labels')

test_images_src = Path('/home/jovyan/xView2 - copia - SAM/xbd/test/images')
test_labels_src = Path('/home/jovyan/xView2 - copia - SAM/xbd/test/labels')


train_images_dst = Path('/home/jovyan/xView2 - copia - SAM/Data/train/images')
train_labels_dst = Path('/home/jovyan/xView2 - copia - SAM/Data/train/labels')

test_images_dst = Path('/home/jovyan/xView2 - copia - SAM/Data/test/images')
test_labels_dst = Path('/home/jovyan/xView2 - copia - SAM/Data/test/labels')


def copy_post_disaster(images_dst, labels_dst, images_src, labels_src, split_name):
    print(f'\n{"="*60}')
    print(f'Procesando split: {split_name}')
    print(f'{"="*60}')

    existing_pre = list(images_dst.glob('*_pre_disaster.png'))
    ids = [f.stem.replace('_pre_disaster', '') for f in existing_pre]
    print(f'IDs pre-desastre encontrados en destino: {len(ids)}')

    copied_images = 0
    copied_labels = 0
    missing_images = []
    missing_labels = []

    for image_id in ids:
        post_img_name = f'{image_id}_post_disaster.png'
        src_img = images_src / post_img_name
        dst_img = images_dst / post_img_name

        if src_img.exists():
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)
            copied_images += 1
        else:
            missing_images.append(post_img_name)

        post_lbl_name = f'{image_id}_post_disaster.json'
        src_lbl = labels_src / post_lbl_name
        dst_lbl = labels_dst / post_lbl_name

        if src_lbl.exists():
            if not dst_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)
            copied_labels += 1
        else:
            missing_labels.append(post_lbl_name)

    print(f'Imágenes post copiadas : {copied_images}/{len(ids)}')
    print(f'Labels post copiados   : {copied_labels}/{len(ids)}')

    if missing_images:
        print(f'Imágenes no encontradas ({len(missing_images)}):')
        for m in missing_images[:5]:
            print(f'  {m}')
        if len(missing_images) > 5:
            print(f'  ... y {len(missing_images) - 5} más')

    if missing_labels:
        print(f'Labels no encontrados ({len(missing_labels)}):')
        for m in missing_labels[:5]:
            print(f'  {m}')
        if len(missing_labels) > 5:
            print(f'  ... y {len(missing_labels) - 5} más')


if __name__ == '__main__':
    copy_post_disaster(train_images_dst, train_labels_dst, train_images_src, train_labels_src, 'TRAIN')
    copy_post_disaster(test_images_dst, test_labels_dst, test_images_src, test_labels_src, 'TEST')

    print('\n' + '='*60)
    print('PROCESO COMPLETADO')
    print('='*60)