import os
import io
import sys
import glob
import time
from multiprocessing import Process

import numpy as np
import cv2
import lmdb
from tqdm import tqdm
from PIL import Image


SUBSAMPLING = 0
DEF_JPG_QUALITY = 90


def run_process(target, kwargs):
    p = Process(target=target, kwargs=kwargs)
    p.start()
    p.join()


def decode_img(value):
    try:
        img = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), 1)
        if img is None:
            raise IOError('cv2.imdecode failed')
        img = img[:, :, ::-1]  # BGR => RGB
    except IOError:
        try:
            img = np.asarray(Image.open(io.BytesIO(value)))
        except:
            return None
    except:
        return None
    return img


def process_wide_img(img, width, height, convert_to_canvas):
    ch = int(np.round(width * img.shape[0] / img.shape[1]))
    if img.shape[1] < width or ch < height:
        return None

    img = img[(img.shape[0] - ch) // 2: (img.shape[0] + ch) // 2]
    img = Image.fromarray(img, 'RGB')
    img = img.resize((width, height), Image.LANCZOS)

    if convert_to_canvas:
        img = np.asarray(img)
        canvas = np.zeros([width, width, 3], dtype=np.uint8)
        canvas[(width - height) // 2: (width + height) // 2, :] = img
        img = Image.fromarray(canvas, 'RGB')

    return img


def prepare_dir_and_fpath(image_idx, target_dir):
    # Allow names for max 1M images
    digits_in_number = 6
    img_format = ('%0' + str(digits_in_number) + 'd')
    idx_dir = img_format % (1000 * (image_idx // 1000))
    img_dir = os.path.join(target_dir, idx_dir)
    os.makedirs(img_dir, exist_ok=True)
    fname = img_format % image_idx
    fpath = os.path.join(img_dir, fname + '.jpg')
    return fpath


def get_all_keys_dict(lmdb_dir):
    start_time = time.time()
    db_keys_dict = {}
    # Set number of images for each task
    task_step = 100000
    with lmdb.open(lmdb_dir, max_readers=1, readonly=True, lock=False,
                   readahead=False, meminit=False).begin(write=False) as txn:
        total_images = txn.stat()['entries']
        digits_in_number = len(str(total_images))
        dict_key_format = ('%0' + str(digits_in_number) + 'd')
        with tqdm(total=total_images, desc='Database keys') as pbar:
            db_iter = txn.cursor().iternext(keys=True, values=False)
            for idx in range(total_images):
                try:
                    db_key = next(db_iter)
                except:
                    print('Except idx:', idx)
                    continue
                dict_key = dict_key_format % (task_step * (idx // task_step))
                if dict_key not in db_keys_dict.keys():
                    db_keys_dict[dict_key] = []
                # db_key = str(db_key, 'utf-8') # ?
                db_keys_dict[dict_key].append(db_key)
                pbar.update(1)

    total_time = time.time() - start_time
    print(f'Obtained all keys in {total_time}, len(keys) = {len(db_keys_dict)}')
    return db_keys_dict


#----------------------------------------------------------------------------


def process_wide_task(lmdb_dir, db_keys, target_dir, width, height, convert_to_canvas, max_images, jpg_quality):
    progress_dirs = glob.glob(os.path.join(target_dir, '*'))
    if len(progress_dirs) > 0:
        last_progress_dir = sorted(progress_dirs)[-1]
        last_img = sorted(glob.glob(os.path.join(last_progress_dir, '*')))[-1]
        last_img_idx = int(os.path.split(last_img)[1].rsplit('.', 1)[0])
        # print('Last progress dir:', last_progress_dir)
        # print('Last img:', last_img)
        # print('Last img idx:', last_img_idx)
    else:
        last_img_idx = -1

    img_idx = last_img_idx + 1
    n_skipped = 0
    n_corrupted = 0
    with lmdb.open(lmdb_dir, max_readers=1, readonly=True, lock=False,
                   readahead=False, meminit=False).begin(write=False) as txn:
        for db_key in tqdm(db_keys, desc='Task images'):
            value = txn.get(db_key)
            img = decode_img(value)

            if img is None:
                n_skipped += 1
                n_corrupted += 1
                continue

            # Skip current image if function is called in restore mode
            if img_idx < last_img_idx:
                img_idx += 1
                continue

            img = process_wide_img(img, width=width, height=height, convert_to_canvas=convert_to_canvas)
            if img is None:
                n_skipped += 1
                continue

            fpath = prepare_dir_and_fpath(img_idx, target_dir)
            img.save(fpath, format='jpeg', optimize=True, subsampling=SUBSAMPLING, quality=jpg_quality)

            img_idx += 1

            if img_idx == max_images:
                break

    print(f'Corrupted images: {n_corrupted}, skipped images: {n_skipped}')


def create_lsun_wide_efficient(lmdb_dir, target_dir, width=512, height=384, convert_to_canvas=False, max_images=None,
                               jpg_quality=DEF_JPG_QUALITY, total_progress=True, restore_progress=False):
    assert width == 2 ** int(np.round(np.log2(width)))
    assert height <= width
    print('Loading LSUN dataset from "%s"' % lmdb_dir)
    start_time = time.time()
    db_keys_dict = get_all_keys_dict(lmdb_dir)
    with tqdm(total=len(db_keys_dict.keys()), desc='Tasks') as pbar:
        for k, db_keys in db_keys_dict.items():
            run_process(
                target=process_wide_task,
                kwargs={
                    'lmdb_dir': lmdb_dir,
                    'db_keys': db_keys,
                    'target_dir': target_dir,
                    'width': width,
                    'height': height,
                    'convert_to_canvas': convert_to_canvas,
                    'max_images': max_images,
                    'jpg_quality': jpg_quality
                }
            )
            pbar.update(1)
    total_time = time.time() - start_time
    print(f'Processed LSUN dataset in {total_time:.3f}s')


#----------------------------------------------------------------------------

def create_lsun(lmdb_dir, target_dir, resolution=256, max_images=None, jpg_quality=DEF_JPG_QUALITY):
    print('Loading LSUN dataset from "%s"' % lmdb_dir)
    with lmdb.open(lmdb_dir, readonly=True).begin(write=False) as txn:
        total_images = txn.stat()['entries']
        if max_images is None:
            max_images = total_images
        image_idx = 0
        with tqdm(total=max_images, desc='Images') as pbar:
            for _idx, (_key, value) in enumerate(txn.cursor()):
                try:
                    img = decode_img(value)
                    if img is None:
                        continue

                    crop = np.min(img.shape[:2])
                    img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
                    img = Image.fromarray(img, 'RGB')
                    img = img.resize((resolution, resolution), Image.LANCZOS)

                    fpath = prepare_dir_and_fpath(image_idx, target_dir)
                    img.save(fpath, format='jpeg', optimize=True, subsampling=SUBSAMPLING, quality=jpg_quality)

                    image_idx += 1
                    pbar.update(1)
                except:
                    print(sys.exc_info()[1])
                if image_idx == max_images:
                    break


#----------------------------------------------------------------------------

def create_lsun_wide(lmdb_dir, target_dir, width=512, height=384, convert_to_canvas=False, max_images=None,
                     jpg_quality=DEF_JPG_QUALITY, total_progress=True, restore_progress=False):
    assert width == 2 ** int(np.round(np.log2(width)))
    assert height <= width
    print('Loading LSUN dataset from "%s"' % lmdb_dir)
    progress_dirs = glob.glob(os.path.join(target_dir, '*'))
    if restore_progress and len(progress_dirs) > 0:
        last_progress_dir = sorted(progress_dirs)[-1]
        last_img = sorted(glob.glob(os.path.join(last_progress_dir, '*')))[-1]
        last_img_idx = int(os.path.split(last_img)[1].rsplit('.', 1)[0])
        print(last_progress_dir)
        print(last_img)
        print(last_img_idx)
    else:
        last_img_idx = 0

    with lmdb.open(lmdb_dir, readonly=True, max_readers=1, lock=False, readahead=False,
                   meminit=False).begin(write=False) as txn:
        total_images = txn.stat()['entries']
        if max_images is None:
            max_images = total_images
        image_idx = 0
        n_skipped = 0
        n_corrupted = 0
        tqdm_total = total_images if total_progress else max_images
        with tqdm(total=tqdm_total, desc='Images') as pbar:
            for idx, (_key, value) in enumerate(txn.cursor()):
                try:
                    img = decode_img(value)
                    if img is None:
                        n_skipped += 1
                        n_corrupted += 1
                        continue

                    # Skip current image if function is called in restore mode
                    if image_idx < last_img_idx:
                        image_idx += 1
                        continue

                    img = process_wide_img(img, width=width, height=height, convert_to_canvas=convert_to_canvas)
                    if img is None:
                        n_skipped += 1
                        if total_progress:
                            pbar.update(1)
                        continue

                    fpath = prepare_dir_and_fpath(image_idx, target_dir)
                    img.save(fpath, format='jpeg', optimize=True, subsampling=SUBSAMPLING, quality=jpg_quality)

                    image_idx += 1
                    pbar.update(1)
                except:
                    print(sys.exc_info()[1])
                    break
                if image_idx == max_images:
                    break

    print(f'Saved images: {image_idx}, corrupted images: {n_corrupted}, skipped images: {n_skipped}')


if __name__ == '__main__':
    # Note: save images in jpg, as database itself consists of jpg images with quality 75
    images_dir = 'car'
    """
    target_dir = 'car_10000'
    resolution = 256
    max_images = 500
    jpg_quality = 95
    os.makedirs(target_dir, exist_ok=True)
    """

    #create_lsun(images_dir, target_dir, resolution=resolution, max_images=max_images, jpg_quality=jpg_quality)

    width = 512
    height = 384
    jpg_quality = 85
    convert_to_canvas = False
    max_images = None
    total_progress = True
    target_dir = f'car_wide_{jpg_quality}'
    os.makedirs(target_dir, exist_ok=True)

    # Create data efficiently to avoid huge RAM usage
    """
    create_lsun_wide_efficient(images_dir, target_dir, width=width, height=height,
                               convert_to_canvas=convert_to_canvas, max_images=max_images,
                               jpg_quality=jpg_quality, total_progress=total_progress, restore_progress=True)
    """


    create_lsun_wide(images_dir, target_dir, width=width, height=height, convert_to_canvas=convert_to_canvas,
                     max_images=max_images, jpg_quality=jpg_quality, total_progress=total_progress, restore_progress=True)

    # Run benchmark to compare size if dirs
    """
    max_images = 10000
    total_progress = True
    for quality in [75, 80, 85, 90, 95]:
        for canvas in [True, False]:
            target_dir = f'car_opt_{max_images}_canvas{1 if canvas else 0}_quality{quality}'
            os.makedirs(target_dir, exist_ok=True)
            create_lsun_wide(images_dir, target_dir, width=width, height=height,
                             convert_to_canvas=canvas, max_images=max_images, jpg_quality=quality, total_progress=total_progress)
    """
