"""Preprocess images folder into L and ab numpy arrays.

This script reads images (png/jpg), resizes them (default 224x224), converts to
LAB color space and saves two numpy arrays:
- l.npy: shape (N, H, W, 1) with L normalized to [0,1]
- ab.npy: shape (N, H, W, 2) with ab normalized to [0,1] where 0.5 -> 0

Usage:
  python preprocess_images.py --input_dir path/to/images --out_dir ./prepared --size 224
"""
import os
import glob
import argparse
import numpy as np
from PIL import Image
from skimage.color import rgb2lab


def process_image(path, size=(224, 224)):
    img = Image.open(path).convert('RGB').resize(size)
    arr = np.array(img).astype('float32') / 255.0
    lab = rgb2lab(arr)  # gives L in [0,100], a and b in approx [-128,127]
    L = lab[:, :, 0:1]
    ab = lab[:, :, 1:3]
    # normalize
    L_norm = L / 100.0  # [0,1]
    ab_norm = (ab / 128.0) / 2.0 + 0.5  # map roughly [-128,128] -> [0,1]
    # Better mapping: (ab / 256) + 0.5 but using 128 for scale stability
    return L_norm.astype('float32'), ab_norm.astype('float32')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--out_dir', default='prepared')
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--ext', default='png')
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.input_dir, f'*.{args.ext}'))
    if not files:
        files = glob.glob(os.path.join(args.input_dir, '*.png')) + glob.glob(os.path.join(args.input_dir, '*.jpg'))
    files = sorted(files)
    os.makedirs(args.out_dir, exist_ok=True)

    L_list = []
    ab_list = []
    for i, f in enumerate(files):
        L, ab = process_image(f, size=(args.size, args.size))
        L_list.append(L)
        ab_list.append(ab)
        if (i + 1) % 100 == 0:
            print(f'Processed {i+1} images')

    L_arr = np.stack(L_list, axis=0)
    ab_arr = np.stack(ab_list, axis=0)

    np.save(os.path.join(args.out_dir, 'l.npy'), L_arr)
    np.save(os.path.join(args.out_dir, 'ab.npy'), ab_arr)
    print('Saved:', os.path.join(args.out_dir, 'l.npy'), os.path.join(args.out_dir, 'ab.npy'))


if __name__ == '__main__':
    main()
