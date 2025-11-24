import os
import glob
import torch
from PIL import Image
import numpy as np
from models import build_models
from utils import lab_to_rgb
from torchvision import transforms
from imageio import imwrite


def load_image_as_L(path, size=(224, 224)):
    img = Image.open(path).convert('L').resize(size)
    arr = np.array(img).astype('float32') / 255.0
    arr = arr.reshape((1, 224, 224, 1))
    return arr


def find_latest_weight(path, pattern='*.pt'):
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, pattern))
        if not files:
            return None
        files = sorted(files, key=os.path.getmtime)
        return files[-1]
    elif os.path.isfile(path):
        return path
    return None


def infer_image(input_path, output_path, gen_weights, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    w = find_latest_weight(gen_weights)
    if w is None:
        raise ValueError(f'No weight file found at {gen_weights}')

    generator, _ = build_models(device=device, gen_weights=w)
    generator.eval()

    arr = load_image_as_L(input_path)
    to_tensor = transforms.ToTensor()
    L = to_tensor(arr[0]).unsqueeze(0).to(device).float()

    with torch.no_grad():
        pred_ab = generator(L)

    rgb = lab_to_rgb(L, pred_ab)
    imwrite(output_path, (rgb[0] * 255).astype('uint8'))


def batch_infer(input_dir, output_dir, gen_weights, device=None):
    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(os.path.join(input_dir, '*'))
    files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        print('No images found in', input_dir)
        return
    w = find_latest_weight(gen_weights)
    if w is None:
        raise ValueError(f'No weight file found at {gen_weights}')
    generator, _ = build_models(device=device, gen_weights=w)
    generator.eval()
    to_tensor = transforms.ToTensor()
    for f in files:
        arr = load_image_as_L(f)
        L = to_tensor(arr[0]).unsqueeze(0).to(device).float()
        with torch.no_grad():
            pred_ab = generator(L)
        rgb = lab_to_rgb(L, pred_ab)
        out = os.path.join(output_dir, os.path.basename(f))
        imwrite(out, (rgb[0] * 255).astype('uint8'))
        print('Saved', out)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='input file or input directory')
    parser.add_argument('--output', required=True, help='output file or output directory')
    parser.add_argument('--weights', required=True, help='weight file or directory')
    args = parser.parse_args()

    if os.path.isdir(args.input):
        batch_infer(args.input, args.output, args.weights)
    else:
        infer_image(args.input, args.output, args.weights)
