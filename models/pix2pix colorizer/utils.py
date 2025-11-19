import torch
import numpy as np
from skimage.color import lab2rgb


def lab_to_rgb(L, ab):
    """Convert batched L and ab tensors to RGB numpy images.
    L: (B,1,H,W) in [0,1]
    ab: (B,2,H,W) in [0,1]
    """
    if torch.is_tensor(L):
        L = L.detach().cpu().numpy()
    if torch.is_tensor(ab):
        ab = ab.detach().cpu().numpy()

    # L shape (B,1,H,W) -> (B,H,W,1)
    L = np.transpose(L, (0, 2, 3, 1)) * 100
    # ab shape (B,2,H,W) -> (B,H,W,2)
    ab = np.transpose(ab, (0, 2, 3, 1))
    # scale ab from [0,1] to [-128,127]
    ab = (ab - 0.5) * 256

    imgs = []
    for i in range(L.shape[0]):
        lab = np.concatenate([L[i], ab[i]], axis=2)
        img_rgb = lab2rgb(lab.astype(np.float64))
        imgs.append(img_rgb)
    return np.stack(imgs, axis=0)


def save_lab_rgb_pair(L, ab, out_path):
    imgs = lab_to_rgb(L.unsqueeze(0), ab.unsqueeze(0))
    from imageio import imwrite
    imwrite(out_path, (imgs[0] * 255).astype('uint8'))
