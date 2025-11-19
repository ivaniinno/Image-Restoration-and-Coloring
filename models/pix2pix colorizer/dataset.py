import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class ImageColorizationDataset(Dataset):
    """Dataset that returns (ab, L) tensors where:
    - L: (1,H,W) grayscale channel normalized to [0,1]
    - ab: (2,H,W) color channels normalized to [0,1]
    """
    def __init__(self, l_array, ab_array, transform=None):
        assert len(l_array) == len(ab_array)
        self.l_array = l_array
        self.ab_array = ab_array
        self.transform = transform
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.l_array)

    def __getitem__(self, idx):
        L = np.array(self.l_array[idx]).reshape((224, 224, 1))
        L = self.to_tensor(L).float()

        ab = np.array(self.ab_array[idx])
        ab = self.to_tensor(ab).float()

        return ab, L
