from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip
class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super(MNISTDataset, self).__init__(transforms=transforms)
        with gzip.open(image_filename, 'rb') as f:
            self.images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28).astype(np.float32) / 255.0
        with gzip.open(label_filename, 'rb') as f:
            self.labels = np.frombuffer(f.read(), np.uint8, offset=8)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        img, label = self.images[index], self.labels[index]
        img = self.apply_transforms(img)
        return (img, label)
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.labels.shape[0]
        ### END YOUR SOLUTION