import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        self.transforms = transforms
        self.train = train  
        target_list = []
        for file in sorted(os.listdir(base_folder)):
            if train and 'data_batch' in file:
                target_list.append(file)
            if not train and 'test_batch' in file:
                target_list.append(file)
        self.X, self.y = np.empty((0, 3, 32, 32)), np.empty((0, ))
        for p in target_list:
            with open(os.path.join(base_folder, p), 'rb') as f:
                data = pickle.load(f, encoding='bytes')
                self.X = np.concatenate((self.X, data[b"data"].reshape(-1, 3, 32, 32)), axis=0)
                self.y = np.concatenate((self.y, data[b"labels"]), axis=0)
        self.X = self.X.astype(np.float32) / 255.0
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        img, label = self.X[index].reshape((3, 32, 32)), self.y[index]
        if self.transforms is not None:
            for t in self.transforms:
                img = t(img)
        return img, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION