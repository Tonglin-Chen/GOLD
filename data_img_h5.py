import torch

from torchvision import transforms
from torch.utils.data import Dataset
import h5py
import numpy as np
from torchvision.utils import save_image



class GlobVideoDataset(Dataset):
    def __init__(self, root, phase, data_name):
        self.root = root
        self.data_name = data_name
        with h5py.File(root, 'r', libver='latest', swmr=True) as f:
            self.total_data =f[phase]
            
            self.video = self.total_data['image'][()]
            self.seg = self.total_data['segment'][()]
            self.clss = self.total_data['cls'][()]

            if self.data_name == 'unique_gso':
                num_video, num_view, _, _, _ = self.video.shape
                self.video = self.video[:, 0]
                self.seg = self.seg[:, 0]

            
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx):

        img = self.transform(self.video[idx])
        seg = torch.from_numpy(self.seg[idx]).to(torch.int64)
        clss = self.clss[idx]


        return img, seg, clss
