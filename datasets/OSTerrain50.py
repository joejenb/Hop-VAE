import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class OSTerrain50(Dataset):
    def __init__(self, data_dir, split='train', transform=None, target_transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        train_data, val_data, test_data = self.collate_data()

        if self.split == 'train':
            self.data = train_data

        if self.split == 'val':
            self.data = val_data

        if self.split == 'test':
            self.data = test_data
    
    def collate_data(self):
        all_height_maps = []
        folders = os.listdir(self.data_dir)
        for folder in folders:
            if not folder.startswith('.'):
                files = os.listdir(f'{self.data_dir}/{folder}/')
                for file in files:
                    if not file.startswith('.'):
                        data = open(f'{self.data_dir}/{folder}/{file}', "r")

                        height_map_str = data.readlines()[5:]
                        height_map_li = []
                        for row in height_map_str:
                            row = [float(str_height) for str_height in row.split()]
                            height_map_li.append(row)

                        height_map = torch.Tensor(height_map_li)
                        all_height_maps.append(height_map)

        data = torch.stack(all_height_maps, dim=0).unsqueeze(1)
        data -= data.min()
        data = (data / (data.max() / 2)) - 1

        lengths = [int(len(data)*0.7), int(len(data)*0.1), len(data) - int(len(data)*0.1) - int(len(data)*0.7)]
        train_data, val_data, test_data = torch.split(data, lengths)

        return train_data, val_data, test_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        image = self.data[idx]
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, torch.zeros((1, 1))