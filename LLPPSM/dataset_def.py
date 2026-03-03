from torch.utils.data import Dataset
import pandas as pd
import os
import torch
import numpy as np


class PhysionetDataset(Dataset):
    """
    Dataset definition for the Physionet Challenge 2012 dataset.
    """

    def __init__(self, csv_file_data, csv_file_label, mask_file, root_dir, transform=None):
        self.data_source = pd.read_csv(os.path.join(root_dir, csv_file_data), header=None)
        self.mask_source = pd.read_csv(os.path.join(root_dir, mask_file), header=None)
        self.label_source = pd.read_csv(os.path.join(root_dir, csv_file_label), header=0)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patient_data = self.data_source.iloc[idx, :]
        patient_data = torch.Tensor(np.array(patient_data))

        mask = self.mask_source.iloc[idx, :]
        mask = np.array(mask, dtype='uint8')

        label = self.label_source.iloc[idx, :8].to_numpy()
        prev_timestamps = self.label_source.iloc[idx, 8:]
        #time, id, age, gender, height, height_presence, icu type, mortality

        label = torch.Tensor(np.nan_to_num(np.array(label)))

        prev_timestamps = torch.Tensor(np.nan_to_num(np.array(prev_timestamps)))

        sample = {'data': patient_data, 'label': label, 'idx': idx, 'mask': mask, 'prev_timestamps': prev_timestamps}
        return sample

class HealthMNISTDatasetConv(Dataset):
    """
    Dataset definiton for the Health MNIST dataset when using CNN-based VAE.

    Data formatted as dataset_length x 36 x 36.
    """

    def __init__(self, csv_file_data, csv_file_label, mask_file, root_dir, transform=None):

        self.data_source = pd.read_csv(os.path.join(root_dir, csv_file_data), header=None)
        self.mask_source = pd.read_csv(os.path.join(root_dir, mask_file), header=None)
        self.label_source = pd.read_csv(os.path.join(root_dir, csv_file_label), header=0)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            return [self.get_item(i) for i in range(start, stop, step)] 
        elif isinstance(key, int):
            return self.get_item(key)
        else:
            raise TypeError

    def get_item(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        digit = self.data_source.iloc[idx, :]
        digit = np.array(digit, dtype='uint8')
        digit = digit.reshape(36, 36)
        digit = digit[..., np.newaxis]

        mask = self.mask_source.iloc[idx, :]
        mask = np.array([mask], dtype='uint8')

        label = self.label_source.iloc[idx, :7].to_numpy()
        prev_timestamps = self.label_source.iloc[idx, 7:]
        # time_age,  disease_time,  subject,  gender,  disease,  location

        #time_age,  subject,  gender,  disease,  location

        label = torch.Tensor(np.nan_to_num(np.array(label[np.array([5, 0, 4, 3, 6])])))

        prev_timestamps = torch.Tensor(np.nan_to_num(np.array(prev_timestamps)))

        if self.transform:
            digit = self.transform(digit)

        sample = {'data': digit, 'label': label, 'idx': idx, 'mask': mask, 'prev_timestamps': prev_timestamps}
        return sample
