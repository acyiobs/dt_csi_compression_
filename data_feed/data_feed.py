import numpy as np
import pandas as pd
import os
import torch
import sklearn
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from einops import rearrange, reduce, repeat


def create_samples(data_root, csv_path, random_state, num_data_point, portion, select_data_idx):
    # load channel data
    channel_ad_clip = loadmat(data_root+'/channel_ad_clip.mat')['all_channel_ad_clip']

    # load data index
    if select_data_idx is None:
        data_idx = pd.read_csv(data_root+csv_path)["data_idx"].to_numpy()
    else:
        data_idx = select_data_idx
    channel_ad_clip = channel_ad_clip[data_idx, ...]

    # shuffle
    channel_ad_clip, data_idx = sklearn.utils.shuffle(channel_ad_clip, data_idx, random_state=random_state)
    channel_ad_clip = np.squeeze(channel_ad_clip)

    if select_data_idx is None: # if no particular data is selected
        if num_data_point:
            channel_ad_clip = channel_ad_clip[:num_data_point, ...]
            data_idx = data_idx[:num_data_point, ...]
        else:
            num_data = data_idx.shape[0]
            p = int(num_data*portion)

            channel_ad_clip = channel_ad_clip[:p, ...]
            data_idx = data_idx[:p, ...]

    channel_ad_clip /= np.linalg.norm(channel_ad_clip, ord='fro', axis=(-1,-2), keepdims=True)
    channel_ad_clip = channel_ad_clip / np.expand_dims(channel_ad_clip[:, 0, 0] / np.abs(channel_ad_clip[:, 0, 0]), (1,2))


    return channel_ad_clip, data_idx


class DataFeed(Dataset):
    def __init__(self, data_root, csv_path, random_state=0, num_data_point=None, portion=1.0, select_data_idx=None):
        self.data_root = data_root
        self.channel_ad_clip, self.data_idx = create_samples(data_root, csv_path, random_state, num_data_point, portion, select_data_idx)

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx, ...]
        channel_ad_clip = self.channel_ad_clip[idx, ...]

        data_idx = torch.tensor(data_idx, requires_grad=False)

        channel_ad_clip = torch.tensor(channel_ad_clip, requires_grad=False)
        channel_ad_clip = torch.view_as_real(channel_ad_clip)
        channel_ad_clip = rearrange(channel_ad_clip, 'Nt Nc RealImag -> RealImag Nt Nc')
        
        return channel_ad_clip.float(), data_idx.long()


if __name__ == "__main__":
    data_root = data_path = "DeepMIMO/DeepMIMO_datasets/Boston5G_3p5_1"
    train_csv = "/train_data_idx.csv"
    val_csv = "/test_data_idx.csv"
    batch_size = 64

    train_loader = DataLoader(DataFeed(data_root, train_csv, portion=1.), batch_size=batch_size)
    channel_ad_clip, data_idx = next(iter(train_loader))

    print('done')


  