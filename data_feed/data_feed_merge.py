import numpy as np
import pandas as pd
import os
import torch
import sklearn
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from einops import rearrange, reduce, repeat

import matplotlib.pyplot as plt
import torchvision.transforms as T

from data_feed.data_feed_real import create_samples as create_samples_real
from data_feed.data_feed_synth import create_samples as create_samples_synth


class DataFeed(Dataset):
    def __init__(self, data_root_synth, data_root_real, csv_path_synth, csv_path_real, random_state=0, num_data_point_synth=None, num_data_point_real=None, portion_synth=1.0, portion_real=1.0, num_classes=16):
        self.inputs_synth, self.all_best_beam_synth, self.all_bbox_synth, self.all_beam_power_synth = create_samples_synth(data_root_synth, csv_path_synth, random_state, num_data_point_synth, portion_synth, num_classes=16)
        self.inputs_real, self.all_best_beam_real, self.all_bbox_real, self.all_beam_power_real,  = create_samples_real(data_root_real, csv_path_real, random_state, num_data_point_real, portion_real, num_classes=16)

        self.all_inputs = np.concatenate([self.inputs_synth, self.inputs_real], 0)
        self.all_beam_power = np.concatenate([self.all_beam_power_synth, self.all_beam_power_real], 0)
        self.all_best_beam = np.concatenate([self.all_best_beam_synth, self.all_best_beam_real], 0)
        self.all_bbox = np.concatenate([self.all_bbox_synth, self.all_bbox_real], 0)

    def __len__(self):
        return len(self.all_best_beam)

    def __getitem__(self, idx):
        all_inputs = self.all_inputs[idx, :]
        beam_power = self.all_beam_power[idx, :]
        best_beam = self.all_best_beam[idx]
        all_bbox = self.all_bbox[idx, :]
        
        all_inputs = torch.tensor(all_inputs, requires_grad=False)
        beam_power = torch.tensor(beam_power, requires_grad=False)
        best_beam = torch.tensor(best_beam, requires_grad=False)
        all_bbox = torch.tensor(all_bbox, requires_grad=False)
        return all_inputs.float(), all_bbox.float(), beam_power.float(), best_beam.long()



if __name__ == "__main__":
    data_root_synth = "../../data/digital_twin_random_car_position"
    train_csv_synth = data_root_synth + "/train_data_idx.csv"

    data_root_real = "../../data/deepsense/Scenario1"
    train_csv_real = data_root_real + "/train_data_idx.csv"

    num_classes = 16
    batch_size = 64

    train_loader = DataLoader(DataFeed(data_root_synth, data_root_real, train_csv_synth, train_csv_real, random_state=0, num_data_point_synth=100, num_data_point_real=100), batch_size=batch_size)
    all_bbox, beam_power, best_beam = next(iter(train_loader))

    print('done')


  