import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from torchinfo import summary
from tqdm import tqdm
import sys, datetime
from models.CsinetPlus import CsinetPlus
from utils.cal_nmse import cal_nmse
from data_feed.data_feed import DataFeed
from scipy.io import savemat
from data_selection import select_data
from train_model import train_model
import os


if __name__ == "__main__":
    torch.use_deterministic_algorithms(True)
    real_data_root = "DeepMIMO/DeepMIMO_datasets/Boston5G_3p5_real"
    synth_data_root = "DeepMIMO/DeepMIMO_datasets/Boston5G_3p5_notree"
    train_csv = "/train_data_idx.csv"
    val_csv = "/test_data_idx.csv"
    test_csv = "/test_data_idx.csv"
    train_batch_size = 4
    test_batch_size = 1024

    ckpt_folder_path = "checkpoint"
    model_paths = [
        os.path.join(ckpt_folder_path, filename)
        for filename in os.listdir(ckpt_folder_path)
        if filename.endswith(".path")
    ]

    np.random.seed(10)
    seeds = np.random.randint(0, 10000, size=(1000,))

    num_epoch = 100

    all_all_nmse = []
    for i in range(10):
        all_nmse = []
        model_path = model_paths[i]
        for num_train_data in [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120]:
            torch.manual_seed(seeds[i])
            train_loader = DataLoader(
                DataFeed(real_data_root, train_csv, num_data_point=5120),
                batch_size=train_batch_size,
                shuffle=True,
            )
            val_loader = DataLoader(
                DataFeed(real_data_root, val_csv, num_data_point=10000),
                batch_size=test_batch_size,
            )
            test_loader = DataLoader(
                DataFeed(real_data_root, test_csv, num_data_point=10000),
                batch_size=test_batch_size,
            )

            now = datetime.datetime.now().strftime("%H_%M_%S")
            date = datetime.date.today().strftime("%y_%m_%d")
            comment = "_".join([now, date])

            select_data_idx = select_data(train_loader, model_path, num_train_data)
            finetune_real_dataset = DataFeed(
                real_data_root, train_csv, select_data_idx=select_data_idx
            )

            # finetune_real_dataset = DataFeed(real_data_root, train_csv, num_data_point=num_train_data)
            # finetune_synth_dataset = DataFeed(synth_data_root, train_csv, num_data_point=5120)

            # finetune_dataset = torch.utils.data.ConcatDataset([finetune_real_dataset, finetune_synth_dataset])

            finetune_loader = DataLoader(
                finetune_real_dataset, batch_size=train_batch_size, shuffle=True
            )

            # all_finetune_data_idx = pd.read_csv(real_data_root+train_csv)["data_idx"].to_numpy()

            # channel_correlation = pd.read_csv("DeepMIMO/DeepMIMO_datasets/real_correlatin_idx_sort.csv").to_numpy()
            # channel_correlation_idx_sort = np.int64(channel_correlation[:, 0]) # the index starts from 0
            # channel_correlation_sort = channel_correlation[:, 1]

            # valid_sample_indicator = np.isin(channel_correlation_idx_sort, all_finetune_data_idx)
            # channel_correlation_idx_sort_valid = channel_correlation_idx_sort[valid_sample_indicator]
            # channel_correlation_sort_valid = channel_correlation_sort[valid_sample_indicator]

            # select_data_idx = channel_correlation_idx_sort_valid[:num_train_data]

            # print('Highest correlation in selected data: %f'%(channel_correlation_sort_valid[num_train_data-1]))

            # finetune_real_dataset = DataFeed(real_data_root, train_csv, select_data_idx=select_data_idx)
            # finetune_synth_dataset = DataFeed(synth_data_root, train_csv, num_data_point=32000)

            # finetune_dataset = torch.utils.data.ConcatDataset([finetune_real_dataset, finetune_synth_dataset])

            # finetune_loader = DataLoader(finetune_dataset, batch_size=train_batch_size, shuffle=True)

            ret = train_model(
                finetune_loader,
                None,
                test_loader,
                comment=comment,
                encoded_dim=32,
                num_epoch=num_epoch,
                if_writer=False,
                model_path=model_path,
                lr=1e-3,
            )
            all_nmse.append(ret["test_nmse"])
        all_nmse = np.asarray(all_nmse)
        all_all_nmse.append(all_nmse)
    all_all_nmse = np.stack(all_all_nmse, 0)
    print(all_all_nmse)
    savemat(
        "result_new_data_2/all_nmse_finetune_select.mat",
        {"all_nmse_finetune_select": all_all_nmse},
    )
    print("done")
