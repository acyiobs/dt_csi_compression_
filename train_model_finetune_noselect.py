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
    model_paths = [os.path.join(ckpt_folder_path, filename) for filename in os.listdir(ckpt_folder_path) if filename.endswith(".path")]

    np.random.seed(10)
    seeds = np.random.randint(0, 10000, size=(1000,))

    num_epoch = 100

    all_all_nmse = []
    for i in range(10):
        all_nmse = []
        model_path = model_paths[i]
        for num_train_data in [10240]:#[10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120]:
            torch.manual_seed(seeds[i])
            train_loader = DataLoader(
                DataFeed(real_data_root, train_csv, num_data_point=num_train_data), batch_size=train_batch_size, shuffle=True
            )
            val_loader = DataLoader(
                DataFeed(real_data_root, val_csv, num_data_point=10000), batch_size=test_batch_size
            )
            test_loader = DataLoader(
                DataFeed(real_data_root, test_csv, num_data_point=10000), batch_size=test_batch_size
            )

            now = datetime.datetime.now().strftime("%H_%M_%S")
            date = datetime.date.today().strftime("%y_%m_%d")
            comment = "_".join([now, date])

            ret = train_model(
                train_loader,
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
        print(all_all_nmse)
    all_all_nmse = np.stack(all_all_nmse, 0)
    print(all_all_nmse)
    savemat(
        "result_new_data_3/all_nmse_finetune_noselect.mat",
        {"all_nmse_finetune_noselect": all_all_nmse},
        )
    print("done")
