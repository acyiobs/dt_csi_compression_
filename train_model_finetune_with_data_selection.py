import numpy as np
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


if __name__ == "__main__":
    torch.use_deterministic_algorithms(True)    
    real_data_root = "DeepMIMO/DeepMIMO_datasets/Boston5G_3p5_1"
    synth_data_root = "DeepMIMO/DeepMIMO_datasets/Boston5G_3p5_small_notree"
    train_csv = "/train_data_idx.csv"
    val_csv = "/test_data_idx.csv"
    test_csv = "/test_data_idx.csv"
    train_batch_size = 64
    test_batch_size = 1024
    model_path = "checkpoint/03_20_23_23_10_18_CsinetPlus-CsinetPlus.path"
    all_nmse = []

    for num_train_data, num_epoch in zip([1000, 2000, 4000, 8000, 16000, 32000], [50, 50, 50, 50, 100, 100]):
        torch.manual_seed(768)
        train_loader = DataLoader(
            DataFeed(real_data_root, train_csv, num_data_point=32000), batch_size=train_batch_size, shuffle=True
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

        select_data_idx = select_data(train_loader, model_path, num_train_data)
        finetune_loader = DataLoader(
            DataFeed(real_data_root, train_csv, select_data_idx=select_data_idx), batch_size=train_batch_size, shuffle=True
        )

        ret = train_model(
            finetune_loader,
            val_loader,
            test_loader,
            comment=comment,
            encoded_dim=32,
            num_epoch=num_epoch,
            if_writer=True,
            model_path=model_path,
            lr=1e-4,
        )
        all_nmse.append(ret["test_nmse"])
    all_nmse = np.asarray(all_nmse)
    print(all_nmse)
    savemat(
        "result3/all_nmse_finetune_select.mat",
        {"all_nmse_finetune_select": all_nmse},
        )
    print("done")
