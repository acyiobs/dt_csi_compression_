import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys, datetime
from train_model import test_model
from utils.cal_nmse import cal_nmse
from data_feed.data_feed import DataFeed
from matplotlib import pyplot as plt
from scipy.io import savemat


def select_data(test_loader, model_path, num_data=None):
    ret = test_model(
        test_loader,
        model_path=model_path,
        encoded_dim=32,
    )
    test_loss_all = ret["test_loss_all"]
    test_nmse_all = ret["test_nmse_all"]
    test_data_idx = ret["test_data_idx"]

    sorted_nmse_idx = np.argsort(test_nmse_all)[::-1]
    test_loss_all = test_loss_all[sorted_nmse_idx]
    test_nmse_all = test_nmse_all[sorted_nmse_idx]
    test_data_idx = test_data_idx[sorted_nmse_idx]

    # # visualization
    # count, bins_count = np.histogram(10*np.log10(test_nmse_all), bins=50)
    # pdf = count #/ sum(count)
    # cdf = np.cumsum(pdf)
    # plt.plot(bins_count[1:], cdf, label="Train on synth (32k)")
    # plt.xlabel('Test on real NMSE (dB)')
    # plt.legend()
    # plt.grid()
    # plt.show()

    if not num_data:
        return test_data_idx

    return test_data_idx[:num_data]


if __name__ == "__main__":
    torch.manual_seed(768)
    torch.use_deterministic_algorithms(True)
    real_data_root = "DeepMIMO/DeepMIMO_datasets/Boston5G_3p5_real"
    synth_data_root = "DeepMIMO/DeepMIMO_datasets/Boston5G_3p5_notree"
    train_csv = "/train_data_idx.csv"
    val_csv = "/test_data_idx.csv"
    test_csv = "/test_data_idx.csv"
    test_batch_size = 1024

    test_loader = DataLoader(
        DataFeed(synth_data_root, test_csv, num_data_point=5120),
        batch_size=test_batch_size,
    )
    model_path = "checkpoint/16_07_59_23_10_24_CsinetPlus-CsinetPlus.path"
    data_idx, test_nmse = select_data(test_loader, model_path)
    savemat(
        "result_new_data_1/select_data_synth.mat",
        {"select_data_idx_synth": data_idx, "test_nmse_synth":test_nmse},
    )

    test_loader = DataLoader(
        DataFeed(real_data_root, test_csv, num_data_point=5120),
        batch_size=test_batch_size,
    )
    model_path = "checkpoint/16_07_59_23_10_24_CsinetPlus-CsinetPlus.path"
    data_idx, test_nmse = select_data(test_loader, model_path)
    savemat(
        "result_new_data_1/select_data_real.mat",
        {"select_data_idx_real": data_idx, "test_nmse_real":test_nmse},
    )
