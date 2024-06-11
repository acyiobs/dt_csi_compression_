import numpy as np
import torch
from torch.utils.data import DataLoader
import datetime
from data_feed.data_feed import DataFeed
from scipy.io import savemat
from train_model import train_model


if __name__ == "__main__":
    torch.use_deterministic_algorithms(True)
    real_data_root = "DeepMIMO/DeepMIMO_datasets/Boston5G_3p5_real"
    synth_data_root = "DeepMIMO/DeepMIMO_datasets/O1_3p5"
    train_csv = "/train_data_idx.csv"
    val_csv = "/test_data_idx.csv"
    test_csv = "/test_data_idx.csv"
    train_batch_size = 4
    test_batch_size = 1024

    np.random.seed(10)
    seeds = np.random.randint(0, 10000, size=(1000,))

    all_avg_nmse = []
    for i in range(10):
        all_nmse = []
        for num_train_data in [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240]: # [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120]:
            torch.manual_seed(seeds[i])
            train_loader = DataLoader(
                DataFeed(synth_data_root, train_csv, num_data_point=num_train_data, random_state=seeds[i]),
                batch_size=train_batch_size,
                shuffle=True,
            )
            val_loader = DataLoader(
                DataFeed(real_data_root, val_csv, num_data_point=10000, random_state=seeds[i]),
                batch_size=test_batch_size,
            )
            test_loader = DataLoader(
                DataFeed(real_data_root, test_csv, num_data_point=10000, random_state=seeds[i]),
                batch_size=test_batch_size,
            )

            now = datetime.datetime.now().strftime("%H_%M_%S")
            date = datetime.date.today().strftime("%y_%m_%d")
            comment = "_".join([now, date])

            print("Number of trainig data points : " + str(num_train_data))
            ret = train_model(
                train_loader,
                val_loader,
                test_loader,
                comment=comment,
                encoded_dim=32,
                num_epoch=160,
                if_writer=False,
                model_path=None,
                lr=1e-2,
                save_model=(num_train_data==10240)
            )
            all_nmse.append(ret["all_val_nmse"])
        avg_nmse = np.asarray([np.asarray(nmse).mean() for nmse in all_nmse])
        all_avg_nmse.append(avg_nmse)
        savemat(
        "result_new_data_3/all_avg_nmse_train_on_synth_O1_"+str(i)+".mat",
        {"all_avg_nmse_train_on_synth"+str(i): avg_nmse},
        )
    all_avg_nmse = np.stack(all_avg_nmse, 0)

    print(all_avg_nmse)
    savemat(
        "result_new_data_3/all_avg_nmse_train_on_synth_O1.mat",
        {"all_avg_nmse_train_on_synth": all_avg_nmse},
    )
    print("done")
