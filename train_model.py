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


def train_model(
    train_loader,
    val_loader,
    test_loader,
    comment="unknown",
    encoded_dim=16,
    num_epoch=200,
    lr=1e-2,
    if_writer=False,
    model_path=None,
    save_model=False
):
    # check gpu acceleration availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)

    # instantiate the model and send to GPU
    net = CsinetPlus(encoded_dim)
    net.to(device)

    # path to save the model
    comment = comment + "_" + net.name
    if model_path:
        net.load_state_dict(torch.load(model_path))
    else:
        model_path = "checkpoint/" + comment + ".path"

    # print model summary
    if if_writer:
        summary(net, input_data=torch.zeros(16, 2, 32, 32).to(device))
        writer = SummaryWriter(log_dir="runs/" + comment)

    # set up loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100], gamma=0.1
    )
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    # training
    all_val_nmse = []
    for epoch in range(num_epoch):
        net.train()
        running_loss = 0.0
        running_nmse = 0.0
        with tqdm(train_loader, unit="batch", file=sys.stdout) as tepoch:
            for i, data in enumerate(tepoch, 0):
                tepoch.set_description(f"Epoch {epoch}")

                # get the inputs
                input_channel, data_idx = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                encoded_vector, output_channel = net(input_channel)
                loss = criterion(output_channel, input_channel)

                nmse = torch.mean(cal_nmse(input_channel, output_channel), 0).item()

                loss.backward()
                optimizer.step()

                # print statistics
                running_loss = (loss.item() + i * running_loss) / (i + 1)
                running_nmse = (nmse + i * running_nmse) / (i + 1)
                log = OrderedDict()
                log["loss"] = "{:.6e}".format(running_loss)
                log["nmse"] = running_nmse
                tepoch.set_postfix(log)
            scheduler.step()

        if val_loader is None:
            continue  # no validation is needed

        if epoch >= num_epoch - 50 or if_writer:
            # validation
            net.eval()
            with torch.no_grad():
                total = 0
                val_loss = 0
                val_nmse = 0

                for data in val_loader:
                    # get the inputs
                    input_channel, data_idx = data[0].to(device), data[1].to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    encoded_vector, output_channel = net(input_channel)

                    val_loss += (
                        nn.MSELoss(reduction="mean")(
                            input_channel, output_channel
                        ).item()
                        * data_idx.shape[0]
                    )
                    val_nmse += torch.sum(cal_nmse(input_channel, output_channel), 0)
                    total += data_idx.shape[0]

                val_loss /= float(total)
                val_nmse /= float(total)
            all_val_nmse.append(val_nmse.item())
            print("val_loss={:.6e}".format(val_loss), flush=True)
            print("val_nmse={:.6f}".format(val_nmse), flush=True)
            if if_writer:
                writer.add_scalar("Loss/train", running_loss, epoch)
                writer.add_scalar("Loss/test", val_loss, epoch)
                writer.add_scalar("NMSE/train", running_nmse, epoch)
                writer.add_scalar("NMSE/test", val_nmse, epoch)

    if if_writer:
        writer.close()
    if save_model:
        torch.save(net.state_dict(), model_path)

    # test
    net.eval()
    with torch.no_grad():
        total = 0
        test_loss = 0
        test_nmse = 0

        for data in test_loader:
            # get the inputs
            input_channel, data_idx = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            encoded_vector, output_channel = net(input_channel)

            test_loss += (
                nn.MSELoss(reduction="mean")(input_channel, output_channel).item()
                * data_idx.shape[0]
            )
            test_nmse += torch.sum(cal_nmse(input_channel, output_channel), 0).item()
            total += data_idx.shape[0]

        test_loss /= float(total)
        test_nmse /= float(total)

        print("test_loss={:.6e}".format(test_loss), flush=True)
        print("test_nmse={:.6f}".format(test_nmse), flush=True)

        return {
            "all_val_nmse": all_val_nmse,
            "test_loss": test_loss,
            "test_nmse": test_nmse,
            "model_path": model_path,
        }


def test_model(
    test_loader,
    net=None,
    model_path=None,
    encoded_dim=16,
):
    # check gpu acceleration availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)

    # instantiate the model and send to GPU
    if model_path:
        net = CsinetPlus(encoded_dim)
        net.load_state_dict(torch.load(model_path))

    net.to(device)

    # test
    net.eval()
    with torch.no_grad():
        test_loss = []
        test_nmse = []
        test_data_idx = []

        for data in test_loader:
            # get the inputs
            input_channel, data_idx = data[0].to(device), data[1].to(device)

            # forward + backward + optimize
            encoded_vector, output_channel = net(input_channel)
            test_loss.append(nn.MSELoss(reduction="none")(input_channel, output_channel).mean((-1,-2,-3)).cpu().numpy())
            test_nmse.append(cal_nmse(input_channel, output_channel).cpu().numpy())
            test_data_idx.append(data_idx.cpu().numpy())
        test_loss = np.concatenate(test_loss)
        test_nmse = np.concatenate(test_nmse)
        test_data_idx = np.concatenate(test_data_idx)
        return {
            "test_loss_all": test_loss,
            "test_nmse_all": test_nmse,
            "test_data_idx": test_data_idx
        }
    

if __name__ == "__main__":
    torch.manual_seed(768)
    torch.use_deterministic_algorithms(True)
    real_data_root = "DeepMIMO/DeepMIMO_datasets/Boston5G_3p5_real"
    synth_data_root = "DeepMIMO/DeepMIMO_datasets/Boston5G_3p5_notree"
    train_csv = "/train_data_idx.csv"
    val_csv = "/test_data_idx.csv"
    test_csv = "/test_data_idx.csv"
    train_batch_size = 4
    test_batch_size = 1024

    train_loader = DataLoader(
        DataFeed(synth_data_root, train_csv, num_data_point=5120),
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

    ret = train_model(
        train_loader,
        val_loader,
        test_loader,
        comment=comment,
        encoded_dim=32,
        num_epoch=160,
        if_writer=True,
        model_path=None,
        lr=1e-2,
        save_model=True
    )
    print("done")
