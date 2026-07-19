import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import os
import argparse
from torch.optim import SGD
import warnings
warnings.filterwarnings("ignore")

from Dataloader.Data import Data
from network import CSFNet
from Metric.mIoU import StreamSegMetrics
from args import ArgumentParser

def parse_args():
    parser = ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_common_args()
    args = parser.parse_args()
    return args


def train_main():
    args = parse_args()

    train_data = Data("train", args.img_path ,args.lidar_path ,args.label_path , [args.val_seq] , (args.crop_H, args.crop_W))
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size , shuffle=True, num_workers=args.num_workers ,drop_last=True)

    val_data = Data("valid", args.img_path ,args.lidar_path ,args.label_path , [args.val_seq] , None )
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=args.batch_size_valid, shuffle=False, num_workers=args.num_workers)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    network = CSFNet(version=args.network , pretrain=args.pretrained, backbone_path=args.backbone_path, num_classes=args.num_classes)


    if torch.cuda.device_count() > 1:
            if args.num_gpus == 2:
                    print("use 2 gpu")
                    network = nn.DataParallel(network)
            if args.num_gpus == 1:
                    print("use 1 gpu")
                    network = nn.DataParallel(network, device_ids=[0])

    elif torch.cuda.device_count() == 1:
            print("use 1 gpu")
            network = nn.DataParallel(network ,device_ids=[0])
    network = network.to(device)


    optimizer = SGD(network.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    lambda1 = lambda epoch: ((1 - (epoch / args.epochs)) ** 0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda1)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)


    if args.weight_path is not None:
         network.load_state_dict(torch.load(args.weight_path))


    metrics = StreamSegMetrics(args.num_classes)
    epoch_losses_train = []
    epoch_losses_val = []
    num_epoch = args.epochs
    z=0
    r=0

    for epoch in range(1, num_epoch+1):
        print("epoch: %d/%d" % (epoch, num_epoch))
        ############################################################################
        # train:
        ############################################################################
        network.train()
        batch_losses = []
        for RGB, X, label in tqdm(train_loader,colour="blue"):
        # for RGB, X, label in train_loader:
            RGB = RGB.to(device)
            X = X.to(device)
            label = (label.type(torch.LongTensor)).to(device)
            outputs = network(RGB, X)

            # compute the loss:
            loss = criterion(outputs, label)
            loss_value = loss.data.detach().cpu().numpy()
            batch_losses.append(loss_value)

            # optimization step:
            optimizer.zero_grad()  # (reset gradients)
            loss.backward()  # (compute gradients)
            optimizer.step()  # (perform optimization step)

        epoch_loss = np.mean(batch_losses)
        epoch_losses_train.append(epoch_loss)
        print("train loss: %g" % epoch_loss)

        scheduler.step()

        ############################################################################
        # test:
        ############################################################################
        if epoch >= args.eval_epochs_start:
            network.eval()
            batch_losses = []
            with torch.no_grad():
                for RGB, X, label in tqdm(val_loader,colour="red"):
                    RGB = RGB.to(device)
                    X = X.to(device)
                    label = (label.type(torch.LongTensor)).to(device)

                    outputs = network(RGB, X)

                    preds = outputs.detach().max(dim=1)[1].cpu().numpy()
                    targets = label.cpu().numpy()
                    metrics.update(targets, preds)

                    # compute the loss:
                    loss = criterion(outputs, label)
                    loss_value = loss.data.cpu().numpy()
                    batch_losses.append(loss_value)

            z = metrics.get_results()
            metrics.reset()
            epoch_loss = np.mean(batch_losses)
            epoch_losses_val.append(epoch_loss)
            print("test/val loss: %g" % epoch_loss)

        if z > r:
            if not os.path.exists("Checkpoints"):
                os.makedirs("Checkpoints")
            torch.save(network.state_dict(), f"Checkpoints/best_weight.pth")
            print("########################################")
            print("              BEST RESULT               ")
            print("########################################")
            r = z


if __name__ == '__main__':
    train_main()