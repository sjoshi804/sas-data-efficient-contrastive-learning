import argparse
import os
import random

import numpy as np
from sas.subset_dataset import CustomSubsetDataset
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import wandb
from configs import SupportedDatasets, get_datasets
from evaluate.lbfgs import test_clf
from resnet import *
from util import Random


def main(rank: int, world_size: int, args: int):
    # Determine Device 
    device = rank
    if args.distributed:
        device = args.device_ids[rank]
        torch.cuda.set_device(args.device_ids[rank])
        args.lr *= world_size

    # WandB Logging
    if not args.distributed or rank == 0:
        wandb.init(
            project="data-efficient-contrastive-learning-linear-probe",
            config=args
        )

    if args.distributed:
        ddp_setup(rank, world_size, str(args.port))

    # Set all seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    Random(args.seed)

    print('==> Preparing data..')
    datasets = get_datasets(args.dataset)

    testloader = torch.utils.data.DataLoader(
        dataset=CustomSubsetDataset(datasets.testset, subset_indices=range(1000)), 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4, 
        pin_memory=True
    )
    clftrainloader = torch.utils.data.DataLoader(
        dataset=datasets.clftrainset, 
        batch_size=args.batch_size, 
        shuffle=not args.distributed,
        sampler=DistributedSampler(CustomSubsetDataset(datasets.clftrainset, subset_indices=range(1000)), shuffle=True) if args.distributed else None,
        num_workers=4, 
        pin_memory=True
    )

    ##############################################################
    # Model and Optimizer
    ##############################################################

    net = torch.load(args.encoder).to(device)

    clf = nn.Linear(net.representation_dim, datasets.num_classes).to(device)
    if args.distributed:
        clf = DDP(clf, device_ids=[device])

    criterion = nn.CrossEntropyLoss()
    clf_optimizer = optim.SGD(clf.parameters(), lr=args.lr, momentum=args.momentum, nesterov=args.nesterov,
                            weight_decay=args.weight_decay)

    ##############################################################
    # Train Function
    ##############################################################

    def train_clf(epoch):
        print('\nEpoch %d' % epoch)
        net.eval()
        clf.train()
        train_loss = 0
        t = tqdm(enumerate(clftrainloader), desc='Loss: **** ', total=len(clftrainloader), bar_format='{desc}{bar}{r_bar}')
        for batch_idx, (inputs, targets) in t:
            clf_optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)
            representation = net(inputs).detach()
            predictions = clf(representation)
            loss = criterion(predictions, targets)
            loss.backward()
            clf_optimizer.step()

            train_loss += loss.item()

            t.set_description('Loss: %.3f ' % (train_loss / (batch_idx + 1)))
        return train_loss
    
    ##############################################################
    # Main Loop
    ##############################################################     
    best_acc = 0
    for epoch in range(args.num_epochs):
        train_loss = train_clf(epoch)
        if not args.distributed or rank == 0:
            acc, _ = test_clf(testloader, device, net, clf)
            wandb.log(
                {
                    "test":
                    {
                        "acc": acc
                    },
                    "train":
                    {
                        "loss": train_loss
                    }
                },
                step=epoch
            )
            if acc > best_acc:
                best_acc = acc
    
    if not args.distributed or rank == 0:
        print("Best test accuracy", best_acc, "%")
        wandb.log(
            {
                "test":
                {
                    "best_acc": best_acc
                }
            }
        )
    destroy_process_group()

##############################################################
# Distributed Training Setup
##############################################################
def ddp_setup(rank: int, world_size: int, port: str):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

##############################################################
# Script Entry Point
##############################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Linear Probe')
    parser = argparse.ArgumentParser(description='Train downstream classifier with gradients.')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument("--momentum", default=0.9, type=float, help='SGD momentum')
    parser.add_argument("--batch-size", type=int, default=512, help='Training batch size')
    parser.add_argument("--num-epochs", type=int, default=2, help='Number of training epochs')
    parser.add_argument("--weight-decay", type=float, default=1e-6, help='Weight decay on the linear classifier')
    parser.add_argument("--nesterov", action="store_true", help="Turn on Nesterov style momentum")
    parser.add_argument("--encoder", type=str, default='ckpt.pth', help='Pretrained encoder')
    parser.add_argument('--temperature', type=float, default=0.5, help='InfoNCE temperature')
    parser.add_argument('--dataset', type=str, default=str(SupportedDatasets.CIFAR100.value), help='dataset',
                        choices=[x.value for x in SupportedDatasets])
    parser.add_argument('--device', type=int, default=-1, help="GPU number to use")
    parser.add_argument("--device-ids", nargs = "+", default = None, help = "Specify device ids if using multiple gpus")
    parser.add_argument('--port', type=int, default=random.randint(49152, 65535), help="free port to use")
    parser.add_argument('--seed', type=int, default=0, help="Seed for randomness")

    # Parse arguments
    args = parser.parse_args()

    # Arguments check and initialize global variables
    device = "cpu"
    device_ids = None
    distributed = False
    if torch.cuda.is_available():
        if args.device_ids is None:
            if args.device >= 0:
                device = args.device
            else:
                device = 0
        else:
            distributed = True
            device_ids = [int(id) for id in args.device_ids]
    args.device = device
    args.device_ids = device_ids
    args.distributed = distributed
    if distributed:
        mp.spawn(
            fn=main, 
            args=(len(device_ids), args),
            nprocs=len(device_ids)
        )
    else:
        main(device, 1, args)