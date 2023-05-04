import argparse
import os
import pickle
from datetime import datetime
import random

import numpy as np
import sas.subset_dataset
import torch
import torch.multiprocessing as mp
import torch.optim as optim
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb

from configs import SupportedDatasets, get_datasets
from projection_heads.critic import LinearCritic
from resnet import *
from trainer import Trainer
from util import Random

def main(rank: int, world_size: int, args):

    # Determine Device 
    device = rank
    if args.distributed:
        device = args.device_ids[rank]
        torch.cuda.set_device(args.device_ids[rank])
        args.lr *= world_size

    # WandB Logging
    if not args.distributed or rank == 0:
        wandb.init(
            project="data-efficient-contrastive-learning",
            config=args
        )

    if args.distributed:
        args.batch_size = int(args.batch_size / world_size)

    # Set all seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    Random(args.seed)

    print('==> Preparing data..')
    datasets = get_datasets(args.dataset)

    ##############################################################
    # Load Subset Indices
    ##############################################################

    if args.random_subset:
        trainset = sas.subset_dataset.RandomSubsetDataset(
            dataset=datasets.trainset,
            subset_fraction=args.subset_fraction
        )
    elif args.subset_indices != "":
        with open(args.subset_indices, "rb") as f:
            subset_indices = pickle.load(f)
        trainset = sas.subset_dataset.CustomSubsetDataset(
            dataset=datasets.trainset,
            subset_indices=subset_indices
        )
    else:
        trainset = datasets.trainset
    print("subset_size:", len(trainset))

    # Model
    print('==> Building model..')

    ##############################################################
    # Encoder
    ##############################################################

    if args.arch == 'resnet10':
        net = ResNet10(stem=datasets.stem)
    elif args.arch == 'resnet18':
        net = ResNet18(stem=datasets.stem)
    elif args.arch == 'resnet34':
        net = ResNet34(stem=datasets.stem)
    elif args.arch == 'resnet50':
        net = ResNet50(stem=datasets.stem)
    else:
        raise ValueError("Bad architecture specification")

    ##############################################################
    # Critic
    ##############################################################

    critic = LinearCritic(net.representation_dim, temperature=args.temperature).to(device)

    # DCL Setup
    optimizer = optim.Adam(list(net.parameters()) + list(critic.parameters()), lr=args.lr, weight_decay=1e-6)
    if args.dataset == SupportedDatasets.TINY_IMAGENET.value:
        optimizer = optim.Adam(list(net.parameters()) + list(critic.parameters()), lr=2 * args.lr, weight_decay=1e-6)
        

    ##############################################################
    # Data Loaders
    ##############################################################

    trainloader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=DistributedSampler(trainset, shuffle=True, num_replicas=world_size, rank=rank, drop_last=True) if args.distributed else None,
        num_workers=4,
        pin_memory=True,
    )

    clftrainloader = torch.utils.data.DataLoader(
        dataset=datasets.clftrainset,
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    testloader = torch.utils.data.DataLoader(
        dataset=datasets.testset,
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
    )
    
    ##############################################################
    # Main Loop (Train, Test)
    ##############################################################

    # Date Time String
    DT_STRING = "".join(str(datetime.now()).split())

    if args.distributed:
        ddp_setup(rank, world_size, str(args.port))

    net = net.to(device)
    critic = critic.to(device)
    if args.distributed:
        net = DDP(net, device_ids=[device])

    trainer = Trainer(
        device=device,
        distributed=args.distributed,
        rank=rank if args.distributed else 0,
        world_size=world_size,
        net=net,
        critic=critic,
        trainloader=trainloader,
        clftrainloader=clftrainloader,
        testloader=testloader,
        num_classes=datasets.num_classes,
        optimizer=optimizer,
    )

    for epoch in range(0, args.num_epochs):
        print(f"step: {epoch}")

        train_loss = trainer.train()
        print(f"train_loss: {train_loss}")
        if not args.distributed or rank == 0:
            wandb.log(
                data={"train": {
                "loss": train_loss,
                }},
                step=epoch
            )

        if (not args.distributed or rank == 0) and ((epoch + 1) % args.test_freq == 0):
            test_acc = trainer.test()
            print(f"test_acc: {test_acc}")
            wandb.log(
                data={"test": {
                "acc": test_acc,
                }},
                step=epoch
            )

        # Checkpoint Model
        if ((not args.distributed or rank == 0) and (epoch + 1) % args.checkpoint_freq == 0):
            trainer.save_checkpoint(prefix=f"{DT_STRING}-{args.dataset}-{args.arch}-{epoch}")

    if not args.distributed or rank == 0:
        print(f"best_test_acc: {trainer.best_acc}")
        wandb.log(
            data={"test": {
            "best_acc": trainer.best_acc,
            }}
        )
        wandb.finish(quiet=True)

    if args.distributed:
        destroy_process_group()


##############################################################
# Distributed Training Setup
##############################################################
def ddp_setup(rank: int, world_size: int, port: str):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Contrastive Learning.')
    parser.add_argument('--temperature', type=float, default=0.5, help='InfoNCE temperature')
    parser.add_argument("--batch-size", type=int, default=512, help='Training batch size')
    parser.add_argument("--lr", type=float, default=1e-3, help='learning rate')
    parser.add_argument("--num-epochs", type=int, default=400, help='Number of training epochs')
    parser.add_argument("--arch", type=str, default='resnet18', help='Encoder architecture',
                        choices=['resnet10', 'resnet18', 'resnet34', 'resnet50'])
    parser.add_argument("--test-freq", type=int, default=10, help='Frequency to fit a linear clf with L-BFGS for testing'
                                                                'Not appropriate for large datasets. Set 0 to avoid '
                                                                'classifier only training here.')
    parser.add_argument("--checkpoint-freq", type=int, default=10000, help="How often to checkpoint model")
    parser.add_argument('--dataset', type=str, default=str(SupportedDatasets.CIFAR100.value), help='dataset',
                        choices=[x.value for x in SupportedDatasets])
    parser.add_argument('--subset-indices', type=str, default="", help="Path to subset indices")
    parser.add_argument('--random-subset', action="store_true", help="Random subset")
    parser.add_argument('--subset_fraction', type=float, )
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