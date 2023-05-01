from datetime import datetime
import argparse
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sas

from configs import SupportedDatasets, get_datasets
from resnet import *
from projection_heads.critic import LinearCritic
from trainer import Trainer
from util import Random

parser = argparse.ArgumentParser(description='PyTorch Contrastive Learning.')
parser.add_argument('--temperature', type=float, default=0.5, help='InfoNCE temperature')
parser.add_argument("--batch-size", type=int, default=512, help='Training batch size')
parser.add_argument("--num-epochs", type=int, default=400, help='Number of training epochs')
parser.add_argument("--arch", type=str, default='resnet18', help='Encoder architecture',
                    choices=['resnet10', 'resnet18', 'resnet34', 'resnet50'])
parser.add_argument("--num-workers", type=int, default=2, help='Number of threads for data loaders')
parser.add_argument("--test-freq", type=int, default=10, help='Frequency to fit a linear clf with L-BFGS for testing'
                                                              'Not appropriate for large datasets. Set 0 to avoid '
                                                              'classifier only training here.')
parser.add_argument("--checkpoint-freq", type=int, default=10000, help="How often to checkpoint model")
parser.add_argument('--dataset', type=str, default=str(SupportedDatasets.CIFAR10.value), help='dataset',
                    choices=[x.value for x in SupportedDatasets])
parser.add_argument('--load-subset-indices', type=str, help="Path to subset indices")
parser.add_argument('--device', type=int, default=-1, help="GPU number to use")
parser.add_argument('--seed', type=int, default=0, help="Seed for randomness")

# Parse arguments
args = parser.parse_args()

# Set all seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)
Random(args.seed)

# Arguments check and initialize global variables
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  
start_epoch = 0  

print('==> Preparing data..')
datasets = get_datasets(args.dataset, imbalance=args.imbalance)

##############################################################
# Load Subset Indices
##############################################################

with open(args.load_subset_indices, "rb") as f:
    subset_indices = pickle.load(f)
trainset = sas.SubsetDataset.CustomSubset(
    trainset=datasets.trainset,
    subset_indices=subset_indices
)
print("subset_size:", len(subset_indices))

##############################################################
# Data Loaders
##############################################################

trainloader = torch.utils.data.DataLoader(
    datasets.trainset, 
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True
)
testloader = torch.utils.data.DataLoader(datasets.testset, batch_size=1000, shuffle=False, num_workers=args.num_workers, pin_memory=True)
clftrainloader = torch.utils.data.DataLoader(datasets.clftrainset, batch_size=4096, shuffle=False, num_workers=args.num_workers, pin_memory=True)

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
net = net.to(device)

##############################################################
# Critic
##############################################################

critic = LinearCritic(net.representation_dim, temperature=args.temperature).to(device)

# Initialize Loss Function, Optimizer and LR Scheduler
criterion = nn.CrossEntropyLoss()

# DCL Setup
lr_scheduler = None
optimizer = optim.Adam(list(net.parameters()) + list(critic.parameters()), lr=1e-3, weight_decay=1e-6)
if args.dataset == SupportedDatasets.TINY_IMAGENET.value:
    optimizer = optim.Adam(list(net.parameters()) + list(critic.parameters()), lr=2e-3, weight_decay=1e-6)
    

##############################################################
# Main Loop (Train, Test)
##############################################################

# Initialize Trainer Object
trainer = Trainer(
    device=device,
    net=net,
    critic=critic,
    trainloader=trainloader,
    clftrainloader=clftrainloader,
    testloader=testloader,
    num_classes=datasets.num_classes,
    optimizer=optimizer,
)

# Date Time String
DT_STRING = "".join(str(datetime.now()).split())

# Iterate over epochs
for epoch in range(start_epoch, start_epoch + args.num_epochs):
    print(f"step: {epoch}")

    train_loss = trainer.train(epoch)
    print(f"train_loss: {train_loss}")

    if (epoch + 1) % args.test_freq == 0:
        test_acc = trainer.test()
        print(f"test_acc: {test_acc}")

    # Checkpoint Model
    if (epoch + 1) % args.checkpoint_freq == 0:
        scl = "scl" if args.is_scl else ""
        torch.save(net, f"checkpoints/{DT_STRING}-{args.dataset}-{args.arch}-{epoch}-net.pt")
        torch.save(critic, f"checkpoints/{DT_STRING}-{args.dataset}-{args.arch}-{epoch}-critic.pt")
    
print(f"best_test_acc: {trainer.best_acc}")
