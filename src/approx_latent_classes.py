from collections import namedtuple
from typing import List

import clip
import torch
import torch.nn as nn
import torch.optim as optim


def clip_approx(
    img_trainset: torch.utils.data.Dataset,
    labeled_examples: tuple(List[int], torch.tensor),
    num_classes: int,
    device: torch.device
):
    pass 

def clip_0shot_approx(
    img_trainset: torch.utils.data.Dataset,
    class_names: List[str],
    device: torch.device
):
    pass 

def kmeans_approx(
    img_trainset: torch.utils.data.Dataset,
    proxy_model: nn.Module,
    num_classes: int,
    device: torch.device
):
    pass 

def encode_using_clip(trainloader):
    pass

def train_linear_classifier(
    X: torch.tensor, 
    y: torch.tensor, 
    representation_dim: int,
    num_classes: int,
    device: torch.device,
    reg_weight: float = 1e-3,
    n_lbfgs_steps: int = 500
):
    print('\nL2 Regularization weight: %g' % reg_weight)

    criterion = nn.CrossEntropyLoss()
    X_gpu = X.to(device)
    y_gpu = y.to(device)

    # Should be reset after each epoch for a completely independent evaluation
    clf = nn.Linear(representation_dim, num_classes).to(device)
    clf_optimizer = optim.LBFGS(clf.parameters())
    clf.train()

    for _ in range(n_lbfgs_steps):
        def closure():
            clf_optimizer.zero_grad()
            raw_scores = clf(X_gpu)
            loss = criterion(raw_scores, y_gpu)
            loss += reg_weight * clf.weight.pow(2).sum()
            loss.backward()
            return loss
        clf_optimizer.step(closure)
    return clf