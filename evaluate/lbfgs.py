import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


def encode_train_set(clftrainloader, device, net):
    net.eval()

    store = []
    with torch.no_grad():
        t = tqdm(enumerate(clftrainloader), desc='Encoded: **/** ', total=len(clftrainloader),
                 bar_format='{desc}{bar}{r_bar}')
        for batch_idx, (inputs, targets) in t:
            inputs, targets = inputs.to(device), targets.to(device)
            representation = net(inputs)
            store.append((representation, targets))

            t.set_description('Encoded %d/%d' % (batch_idx, len(clftrainloader)))

    X, y = zip(*store)
    X, y = torch.cat(X, dim=0), torch.cat(y, dim=0)
    return X, y

def encode_train_set_w_augmentations(trainloader, device, net, critic=None, num_pos = 1):
    net.eval()

    X = []
    for _ in range(num_pos):
        X.append([])
    y = []

    with torch.no_grad():
        t = tqdm(enumerate(trainloader), desc='Encoded: **/** ', total=len(trainloader),
                 bar_format='{desc}{bar}{r_bar}')
        for batch_idx, (input, targets, _) in t:
            targets = targets.to(device)
            for i in range(num_pos):
                x = None
                if num_pos > 2:
                    x = input[:, i, :, :, :].to(device)
                else:
                    x = input[i].to(device)
                if critic is not None:
                    X[i].append(critic.project(net(x)))
                else:
                    X[i].append(net(x))
            y.append(targets)
            t.set_description('Encoded %d/%d' % (batch_idx, len(trainloader)))

    y = torch.cat(y, dim=0)
    for i, X_i in enumerate(X):
        X[i] = torch.cat(X_i, dim=0)
    return X, y

def encode_train_set_projection(trainloader, device, net, critic):
    net.eval()
    critic.eval()
    store = []
    with torch.no_grad():
        t = tqdm(enumerate(trainloader), desc='Encoded: **/** ', total=len(trainloader),
                 bar_format='{desc}{bar}{r_bar}')
        for batch_idx, train_tuple in t:
            x1 = train_tuple[0]
            x1 = x1.to(device)
            representation = critic.project(net(x1))
            store.append(representation)
            t.set_description('Encoded Projections %d/%d' % (batch_idx, len(trainloader)))
    X = torch.cat(store, dim=0)
    return X

def train_clf(X, y, representation_dim, num_classes, device, reg_weight=1e-3, iter=500):
    print('\nL2 Regularization weight: %g' % reg_weight)

    criterion = nn.CrossEntropyLoss()
    n_lbfgs_steps = iter

    # Should be reset after each epoch for a completely independent evaluation
    clf = nn.Linear(representation_dim, num_classes).to(device)
    clf_optimizer = optim.LBFGS(clf.parameters())
    clf.train()

    t = tqdm(range(n_lbfgs_steps), desc='Loss: **** | Train Acc: ****% ', bar_format='{desc}{bar}{r_bar}')
    for _ in t:
        def closure():
            clf_optimizer.zero_grad()
            raw_scores = clf(X)
            loss = criterion(raw_scores, y)
            loss += reg_weight * clf.weight.pow(2).sum()
            loss.backward()

            _, predicted = raw_scores.max(1)
            correct = predicted.eq(y).sum().item()

            t.set_description('Loss: %.3f | Train Acc: %.3f%% ' % (loss, 100. * correct / y.shape[0]))

            return loss

        clf_optimizer.step(closure)

    return clf


def test_clf(testloader, device, net, clf):
    criterion = nn.CrossEntropyLoss()
    net.eval()
    clf.eval()
    test_clf_loss = 0
    correct = 0
    total = 0
    acc_per_point = []
    with torch.no_grad():
        t = tqdm(enumerate(testloader), total=len(testloader), desc='Loss: **** | Test Acc: ****% ',
                 bar_format='{desc}{bar}{r_bar}')
        for batch_idx, (inputs, targets) in t:
            inputs, targets = inputs.to(device), targets.to(device)
            representation = net(inputs)
            # test_repr_loss = criterion(representation, targets)
            raw_scores = clf(representation)
            clf_loss = criterion(raw_scores, targets)
            test_clf_loss += clf_loss.item()
            _, predicted = raw_scores.max(1)
            total += targets.size(0)
            acc_per_point.append(predicted.eq(targets))
            correct += acc_per_point[-1].sum().item()
            t.set_description('Loss: %.3f | Test Acc: %.3f%% ' % (test_clf_loss / (batch_idx + 1), 100. * correct / total))
            
    acc = 100. * correct / total
    return acc

def top5accuracy(output, target, topk=(5,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        print(correct)
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res