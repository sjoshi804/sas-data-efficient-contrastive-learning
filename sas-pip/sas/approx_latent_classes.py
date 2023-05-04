from copy import deepcopy
from typing import List

import clip
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from fast_pytorch_kmeans import KMeans
from tqdm import tqdm

def clip_approx(
    img_trainset: torch.utils.data.Dataset,
    labeled_example_indices: List[int], 
    labels: np.array,
    num_classes: int,
    device: torch.device, 
    batch_size: int = 512,
    verbose: bool = False,
):
    Z = encode_using_clip(
        img_trainset=img_trainset,
        device=device,
        verbose=verbose
    )
    clf = train_linear_classifier(
        X=Z[labeled_example_indices], 
        y=torch.tensor(labels), 
        representation_dim=len(Z[0]),
        num_classes=num_classes,
        device=device,
        verbose=verbose
    )
    preds = []
    for start_idx in range(0, len(Z), batch_size):
        preds.append(torch.argmax(clf(Z[start_idx:start_idx + batch_size]).detach(), dim=1).cpu())
    preds = torch.cat(preds).numpy()

    return partition_from_preds(preds)

def clip_0shot_approx(
    img_trainset: torch.utils.data.Dataset,
    class_names: List[str],
    device: torch.device,
    verbose: bool = False,
):
    model, preprocess = clip.load("ViT-B/32")
    img_trainset = deepcopy(img_trainset)
    img_trainset.transform = preprocess
    model = model.to(device)

    zeroshot_weights = zeroshot_classifier(
        class_names=class_names,
        device=device,
        verbose=verbose
    )
    logits = []
    loader = torch.utils.data.DataLoader(img_trainset, batch_size=32, num_workers=2)
    with torch.no_grad():
        for input in tqdm(loader, "0-shot classification using provided text names for classes", disable=not verbose):
            # predict
            image_features = model.encode_image(input[0].to(device=device))
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits.append(100. * image_features @ zeroshot_weights)

    preds = []
    for logit in logits:
        preds.append(logit.topk(1, 1, True, True)[1].t()[0])

    return partition_from_preds(preds)

def kmeans_approx(
    trainset: torch.utils.data.Dataset,
    proxy_model: nn.Module,
    num_classes: int,
    device: torch.device,
    verbose: bool = False
):
    proxy_model.eval()
    Z = []
    with torch.no_grad():
        loader = torch.utils.data.DataLoader(trainset, batch_size=32, num_workers=2)
        for input in tqdm(loader, "Encoding data using proxy model provided", disable=not verbose):
            Z.append(proxy_model(input[0].to(device)))
    Z = torch.cat(Z, dim=0).to("cpu")
    kmeans = KMeans(n_clusters=num_classes, mode='euclidean', verbose=int(verbose), max_iter=1000)
    preds = kmeans.fit_predict(Z).cpu().numpy()
    return partition_from_preds(preds)

def encode_using_clip(
        img_trainset: torch.utils.data.Dataset,
        device: torch.device,
        verbose: bool = False,
):
    model, preprocess = clip.load("ViT-B/32")
    model = model.to(device)
    img_trainset = deepcopy(img_trainset)
    img_trainset.transform = preprocess

    loader = torch.utils.data.DataLoader(img_trainset, batch_size=32, num_workers=2)
    Z = []
    with torch.no_grad():
        for input in tqdm(loader, desc="Encoding images using CLIP", disable=not verbose):
            Z.append(model.encode_image(input[0].to(device)))
    Z = torch.cat(Z, dim=0).to(torch.float32)
    return Z

def partition_from_preds(preds):
    partition = {}
    for i, pred in enumerate(preds):
        if pred not in partition:
            partition[pred] = []
        partition[pred].append(i)
    return partition

def train_linear_classifier(
    X: torch.tensor, 
    y: torch.tensor, 
    representation_dim: int,
    num_classes: int,
    device: torch.device,
    reg_weight: float = 1e-3,
    n_lbfgs_steps: int = 500,
    verbose=False,
):
    if verbose:
        print('\nL2 Regularization weight: %g' % reg_weight)

    criterion = nn.CrossEntropyLoss()
    X_gpu = X.to(device)
    y_gpu = y.to(device)

    # Should be reset after each epoch for a completely independent evaluation
    clf = nn.Linear(representation_dim, num_classes).to(device)
    clf_optimizer = optim.LBFGS(clf.parameters())
    clf.train()

    for _ in tqdm(range(n_lbfgs_steps), desc="Training linear classifier using fraction of labels", disable=not verbose):
        def closure():
            clf_optimizer.zero_grad()
            raw_scores = clf(X_gpu)
            loss = criterion(raw_scores, y_gpu)
            loss += reg_weight * clf.weight.pow(2).sum()
            loss.backward()
            return loss
        clf_optimizer.step(closure)
    return clf

def zeroshot_classifier(
    class_names: List[str],
    device: torch.device,
    verbose: bool = False
):
    templates = [
        'itap of the {}.',
        'a bad photo of the {}',
        'a origami {}.',
        'a photo of the large {}.',
        'a {} in a video game.',
        'art of the {}.',
        'a photo of the small {}.',
    ]
    
    model, _ = clip.load("ViT-B/32")
    model = model.to(device)

    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(class_names, desc="Creating zero shot classifier", disable=not verbose):
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).to(device) #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights