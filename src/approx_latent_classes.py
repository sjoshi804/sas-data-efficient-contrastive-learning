from typing import List

from fast_pytorch_kmeans import KMeans
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
    Z = encode_using_clip(img_trainset)
    clf = train_linear_classifier(
        X=Z[labeled_examples[0]], 
        y=labeled_examples[1], 
        num_classes=num_classes,
        device=device
    )
    preds = clf(Z).detach().cpu().numpy()
    return partition_from_preds(preds)

def clip_0shot_approx(
    img_trainset: torch.utils.data.Dataset,
    class_names: List[str],
    device: torch.device
):
    model, preprocess = clip.load("ViT-B/32")

    zeroshot_weights = zeroshot_classifier(class_names)
    logits = []
    loader = torch.utils.data.DataLoader(img_trainset, batch_size=32, num_workers=2, transforms=preprocess)
    with torch.no_grad():
        for input in loader:
            # predict
            image_features = model.encode_image(input[0].to(device=device))
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits.append(100. * image_features @ zeroshot_weights)

    preds = []
    for logit in logits:
        preds.append(logit.topk(1, 1, True, True)[1].t()[0])

    return partition_from_preds(preds)

def kmeans_approx(
    img_trainset: torch.utils.data.Dataset,
    proxy_model: nn.Module,
    num_classes: int,
    device: torch.device
):
    proxy_model.eval()
    Z = []
    with torch.no_grad():
        loader = torch.utils.data.DataLoader(img_trainset, batch_size=32, num_workers=2)
        for input in loader:
            Z.append(proxy_model(input.to(device)))
    Z = torch.cat(Z, dim=0).to(device=device)

    kmeans = KMeans(n_clusters=num_classes, mode='euclidean', verbose=0, max_iter=1000)
    preds = kmeans.fit_predict(Z).cpu().numpy()
    return partition_from_preds(preds)


def encode_using_clip(img_trainset, device):
    model, preprocess = clip.load("ViT-B/32")
    loader = torch.utils.data.DataLoader(img_trainset, batch_size=32, num_workers=2, transforms=preprocess)
    Z = []
    with torch.no_grad():
        for input in loader:
            Z.append(model.encode_image(input[0].to(device)))
    return torch.cat(Z, dim=0)

def partition_from_preds(preds):
    partition = {}
    for i in enumerate(preds):
        if preds[i] not in partition:
            partition[preds[i]] = []
        partition[preds[i]].append(i)
    return partition

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

def zeroshot_classifier(class_names):
    templates = [
        'itap of the {}.',
        'a bad photo of the {}',
        'a origami {}.',
        'a photo of the large {}.',
        'a {} in a video game.',
        'art of the {}.',
        'a photo of the small {}.',
    ]
    model, preprocess = clip.load("ViT-B/32")
    with torch.no_grad():
        zeroshot_weights = []
        for classname in class_names:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights