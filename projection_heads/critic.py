import torch
from torch import nn
from util import Random

class LinearCritic(nn.Module):

    def __init__(self, latent_dim, temperature=1., num_negatives = -1):
        super(LinearCritic, self).__init__()
        self.temperature = temperature
        self.projection_dim = 128
        self.w1 = nn.Linear(latent_dim, latent_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(latent_dim)
        self.relu = nn.ReLU()
        self.w2 = nn.Linear(latent_dim, self.projection_dim, bias=False)
        self.bn2 = nn.BatchNorm1d(self.projection_dim, affine=False)
        self.cossim = nn.CosineSimilarity(dim=-1)
        self.num_negatives = num_negatives

    def project(self, h):
        return self.bn2(self.w2(self.relu(self.bn1(self.w1(h)))))

    def compute_sim(self, z):
        p = []
        for i in range(len(z)):
            p.append(self.project(z[i]))
        
        sim = {}
        for i in range(len(p)):
            for j in range(i, len(p)):
                sim[(i, j)] = self.cossim(p[i].unsqueeze(-2), p[j].unsqueeze(-3)) / self.temperature

        d = sim[(0,1)].shape[-1]
        for i in range(len(p)):
            sim[(i,i)][..., range(d), range(d)] = float('-inf')   

        for i in range(len(p)):
            sim[i] = torch.cat([sim[(j, i)].transpose(-1, -2) for j in range(0, i)] + [sim[(i, j)] for j in range(i, len(p))], dim=-1)
        sim = torch.cat([sim[i] for i in range(len(p))], dim=-2)
        
        return sim

    def forward(self, z1, z2):
        raw_scores = self.compute_sim(z1, z2)
        targets = torch.arange(2 * (z1.shape[0]), dtype=torch.long, device=raw_scores.device)
        return raw_scores, targets
