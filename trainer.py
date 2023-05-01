from typing import List

from torch import Tensor, nn
import torch
from torch.optim import Optimizer
from torch.utils.data import BatchSampler, DataLoader
from tqdm import tqdm

from evaluate.lbfgs import encode_train_set, train_clf, test_clf
from projection_heads.critic import LinearCritic


class Trainer():
    def __init__(
        self,
        device: torch.device,
        net: nn.Module,
        critic: LinearCritic,
        trainloader: DataLoader,
        clftrainloader: DataLoader,
        testloader: DataLoader,
        num_classes: int,
        optimizer: Optimizer,
        lr_scheduler = None,
    ):
        """
        :param device: Device to run on (GPU)
        :param net: encoder network
        :param critic: projection head
        :param sampler: batch sampler (creates batches of specified kind)
        :param trainloader: dataloader for train data (for contrastive learning)
        :param encoder_optimizer: Optimizer for the encoder network (net)
        :param warm_up_epochs: Epochs up to which to use warm up batches
        :param is_scl: Flag to determine where supervised contrastive learning loss should be used
        :param rare_testloader: Test loader with rare classes upsampled (for imbalance dataset)
        """
        self.device = device
        self.net = net 
        self.critic = critic
        self.trainloader = trainloader
        self.clftrainloader = clftrainloader
        self.testloader = testloader
        self.num_classes = num_classes
        self.encoder_optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
        self.criterion = nn.CrossEntropyLoss()
        self.best_acc = 0
        self.best_rare_acc = 0

    #########################################
    #           Loss Functions              #
    #########################################
    def un_supcon_loss(self, z: List[Tensor]):
        sim = self.critic.compute_sim(z)
        log_sum_exp_sim = torch.log(torch.sum(torch.exp(sim), dim=1))

        # Positive Pairs Mask 
        p_targets = torch.cat([torch.tensor(range(int(len(sim) / len(z))))] * len(z))
        pos_pairs = (p_targets.unsqueeze(1) == p_targets.unsqueeze(0)).to(self.device)
        inf_mask = (sim != float('-inf')).to(self.device)
        pos_pairs = torch.logical_and(pos_pairs, inf_mask)
        pos_count = torch.sum(pos_pairs, dim=1)
        pos_sims = torch.nansum(sim * pos_pairs, dim=-1)
        return torch.mean(-pos_sims / pos_count + log_sum_exp_sim)
    
    #########################################
    #           Train & Test Modules        #
    #########################################
    def train(self, epoch):
        self.net.train()
        self.critic.train()

        # Traininig Loop (over batches in epoch)
        train_loss = 0
        t = tqdm(enumerate(self.trainloader), desc='Loss: **** ', total=len(self.trainloader), bar_format='{desc}{bar}{r_bar}')
        for batch_idx, inputs in t:
            x = inputs.to(self.device)
            self.encoder_optimizer.zero_grad()
            z = []
            for i in range(x.shape[1]):
                z.append(self.net(x[:, i, :, :, :]))

            loss = self.un_supcon_loss(z)
            loss.backward()

            self.encoder_optimizer.step()
            train_loss += loss.item()
            t.set_description('Loss: %.3f ' % (train_loss / (batch_idx + 1)))

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            print("lr:", self.scale_lr * self.lr_scheduler.get_last_lr()[0])
            
        return train_loss / len(self.trainloader)

    def test(self):
        X, y = encode_train_set(self.clftrainloader, self.device, self.net)
        clf = train_clf(X, y, self.net.representation_dim, self.num_classes, self.device, reg_weight=1e-5, iter=500)
        acc, _ = test_clf(self.testloader, self.device, self.net, clf)

        if acc > self.best_acc:
            self.best_acc = acc

        return acc