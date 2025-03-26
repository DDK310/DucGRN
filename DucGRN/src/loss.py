import numpy as np
import torch
import torch.nn.functional as F
def varga_loss_function(preds, labels, dis_real, dis_fake, mu, logvar, norm, weight, sparsity_weight=0.1):
    # 对抗变分图正则化图自编码损失：生成和判别的loss
    cost = norm * F.binary_cross_entropy(preds.view(-1), labels.to_dense().view(-1), weight=weight)
    KLD = 0.5 / preds.size(0) * (1 + 2 * logvar - mu ** 2 - torch.exp(logvar) ** 2).sum(1).mean()
    dis_real_loss = F.binary_cross_entropy(dis_real, torch.ones(dis_real.shape).to(DEVICE))
    dis_fake_loss = F.binary_cross_entropy(dis_fake, torch.zeros(dis_fake.shape).to(DEVICE))
    sparsity_loss = torch.mean(torch.abs(preds))
    #
    return cost - KLD+ dis_real_loss + dis_fake_loss +sparsity_weight * sparsity_loss