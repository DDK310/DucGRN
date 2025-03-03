import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import numpy as np
from torch_geometric.nn import GCNConv
#from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch_geometric.nn.pool.connect import FilterEdges
from torch_geometric.nn.pool.select import SelectTopK
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch.nn import Parameter
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import coalesce
from torch_geometric.utils import dropout_adj, subgraph
import args

DEVICE = torch.device("cuda")


class KHopGraphAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, k, dropout, alpha):
        super(KHopGraphAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k = k
        self.weight = Parameter(torch.FloatTensor(input_dim, output_dim))
        self.a = Parameter(torch.FloatTensor(2 * output_dim, 1))
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.relu = F.relu

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_normal_(self.a)

    def get_k_hop_neighbors(self, adj):
        hop_neighbors = adj
        cumulative_neighbors = adj.clone()  # 初始化累加矩阵为adj的克隆

        for _ in range(self.k - 1):
            hop_neighbors = hop_neighbors @ adj
            cumulative_neighbors += hop_neighbors  # 累加本次计算结果

        return cumulative_neighbors

    def forward(self, input, adj):
        Wh = torch.mm(input, self.weight)  # (N, output_dim) = (N, input_dim) * (input_dim, output_dim)
        Wh1 = torch.matmul(Wh, self.a[:self.output_dim, :])  # (N, 1) = (N, output_dim) * (out_dim, 1)
        Wh2 = torch.matmul(Wh, self.a[self.output_dim:, :])  # (N, 1) = (N, output_dim) * (out_dim, 1)
        e = Wh1 + Wh2.T  # (N, N)
        e = self.leakyrelu(e)

        k_hop_adj = self.get_k_hop_neighbors(adj)  # 获取k跳邻居
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(k_hop_adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        attention_wh = torch.matmul(attention, Wh)  # (N, output_dim) = (N, N) * (N, output_dim)
        return self.relu(attention_wh)


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim , dropout):
        super(GraphConvolution, self).__init__()
        self.weight = glorot_init(input_dim, output_dim)
        self.relu = F.relu
        self.dropout = nn.Dropout(dropout)
    def forward(self, input, adj):
        x = input
        support = torch.mm(x, self.weight)
        x = torch.mm(adj, support)
        output = self.relu(x)
        return output

class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, input_dim, output_dim, scales, dropout):
        super(MultiScaleFeatureFusion, self).__init__()
        self.scales = scales
        self.convs = nn.ModuleList([GraphConvolution(input_dim, output_dim, dropout) for _ in range(scales)])
        self.fc = nn.Linear(scales * output_dim, output_dim)

    def forward(self, x, adj):
        multi_scale_features = [conv(x, adj) for conv in self.convs]
        fused_features = torch.cat(multi_scale_features, dim=-1)
        output = self.fc(fused_features)
        return output

class FusionEnhancedResidualGCN (nn.Module):
    def __init__(self, input_dim, output_dim, dropout, scales=3):
        super(FusionEnhancedResidualGCN , self).__init__()
        self.multi_scale_fusion = MultiScaleFeatureFusion(input_dim, output_dim, scales, dropout)
        if input_dim != output_dim:
            self.residual = nn.Linear(input_dim, output_dim)
        else:
            self.residual = None

    def forward(self, input, adj):
        multi_scale_output = self.multi_scale_fusion(input, adj)
        if self.residual:
            residual_output = self.residual(input)
        else:
            residual_output = input
        return multi_scale_output + residual_output


class NHGATModelGAN(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, hidden_dim3, num_heads, dropout, alpha, k=2,
                 vae_bool=True, pooling_ratio=0.5):
        super(NHGATModelGAN, self).__init__()
        self.vae_bool = vae_bool
        self.n_heads_attentions = nn.ModuleList(
            [KHopGraphAttentionLayer(input_feat_dim, hidden_dim1, k, dropout=dropout, alpha=alpha).to(DEVICE) for _ in
             range(num_heads)])
        self.gc_att = KHopGraphAttentionLayer(num_heads * hidden_dim1, hidden_dim1, k, dropout=dropout, alpha=alpha).to(
            DEVICE)
        self.gc1 = FusionEnhancedResidualGCN(hidden_dim1, hidden_dim2, dropout).to(DEVICE)
        self.gc2 = FusionEnhancedResidualGCN(hidden_dim2, hidden_dim3, dropout).to(DEVICE)
        self.gc3 = FusionEnhancedResidualGCN(hidden_dim2, hidden_dim3, dropout).to(DEVICE)

        #self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout).to(DEVICE)
        #self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout).to(DEVICE)
        #self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout).to(DEVICE)

        self.discriminator = Discriminator(hidden_dim3, hidden_dim3).to(DEVICE)
        self.ip = InnerProductDecoder(dropout).to(DEVICE)
        self.relu = F.relu
        self.to(DEVICE)

    def encode(self, input, adj):
        nheads_att_out = torch.cat([attention(input, adj.to_dense()) for attention in self.n_heads_attentions], dim=-1)
        gc_att = self.gc_att(nheads_att_out, adj.to_dense())
        hidden1 = self.gc1(gc_att, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.vae_bool:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)  # 乘std加mu
        else:
            return mu

    def forward(self, input, adj):
        mu, logvar = self.encode(input, adj)
        z_fake = self.reparameterize(mu, logvar)  # 得到z_fake
        z_real = torch.randn(z_fake.shape).to(DEVICE)  # 得到高斯分布的z_real
        dis_real = torch.sigmoid(self.discriminator(z_real))
        dis_fake = torch.sigmoid(self.discriminator(z_fake))
        pred = self.ip(z_fake)
        return pred, dis_real, dis_fake, mu, logvar  # ip(z_fake)生成邻接矩阵

class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout, vae_bool=True):
        super(GCNModelVAE, self).__init__()
        self.vae_bool = vae_bool
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout)
        self.ip = InnerProductDecoder(dropout)
        self.relu = nn.ReLU()

    def encode(self, input, adj):
        hidden1 = self.relu(self.gc1(input, adj))
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.vae_bool:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)  # 乘std加mu
        else:
            return mu

    def forward(self, input, adj):
        mu, logvar = self.encode(input, adj) #两个GCN分别得到mean和std
        z = self.reparameterize(mu, logvar) #得到z
        return self.ip(z), mu, logvar

class InnerProductDecoder(nn.Module):
    '''
    内积用来做decoder，用来生成邻接矩阵
    '''
    def __init__(self, dropout):
        super(InnerProductDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, z):
        z = self.dropout(z)
        #z=torch.sigmoid(z)
        adj = torch.sigmoid(torch.mm(z, z.t()))
        return adj

class Discriminator(nn.Module):
    # 判别器
    def __init__(self, hidden_dim2, hidden_dim1):
        super(Discriminator, self).__init__()

        self.fc = nn.Sequential(nn.Linear(hidden_dim2, hidden_dim1),
                                nn.ReLU(),
                                nn.Linear(hidden_dim1, hidden_dim2),
                                nn.ReLU(),
                                nn.Linear(hidden_dim2, 1))

    def forward(self, z):
        return self.fc(z)



def glorot_init(input_dim, output_dim):
	init_range = np.sqrt(6.0/(input_dim + output_dim))
	initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
	return nn.Parameter(initial)

