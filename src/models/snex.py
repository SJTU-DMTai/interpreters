import torch
import dgl
import copy
import random
import torch.nn as nn
import torch.nn.functional as F
from .graph_model import GraphModel


class ExtractorMLP(nn.Module):
    def __init__(self, in_dim, bias=True):
        super().__init__()
        hid_dim = in_dim * 2
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_dim, hid_dim, bias), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(hid_dim, 1, bias)
        )

    def forward(self, emb):
        att_log_logits = self.feature_extractor(emb)
        return att_log_logits


class SNex(GraphModel):
    def __init__(self, encoder, extractor, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 base_model="LSTM", use_residual=True, is_homogeneous=True):
        super().__init__(d_feat=d_feat, hidden_size=hidden_size, num_layers=num_layers,
                         dropout=dropout, base_model=base_model, use_residual=use_residual,
                         is_homogeneous=is_homogeneous)
        # GNN encoder
        # self.pre_encoder = pre_encoder
        self.encoder = encoder
        self.extractor = extractor
        self.proj_head = nn.Sequential(nn.Linear(64, 16), nn.ReLU(), nn.Dropout(dropout))
        self.sparsity_loss_coef = 0.5
        self.sparsity_mask_coef = 0.01
        self.sparsity_ent_coef = 0.1
        self.cts_loss_coef = 0.1
        self.temp = 1
        self.k = 0.05
        self.n_pos, self.n_neg = 5, 5
        self.pool = dgl.nn.SumPooling()
        self.max_topk = 0.5
        self.min_topk = 0.1
        self.max_epoch = 30

    def sampling(self, att_log_logit, training):
        # concrete_sample
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gated_input = ((att_log_logit + random_noise) / self.temp).sigmoid()
        else:
            gated_input = att_log_logit.sigmoid()
        att_bern = gated_input
        return att_bern

    def loss(self, anchor_emb, edge_mask, epoch, eps=1e-6):
        sparsity = 0.
        ent = -edge_mask * torch.log(edge_mask + eps) - (1 - edge_mask) * torch.log(1 - edge_mask + eps)
        sparsity += self.sparsity_mask_coef * edge_mask.mean()
        sparsity += self.sparsity_ent_coef * ent.mean()
        cts_loss = self.cts_loss(anchor_emb, edge_mask, epoch)
        loss = self.sparsity_loss_coef * sparsity + self.cts_loss_coef * cts_loss
        return loss

    def get_cts_mask(self, edge_mask, topk, k):
        edge_mask = edge_mask.detach().view(-1)
        num_edges = edge_mask.shape[0]
        num_pos = int(num_edges * topk)
        num_neg = num_edges - num_pos
        _, pos_idx = torch.topk(edge_mask, num_pos)
        _, neg_idx = torch.topk(-edge_mask, num_neg)
        pos_avg = edge_mask[pos_idx].mean()
        neg_avg = edge_mask[neg_idx].mean()

        perturb_e_num = int(num_edges * k)
        if perturb_e_num <= 0:
            perturb_e_num = 1
        # positive samples
        pos_edge_mask = torch.tensor([]).to(edge_mask.device)
        for _ in range(self.n_pos):
            m = copy.deepcopy(edge_mask)
            idx = neg_idx[torch.randperm(num_neg)[:perturb_e_num]]
            m[idx] = pos_avg
            pos_edge_mask = torch.cat([pos_edge_mask, m])
        # negative samples
        neg_edge_mask = torch.tensor([]).to(edge_mask.device)
        for _ in range(self.n_neg):
            m = copy.deepcopy(edge_mask)
            idx = pos_idx[torch.randperm(num_pos)[:perturb_e_num]]
            m[idx] = neg_avg
            neg_edge_mask = torch.cat([neg_edge_mask, m])
        return pos_edge_mask.view(self.n_pos, -1, 1), neg_edge_mask.view(self.n_neg, -1, 1)

    def cts_loss(self, anchor_emb, edge_mask, epoch, tau=0.1):
        # topk = 0.3
        topk = (self.max_topk - (self.max_topk - self.min_topk) * epoch / self.max_epoch)
        with torch.no_grad():
            pos_edge_mask, neg_edge_mask = self.get_cts_mask(edge_mask, topk, self.k)
        pos_emb = torch.tensor([]).to(edge_mask.device)
        for i in range(self.n_pos):
            emb = self.encoder.forward_graph(h=self.h, index=self.index, edge_weight=pos_edge_mask[i])
            emb = self.pool(self.sub_g, self.proj_head(emb))
            pos_emb = torch.cat([pos_emb, emb])
        neg_emb = torch.tensor([]).to(edge_mask.device)
        for i in range(self.n_neg):
            emb = self.encoder.forward_graph(h=self.h, index=self.index, edge_weight=neg_edge_mask[i])
            emb = self.pool(self.sub_g, self.proj_head(emb))
            neg_emb = torch.cat([neg_emb, emb])
        pos_sim = torch.cosine_similarity(anchor_emb, pos_emb, dim=-1)
        neg_sim = torch.cosine_similarity(anchor_emb, neg_emb, dim=-1)
        pos_sim = torch.exp(pos_sim / tau)
        neg_sim = torch.exp(neg_sim / tau)
        loss = -torch.log(pos_sim / (pos_sim + neg_sim)).mean()
        return loss

    def forward_graph(self, h, index=None, training=True, get_att=True):
        self.encoder.g = self.g
        self.h = h
        self.index = index
        emb, sub_g = self.encoder.forward_graph(h, index=index, return_subgraph=True)
        if not sub_g.is_homogeneous:
            sub_g = dgl.to_homogeneous(sub_g)
        self.sub_g = sub_g
        # the attention score for each node
        att = self.extractor(emb)
        att = self.sampling(att, training)
        edge_att = att[sub_g.edges()[0]] * att[sub_g.edges()[1]]
        # for each layer, stack the same attention score for each edge
        encoder_emb = self.encoder.forward_graph(h=h, index=index, edge_weight=edge_att)
        if get_att:
            return encoder_emb, edge_att
        return encoder_emb

    def forward(self, x, index=None, training=True):
        if not self.g:
            raise ValueError("graph not specified")
        h0 = self.forward_rnn(x)
        h, att = self.forward_graph(h0, index, training)
        return self.forward_predictor(h0, h), (att, self.pool(self.sub_g, self.proj_head(h)))

