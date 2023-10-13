import torch
import dgl
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


class GSAT(GraphModel):
    def __init__(self, clf, extractor, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 base_model="LSTM", use_residual=True, is_homogeneous=True):
        super().__init__(d_feat=d_feat, hidden_size=hidden_size, num_layers=num_layers,
                         dropout=dropout, base_model=base_model, use_residual=use_residual,
                         is_homogeneous=is_homogeneous)
        # GNN encoder
        self.clf = clf
        self.extractor = extractor
        self.info_loss_coef = 0.5
        self.decay_interval = 10
        self.decay_r = 0.1
        self.init_r = 0.9
        self.final_r = 0.5
        self.temp = 1

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

    def info_loss(self, att, epoch):
        loss = 0.
        r = self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r, init_r=self.init_r)
        info_loss = (att * torch.log(att / r + 1e-6) + (1 - att) *
                     torch.log((1 - att) / (1 - r + 1e-6) + 1e-6)).mean()
        info_loss = info_loss * self.info_loss_coef
        loss += info_loss
        return loss

    def get_r(self, decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
        r = init_r - current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        return r

    def forward_graph(self, h, index=None, training=True, get_att=True):
        self.clf.g = self.g
        emb, g = self.clf.forward_graph(h, index=index, return_subgraph=True)
        if not g.is_homogeneous:
            g = dgl.to_homogeneous(g)
        # the attention score for each node
        att = self.extractor(emb)
        att = self.sampling(att, training)
        edge_att = att[g.edges()[0]] * att[g.edges()[1]]
        # for each layer, stack the same attention score for each edge
        clf_emb = self.clf.forward_graph(h=h, index=index, edge_weight=edge_att)
        if get_att:
            return clf_emb, att
        return clf_emb

    def forward(self, x, index=None, training=True):
        if not self.g:
            raise ValueError("graph not specified")
        h0 = self.forward_rnn(x)
        h, att = self.forward_graph(h0, index, training)
        return self.forward_predictor(h0, h), att
