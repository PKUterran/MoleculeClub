import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from typing import List


class GraphConvolutionLayer(nn.Module):
    def __init__(self, i_dim: int, o_dim: int, h_dims: List[int] = None,
                 activation=None, dropout=0.0, residual=False):
        super(GraphConvolutionLayer, self).__init__()
        if h_dims is None:
            h_dims = [128]
        in_dims = [i_dim] + h_dims
        out_dims = h_dims + [o_dim]
        self.linears = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=True)
                                      for in_dim, out_dim in zip(in_dims, out_dims)])
        self.relu = nn.LeakyReLU()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.residual = residual

    def forward(self, x: torch.Tensor, a: torch.Tensor):
        hs = [self.dropout(x)]
        for i, linear in enumerate(self.linears):
            h = a @ linear(hs[i])
            if i < len(self.linears) - 1:
                h = self.relu(h)
            hs.append(h)
        if self.residual:
            h = torch.cat(hs, dim=-1)
        else:
            h = hs[-1]
        if self.activation == 'sigmoid':
            h = torch.sigmoid(h)
        elif self.activation == 'tanh':
            h = torch.tanh(h)
        elif not self.activation:
            pass
        else:
            assert False, 'Undefined activation: {}.'.format(self.activation)
        return h


class LstmPQEncoder(nn.Module):
    def __init__(self, in_dim, pq_dim, h_dim=128, num_layers=1, use_cuda=False, disturb=False, use_lstm=True):
        super(LstmPQEncoder, self).__init__()
        self.use_cuda = use_cuda
        self.disturb = disturb
        self.use_lstm = use_lstm
        self.pq_dim = pq_dim
        if self.use_lstm:
            self.gcl = GraphConvolutionLayer(in_dim, h_dim, h_dims=[h_dim], activation='tanh', residual=True)
            self.relu = nn.ELU()
            self.rnn = nn.LSTM(in_dim + h_dim * 2, 2 * pq_dim, num_layers)
        else:
            self.gcl = GraphConvolutionLayer(in_dim, 2 * pq_dim, h_dims=[h_dim], activation='tanh', residual=False)

    def forward(self, node_features: torch.Tensor, mol_mode_matrix: torch.Tensor, e: torch.Tensor) \
            -> (torch.Tensor, torch.Tensor):
        if self.use_lstm:
            hidden_node_features = self.relu(self.gcl(node_features, e))
            seqs = [hidden_node_features[n == 1, :] for n in mol_mode_matrix]
            lengths = [s.shape[0] for s in seqs]
            m = pad_sequence(seqs)
            output, _ = self.rnn(m)
            ret = torch.cat([output[:lengths[i], i, :] for i in range(len(lengths))])
        else:
            ret = self.gcl(node_features, e)
        if self.disturb:
            d = torch.normal(mean=torch.zeros(ret.size()), std=torch.full(ret.size(), 0.1))
            if self.use_cuda:
                d = d.cuda()
            ret = ret + d
        ret = ret - torch.sum(ret, dim=0) / ret.shape[0]
        return ret[:, :self.pq_dim], ret[:, self.pq_dim:]
