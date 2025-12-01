import abc
import re
from typing import Dict, Callable, List, Tuple, cast

import more_itertools as xitertools
import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as geo_nn
from torch_geometric.data import Data, Batch


# ============================================================
# Utility functions
# ============================================================

def activatize(name: str) -> nn.Module:
    """
    Get activation module.
    """
    if name == "softplus":
        return nn.Softplus()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "identity":
        return nn.Identity()
    else:
        raise RuntimeError(f'Activation module "{name}" is not supported.')


def auto_num_heads(embed_size: int) -> int:
    """
    Automatically choose the number of heads for multi-head attention.
    Prefer a power-of-two divisor close to sqrt(embed_size).
    """
    return xitertools.first_true(
        range(int(np.ceil(np.sqrt(embed_size))), 0, -1),
        default=1,
        pred=lambda x: embed_size % x == 0 and (x & (x - 1)) == 0,
    )


def glorot(module: nn.Module, rng: torch.Generator) -> int:
    """
    Xavier-like initialization for all parameters in a module.
    Returns the number of parameters initialized.
    """
    num_params = 0
    for name, param in module.named_parameters(recurse=True):
        if not param.requires_grad:
            continue
        num_params += param.numel()
        if param.dim() >= 2:
            # Weight-like
            nn.init.xavier_uniform_(param, gain=1.0)
        else:
            # Bias-like
            nn.init.zeros_(param)
    return num_params


# ============================================================
# Sequential modules
# ============================================================

class Linear(nn.Module):
    """
    Linear but recurrent-like module.
    Input:  (T, B, D_in)
    Output: (T, B, D_out), last_state=(B, D_in)  (same as original)
    """

    def __init__(self, feat_input_size: int, feat_target_size: int) -> None:
        super().__init__()
        self.feat_input_size = feat_input_size
        self.feat_target_size = feat_target_size
        self.lin = nn.Linear(self.feat_input_size, self.feat_target_size)

    def forward(
        self,
        tensor: torch.Tensor,  # (T, B, D_in)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T, B, _ = tensor.shape
        out = self.lin(tensor.reshape(T * B, self.feat_input_size))
        out = out.reshape(T, B, self.feat_target_size)
        # 仕様通り、last_state は入力の最終時刻を返す
        return out, tensor[-1]


class Static(nn.Module):
    """
    Treat static features as dynamic (length=1 sequence).
    Input:  (B, D)
    Output: (1, B, D), last_state=(B, D)
    """

    def forward(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return tensor.reshape(1, *tensor.shape), tensor


class MultiheadAttention(nn.Module):
    """
    Multi-head attention with recurrent-like forward.
    Input:  (T, B, D_in)
    Output: (T, B, D_out), attn_weights
    """

    def __init__(self, feat_input_size: int, feat_target_size: int) -> None:
        super().__init__()
        embed_size = feat_target_size
        self.num_heads = auto_num_heads(embed_size)
        self.mha = nn.MultiheadAttention(embed_size, self.num_heads)
        if feat_input_size != embed_size:
            self.transform = nn.Linear(feat_input_size, embed_size, bias=False)
        else:
            self.transform = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,  # (T, B, D_in)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.transform(x)
        y, attn = self.mha(x, x, x)  # (T, B, D), (B*num_heads, T, T)
        return y, cast(torch.Tensor, attn)


class GRUCellSeq(nn.Module):
    """
    Wrapper to use GRUCell as a recurrent module over time.
    Input:  (T, B, D_in)
    Output: (T, B, D_out), last_state=(B, D_out)
    """

    def __init__(self, feat_input_size: int, feat_target_size: int) -> None:
        super().__init__()
        self.cell = nn.GRUCell(feat_input_size, feat_target_size)
        self.feat_target_size = feat_target_size

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T, B, _ = x.shape
        h = x.new_zeros(B, self.feat_target_size)
        outs = []
        for t in range(T):
            h = self.cell(x[t], h)
            outs.append(h.unsqueeze(0))
        out = torch.cat(outs, dim=0)
        return out, h


class LSTMCellSeq(nn.Module):
    """
    Wrapper to use LSTMCell as a recurrent module over time.
    Input:  (T, B, D_in)
    Output: (T, B, D_out), last_state=(B, D_out)
    """

    def __init__(self, feat_input_size: int, feat_target_size: int) -> None:
        super().__init__()
        self.cell = nn.LSTMCell(feat_input_size, feat_target_size)
        self.feat_target_size = feat_target_size

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T, B, _ = x.shape
        h = x.new_zeros(B, self.feat_target_size)
        c = x.new_zeros(B, self.feat_target_size)
        outs = []
        for t in range(T):
            h, c = self.cell(x[t], (h, c))
            outs.append(h.unsqueeze(0))
        out = torch.cat(outs, dim=0)
        return out, h


def sequentialize(
    name: str,
    feat_input_size: int,
    feat_target_size: int,
) -> nn.Module:
    """
    Get sequential module that always returns (seq, last_state).
    """
    if name == "linear":
        return Linear(feat_input_size, feat_target_size)
    elif name == "gru":
        return nn.GRU(feat_input_size, feat_target_size)
    elif name == "lstm":
        return nn.LSTM(feat_input_size, feat_target_size)
    elif name == "gru[]":
        return GRUCellSeq(feat_input_size, feat_target_size)
    elif name == "lstm[]":
        return LSTMCellSeq(feat_input_size, feat_target_size)
    elif name == "mha":
        return MultiheadAttention(feat_input_size, feat_target_size)
    elif name == "static":
        return Static()
    else:
        raise RuntimeError(f'Sequential module "{name}" is not supported.')


# ============================================================
# GNN modules
# ============================================================

class GNNx2(nn.Module):
    """
    Graph neural network (2-layer).
    """

    def __init__(
        self,
        feat_input_size_edge: int,
        feat_input_size_node: int,
        feat_target_size: int,
        embed_inside_size: int,
        *,
        convolve: str,
        skip: bool,
        activate: str,
    ) -> None:
        super().__init__()

        self.activate = activatize(activate)

        # GNN layers
        self.gnn1 = self._build_conv(
            convolve,
            feat_input_size_edge,
            feat_input_size_node,
            embed_inside_size,
            activate=activate,
        )
        self.gnn2 = self._build_conv(
            convolve,
            feat_input_size_edge,
            embed_inside_size,
            feat_target_size,
            activate=activate,
        )

        # Edge transform (for scalar weights in GCN/Cheb)
        if feat_input_size_edge > 1 and convolve in ("gcn", "gcnub", "cheb"):
            self.edge_transform = nn.Linear(feat_input_size_edge, 1)
            self.edge_activate = activatize("softplus")
        else:
            self.edge_transform = nn.Identity()
            self.edge_activate = activatize("identity")

        # Skip connection
        if feat_input_size_node == feat_target_size:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Linear(feat_input_size_node, feat_target_size)

        self.doskip = int(skip)

    def _build_conv(
        self,
        name: str,
        feat_input_size_edge: int,
        feat_input_size_node: int,
        feat_target_size: int,
        *,
        activate: str,
    ) -> nn.Module:
        if name == "gcn":
            module = geo_nn.GCNConv(feat_input_size_node, feat_target_size)
        elif name == "gcnub":
            module = geo_nn.GCNConv(
                feat_input_size_node, feat_target_size, bias=False
            )
        elif name == "gat":
            heads = auto_num_heads(feat_target_size)
            module = geo_nn.GATConv(
                feat_input_size_node,
                feat_target_size // heads,
                heads=heads,
                edge_dim=feat_input_size_edge,
            )
        elif name == "cheb":
            module = geo_nn.ChebConv(feat_input_size_node, feat_target_size, K=2)
        elif name == "gin":
            nn_sequential = nn.Sequential(
                nn.Linear(feat_input_size_node, feat_target_size),
                activatize(activate),
                nn.Linear(feat_target_size, feat_target_size),
            )
            module = geo_nn.GINEConv(
                nn_sequential,
                edge_dim=feat_input_size_edge,
            )
        else:
            raise RuntimeError(f'GNN conv "{name}" is not supported.')
        return module

    def reset(self, rng: torch.Generator) -> int:
        resetted = 0
        resetted += glorot(self.gnn1, rng)
        resetted += glorot(self.gnn2, rng)
        resetted += glorot(self.edge_transform, rng)
        resetted += glorot(self.skip, rng)
        return resetted

    def convolve(
        self,
        edge_index: torch.Tensor,  # (2, E_total)
        edge_feats: torch.Tensor,  # (E_total, edge_dim or 1)
        node_feats: torch.Tensor,  # (N_total, D_node)
    ) -> torch.Tensor:
        """
        Core GNN propagation; works on batched graphs as well (via PyG Batch).
        """
        if isinstance(self.gnn1, geo_nn.GCNConv) or isinstance(
            self.gnn1, geo_nn.ChebConv
        ):
            # GCN/Cheb use scalar edge weights
            edge_weight = edge_feats.squeeze(-1)
            x = self.gnn1(node_feats, edge_index, edge_weight=edge_weight)
            x = self.gnn2(self.activate(x), edge_index, edge_weight=edge_weight)
        else:
            # GAT / GINE use edge_attr
            x = self.gnn1(node_feats, edge_index, edge_feats.squeeze())
            x = self.gnn2(self.activate(x), edge_index, edge_feats.squeeze())
        return x

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_feats: torch.Tensor,
        node_feats: torch.Tensor,
    ) -> torch.Tensor:
        # Edge features transform
        edge_embeds = self.edge_activate(self.edge_transform(edge_feats))

        # GNN propagation
        node_embeds = self.convolve(edge_index, edge_embeds, node_feats)

        # Skip connection
        node_residuals = self.skip(node_feats)
        return node_embeds + self.doskip * node_residuals


class GNNx2Concat(GNNx2):
    """
    2-layer GNN with input concatenation (output = [GNN(x); x]).
    """

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_feats: torch.Tensor,
        node_feats: torch.Tensor,
    ) -> torch.Tensor:
        node_embeds = super().forward(edge_index, edge_feats, node_feats)
        return torch.cat((node_embeds, node_feats), dim=-1)


def graphicalize_gnn(
    name: str,
    feat_input_size_edge: int,
    feat_input_size_node: int,
    feat_target_size: int,
    embed_inside_size: int,
    *,
    skip: bool,
    activate: str,
    concat: bool,
) -> nn.Module:
    """
    Get 2-layer GNN module (with/without concat).
    """
    if concat:
        return GNNx2Concat(
            feat_input_size_edge,
            feat_input_size_node,
            feat_target_size,
            embed_inside_size,
            convolve=name,
            skip=skip,
            activate=activate,
        )
    else:
        return GNNx2(
            feat_input_size_edge,
            feat_input_size_node,
            feat_target_size,
            embed_inside_size,
            convolve=name,
            skip=skip,
            activate=activate,
        )


# ============================================================
# GRU_GCN core model
# ============================================================

class GRU_GCN(nn.Module):
    """
    Sequential neural network (GRUなど) → 2-layer GNN (batch対応)
    inputs:  (T, B, N, F)
    supports(adj): (B, T, N, N) or (B, T, 1, N, N)
    出力: (B, N, D_out)
    """

    def __init__(
        self,
        feat_input_size_edge: int,
        feat_input_size_node: int,
        feat_target_size: int,
        embed_inside_size: int,
        *,
        convolve: str,
        reduce_edge: str,
        reduce_node: str,
        skip: bool,
        activate: str,
        concat: bool,
        neo_gnn: bool,
    ) -> None:
        super().__init__()

        feat_input_size_edge = 1

        print("feat_input_size_edge:", feat_input_size_edge)
        print("feat_input_size_node:", feat_input_size_node)
        print("embed_inside_size:", embed_inside_size)
        print("feat_target_size:", feat_target_size)

        self.reduce_edge = reduce_edge
        self.reduce_node = reduce_node
        self.embed_inside_size = embed_inside_size

        self.snn_edge = sequentialize(reduce_edge, feat_input_size_edge, embed_inside_size)
        self.snn_node = sequentialize(reduce_node, feat_input_size_node, embed_inside_size)

        gnn_edge_in_dim = feat_input_size_edge if reduce_edge == "static" else embed_inside_size
        self.gnnx2 = graphicalize_gnn(
            convolve,
            gnn_edge_in_dim,
            embed_inside_size,
            feat_target_size,
            embed_inside_size,
            skip=skip,
            activate=activate,
            concat=concat,
        )
        self.activate = activatize(activate)
        self.SIMPLEST = False

        self.feat_target_size = feat_target_size + (int(concat) * embed_inside_size)
        self.neo_gnn = neo_gnn

    def reset(self, rng: torch.Generator) -> int:
        resetted = 0
        resetted += glorot(self.snn_edge, rng)
        resetted += glorot(self.snn_node, rng)
        if hasattr(self.gnnx2, "reset"):
            resetted += self.gnnx2.reset(rng)  # type: ignore
        else:
            resetted += glorot(self.gnnx2, rng)
        return resetted

    @staticmethod
    def _create_full_edge_index(num_nodes: int, device: torch.device) -> torch.Tensor:
        node_idx = torch.arange(num_nodes, device=device)
        # meshgrid → (2, E)
        edge_index = torch.stack(torch.meshgrid(node_idx, node_idx, indexing="ij")).reshape(2, -1)
        return edge_index  # (2, N*N)

    def forward(
        self,
        inputs: torch.Tensor,   # (T, B, N, F)
        supports: torch.Tensor, # (B, T, N, N) or (B, T, 1, N, N)
    ) -> torch.Tensor:
        T, B, N, F = inputs.shape

        device = inputs.device

        node_in = inputs.reshape(T, B * N, F)

        if isinstance(self.snn_node, (nn.GRU, nn.LSTM)):
            node_seq, _ = self.snn_node(node_in)  # (T, B*N, D)
        else:
            node_seq, _ = self.snn_node(node_in)

        node_seq = node_seq.reshape(T, B, N, self.embed_inside_size)
        node_final = self.activate(node_seq[-1])  # (B, N, D_node)

        if supports.dim() == 5 and supports.shape[2] == 1:
            # (B, T, 1, N, N) → (B, T, N, N)
            adj = supports.squeeze(2)
        elif supports.dim() == 4:
            adj = supports
        else:
            raise ValueError(
                f"supports must have shape (B, T, N, N) or (B, T, 1, N, N), "
                f"but got {supports.shape}"
            )

        assert adj.shape[0] == B and adj.shape[1] == T and adj.shape[2] == N

        edge_in = adj.permute(1, 0, 2, 3).reshape(T, B * N * N, 1)

        if isinstance(self.snn_edge, (nn.GRU, nn.LSTM)):
            edge_seq, _ = self.snn_edge(edge_in) 
        else:
            edge_seq, _ = self.snn_edge(edge_in)  

        edge_seq = edge_seq.reshape(T, B, N * N, self.embed_inside_size)
        edge_final = edge_seq[-1]  # (B, E=N*N, D_edge)

        edge_index = self._create_full_edge_index(N, device=device)  # (2, E)

        graphs: List[Data] = []
        for b in range(B):
            data = Data(
                x=node_final[b],          # (N, D_node)
                edge_index=edge_index,    # (2, E)
                edge_attr=edge_final[b],  # (E, D_edge)
            )
            graphs.append(data)

        batch_graph = Batch.from_data_list(graphs)  

        out_nodes = self.gnnx2(
            batch_graph.edge_index,
            batch_graph.edge_attr,  # (E_total, D_edge)
            batch_graph.x,          # (N_total, D_node)
        )  # (N_total, D_out or D_out+inside)

        out_nodes = out_nodes.reshape(B, N, -1)
        return out_nodes


# ============================================================
# Classification wrapper
# ============================================================

class GRU_GCN_classification(nn.Module):
    """
    Sequential neural network then graph neural network (2-layer)
    adapted for classification.
    """

    def __init__(self, args, num_classes: int, device=None, gnn: str = "gcn"):
        super().__init__()

        self.num_nodes = args.num_nodes
        self.num_classes = num_classes
        self.device = device

        self.gru_gcn = GRU_GCN(
            feat_input_size_edge=args.input_dim,
            feat_input_size_node=args.input_dim,
            feat_target_size=args.rnn_units,
            embed_inside_size=args.input_dim,
            convolve=gnn,
            reduce_edge="gru",
            reduce_node="gru",
            skip=False,
            activate="tanh",
            concat=True,
            neo_gnn=True,
        )

        if args.agg != "concat":
            self.fc = nn.Linear(self.gru_gcn.feat_target_size, num_classes)
        else:
            self.fc = nn.Linear(self.gru_gcn.feat_target_size * self.num_nodes, num_classes)

        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()
        self.agg = args.agg

    def forward(
        self,
        input_seq: torch.Tensor,  # (B, T, N, F)
        seq_lengths: torch.Tensor,  # (B,)  
        adj: torch.Tensor,  # (B, T, N, N) or (B, T, 1, N, N)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_seq: (batch, seq_len, num_nodes, input_dim)
            seq_lengths: (batch,)  -- not used currently, kept for API compatibility
            adj: adjacency, (batch, seq_len, num_nodes, num_nodes) or (batch, seq_len, 1, N, N)
        Returns:
            pool_logits: logits from last FC layer (before sigmoid/softmax)
            final_hidden: last hidden from GRU_GCN (shape depends on agg)
        """
        batch_size, max_seq_len = input_seq.shape[0], input_seq.shape[1]

        input_seq = input_seq.transpose(0, 1)  # (T, B, N, F)

        final_hidden = self.gru_gcn(input_seq, adj)

        if self.agg == "concat":
            final_hidden = final_hidden.view(batch_size, -1)  # (B, N*D)

        logits = self.fc(self.relu(self.dropout(final_hidden)))

        if self.agg == "max":
            # max pooling over nodes
            pool_logits, _ = torch.max(logits, dim=1)  # (B, num_features)
        elif self.agg == "mean":
            pool_logits = torch.mean(logits, dim=1)    # (B, num_features)
        elif self.agg == "sum":
            pool_logits = torch.sum(logits, dim=1)     # (B, num_features)
        elif self.agg == "concat":
            pool_logits = logits                       # (B, N*D_out)
        else:
            raise ValueError(f"Unsupported aggregation method: {self.agg}")

        return pool_logits, final_hidden
