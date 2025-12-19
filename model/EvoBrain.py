import torch
import torch.nn as nn
import time
import torch_geometric.nn as geo_nn
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import AddLaplacianEigenvectorPE
import more_itertools as xitertools
import numpy as np
import abc
#
from typing import Dict, Callable, cast, List, Tuple
from mamba_ssm import Mamba

GLOROTS: Dict[type, Callable[[torch.nn.Module, torch.Generator], int]]


def activatize(name: str, /) -> torch.nn.Module:
    R"""
    Get activation module.
    """
    #
    if name == "softplus":
        #
        return torch.nn.Softplus()
    elif name == "sigmoid":
        #
        return torch.nn.Sigmoid()
    elif name == "tanh":
        #
        return torch.nn.Tanh()
    elif name == "identity":
        #
        return torch.nn.Identity()
    else:
        # EXPECT:
        # It is possible to require unsupporting sequential model.
        raise RuntimeError(
            "Activation module identifier \"{:s}\" is not supported."
            .format(name),
        )

def glorot(module: torch.nn.Module, rng: torch.Generator, /) -> int:
    R"""
    Module initialization.
    """
    #
    return GLOROTS[type(module)](module, rng)

def noescape(string: str, /) -> str:
    R"""
    Remove escaping charaters.
    """
    #
    return re.sub(r"\x1b\[[0-9]+(;[0-9]+)*m", "", string)


def infotab5(title: str, lines: List[str]) -> List[str]:
    R"""
    Wrap given lines into a named tab.
    """
    # Format final generated lines according to their maximum length.
    linlen = (
        0 if len(lines) == 0 else max(len(noescape(line)) for line in lines)
    )
    barlen = max(5, (linlen - len(title)) // 2)
    return (
        [
            "\x1b[2m{:s}\x1b[0m".format("-" * barlen)
            + "\x1b[1m{:s}\x1b[0m".format(" " + title + " ")
            + "\x1b[2m{:s}\x1b[0m".format("-" * barlen),
        ]
        + lines
        + ["\x1b[2m{:s}\x1b[0m".format("-" * ((barlen + 1) * 2 + len(title)))]
    )


def auto_num_heads(embed_size: int, /) -> int:
    R"""
    Automatically get number of multi-heads.
    """
    #
    return (
        xitertools.first_true(
            range(int(np.ceil(np.sqrt(embed_size))), 0, -1),
            default=1, pred=lambda x: embed_size % x == 0 and x & (x - 1) == 0,
        )
    )

class Model(abc.ABC, torch.nn.Module):
    R"""
    Model.
    """
    #
    COSTS: Dict[str, List[float]]

    #
    COSTS = {"graph": [], "non-graph": [], "edges": []}

    # Signal to use simplest model for greatest efficency on synthetic tasks.
    SIMPLEST = False

    def __annotation__(self, /) -> None:
        R"""
        Annotate for class instance attributes.
        """
        #
        self.feat_target_size: int

    @abc.abstractmethod
    def reset(self, rng: torch.Generator, /) -> int:
        R"""
        Reset model parameters by given random number generator.
        """
        #
        ...

    def initialize(self, seed: int, /) -> None:
        R"""
        Explicitly initialize the model.
        """
        #
        rng = torch.Generator("cpu")
        rng.manual_seed(seed)
        resetted = self.reset(rng)
        if resetted != sum(param.numel() for param in self.parameters()):
            # UNEXPECT:
            # All defined parameters should match exactly with initialization.
            raise NotImplementedError(
                "Defined parameters do not exactly match with initialized "
                "parameters.",
            )
        self.num_resetted_params = resetted

    def __repr__(self) -> str:
        R"""
        Get representation of the class.
        """
        # Model parameter key info has a special visible representation.
        names = []
        shapes = []
        for name, param in self.named_parameters():
            #
            names.append(name.split("."))
            shapes.append(
                "\x1b[90mx\x1b[0m".join(str(dim) for dim in param.shape),
            )
        depth = 0 if len(names) == 0 else max(len(levels) for levels in names)
        padded = [levels + [""] * (depth - len(levels)) for levels in names]

        #
        keys = (
            "\x1b[90m-\x1b[92m→\x1b[0m".join(levels).replace(
                "\x1b[92m→\x1b[0m\x1b[90m", "\x1b[90m→",
            )
            for levels in (
                [
                    [
                        "{:<{:d}s}".format(name, maxlen).replace(
                            " ", "\x1b[90m-\x1b[0m",
                        )
                        for (name, maxlen) in (
                            zip(
                                levels,
                                (
                                    max(len(name) for name in level)
                                    for level in zip(*padded)
                                ),
                            )
                        )
                    ]
                    for levels in padded
                ]
            )
        )

        # We may also care about the product besides shape.
        maxlen = (
            0
            if len(shapes) == 0 else
            max(len(noescape(shape)) for shape in shapes)
        )
        shapes = (
            [
                "{:s}{:s} ({:d})".format(
                    " " * (maxlen - len(noescape(shape))), shape,
                    int(
                        np.prod(
                            [
                                int(dim)
                                for dim in shape.split("\x1b[90mx\x1b[0m")
                            ],
                        ),
                    ),
                )
                for shape in shapes
            ]
        )

        # Generate final representation.
        return "\n".join(infotab5(
            "(Param)eter",
            [
                key + "\x1b[90m→\x1b[94m:\x1b[0m " + shape
                for (key, shape) in zip(keys, shapes)
            ],
        ))

    def moveon(self, notembedon: List[int]) -> None:
        R"""
        Set axes for moving window model.
        """
        # EXPECT:
        # By default, the model is not moving window unless it is overloaded.
        raise RuntimeError(
            "Default model is not a moving window model, and you need to "
            "explicitly overload to use moving window.",
        )

    def pretrain(self, partname: str, path: str, /) -> None:
        R"""
        Use pretrained model.
        """
        # EXPECT:
        # By default, there is no pretraining definition.
        raise RuntimeError(
            "No pretraining of \"{:s}\" is defined."
            .format(self.__class__.__name__),
        )
    
class GNNx2(Model):
    R"""
    Graph neural network (2-layer).
    """
    def __init__(
        self,
        feat_input_size_edge: int, feat_input_size_node: int,
        feat_target_size: int, embed_inside_size: int,
        /,
        *,
        convolve: str, skip: bool, activate: str,
    ) -> None:
        R"""
        Initialize the class.
        """
        #
        Model.__init__(self)

        # TODO:
        # Given more than 2 layers, we can introduce dense connection.
        self.gnn1 = (
            self.graphicalize(
                convolve, feat_input_size_edge, feat_input_size_node,
                embed_inside_size,
                activate=activate,
            )
        )
        self.gnn2 = (
            self.graphicalize(
                convolve, feat_input_size_edge, embed_inside_size,
                feat_target_size,
                activate=activate,
            )
        )

        #
        self.edge_transform: torch.nn.Module
        self.skip: torch.nn.Module

        #
        if feat_input_size_edge > 1 and convolve in ("gcn", "gcnub", "cheb"):
            #
            self.edge_transform = torch.nn.Linear(feat_input_size_edge, 1)
            self.edge_activate = activatize("softplus")
        else:
            self.edge_transform = torch.nn.Identity()
            self.edge_activate = activatize("identity")

        #
        if feat_input_size_node == feat_target_size:
            #
            self.skip = torch.nn.Identity()
        else:
            #
            self.skip = (
                torch.nn.Linear(feat_input_size_node, feat_target_size)
            )

        #
        self.activate = activatize(activate)

        # Use a 0-or-1 integer to mask skip connection.
        self.doskip = int(skip)

    def graphicalize(
        self,
        name: str, feat_input_size_edge: int, feat_input_size_node: int,
        feat_target_size: int,
        /,
        *,
        activate: str,
    ) -> torch.nn.Module:
        R"""
        Get unit graphical module.
        """
        # TODO:
        # Wait for Pytorch Geometric type annotation supporting.
        if name == "gcn":
            #
            module = (
                geo_nn.GCNConv(feat_input_size_node, feat_target_size)
            )
        elif name == "gcnub":
            #
            module = (
                geo_nn.GCNConv(
                    feat_input_size_node, feat_target_size,
                    bias=False,
                )
            )
        elif name == "gatedgcn":
            #
            module = (
                geo_nn.ResGatedGraphConv(
                    feat_input_size_node, feat_target_size,
                    edge_dim=1,
                )
            )
        elif name == "ssg":
            #
            module = (
                geo_nn.SSGConv(
                    feat_input_size_node, feat_target_size,alpha=0.05
                )
            )
        elif name == "sg":
            #
            module = (
                geo_nn.SGConv(
                    feat_input_size_node, feat_target_size,
                )
            )
        elif name == "gat":
            #
            heads = auto_num_heads(feat_target_size)
            module = (
                geo_nn.GATConv(
                    feat_input_size_node, feat_target_size // heads,
                    heads=heads, edge_dim=feat_input_size_edge,
                )
            )
        elif name == "cheb":
            #
            module = (
                geo_nn.ChebConv(feat_input_size_node, feat_target_size, 2)
            )
        elif name == "gin":
            #
            module = (
                geo_nn.GINEConv(
                    torch.nn.Sequential(
                        torch.nn.Linear(
                            feat_input_size_node, feat_target_size,
                        ),
                        activatize(activate),
                        torch.nn.Linear(feat_target_size, feat_target_size),
                    ),
                    edge_dim=1,
                )
            )
        else:
            # EXPECT:
            # It is possible to require unsupporting sequential model.
            raise RuntimeError(
                "Graphical module identifier \"{:s}\" is not supported."
                .format(name),
            )
        return cast(torch.nn.Module, module)

    def reset(self, rng: torch.Generator, /) -> int:
        R"""
        Reset model parameters by given random number generator.
        """
        #
        resetted = 0
        resetted = resetted + glorot(self.gnn1, rng)
        resetted = resetted + glorot(self.gnn2, rng)
        resetted = resetted + glorot(self.edge_transform, rng)
        resetted = resetted + glorot(self.skip, rng)
        return resetted

    def convolve(
        self,
        edge_tuples: torch.Tensor, edge_feats: torch.Tensor,
        node_feats: torch.Tensor,
        /,
    ) -> torch.Tensor:
        R"""
        Convolve.
        """
        # TODO:
        # Wait for Pytorch Geometric type annotation supporting.
        node_embeds: torch.Tensor

        #
        node_embeds = (
            self.gnn1.forward(node_feats, edge_tuples, edge_feats)
        )
        node_embeds = (
            self.gnn2.forward(
                self.activate(node_embeds), edge_tuples, edge_feats,
            )
        )
        return node_embeds

    def forward(
        self,
        edge_tuples: torch.Tensor, edge_feats: torch.Tensor,
        node_feats: torch.Tensor,
        /,
    ) -> torch.Tensor:
        R"""
        Forward.
        """
        # TODO:
        # Wait for Pytorch Geometric type annotation supporting.
        edge_embeds: torch.Tensor
        node_embeds: torch.Tensor
        node_residuals: torch.Tensor

        #
        edge_embeds = (
            self.edge_activate(self.edge_transform.forward(edge_feats))
        )
        node_embeds = self.convolve(edge_tuples, edge_embeds, node_feats)
        node_residuals = self.skip.forward(node_feats)
        return node_embeds + self.doskip * node_residuals


class GNNx2Concat(GNNx2):
    R"""
    Graph neural network (2-layer) with input concatenation.
    """
    #
    def forward(
        self,
        edge_tuples: torch.Tensor, edge_feats: torch.Tensor,
        node_feats: torch.Tensor,
        /,
    ) -> torch.Tensor:
        R"""
        Forward.
        """
        #
        node_embeds: torch.Tensor

        # Super call.
        node_embeds = GNNx2.forward(self, edge_tuples, edge_feats, node_feats)
        node_embeds = torch.cat((node_embeds, node_feats), dim=1)
        return node_embeds


def graphicalize(
    name: str, feat_input_size_edge, feat_input_size_node: int,
    feat_target_size: int, embed_inside_size: int,
    /,
    *,
    skip: bool, activate: str, concat: bool,
) -> Model:
    R"""
    Get 2-layer graphical module.
    """
    #
    if concat:
        #
        return (
            GNNx2Concat(
                feat_input_size_edge, feat_input_size_node, feat_target_size,
                embed_inside_size,
                convolve=name, skip=skip, activate=activate,
            )
        )
    else:
        #
        return (
            GNNx2(
                feat_input_size_edge, feat_input_size_node, feat_target_size,
                embed_inside_size,
                convolve=name, skip=skip, activate=activate,
            )
        )

class GNNx2Concat(GNNx2):
    R"""
    Graph neural network (2-layer) with input concatenation.
    """
    #
    def forward(
        self,
        edge_tuples: torch.Tensor, edge_feats: torch.Tensor,
        node_feats: torch.Tensor,
        /,
    ) -> torch.Tensor:
        R"""
        Forward.
        """
        #
        node_embeds: torch.Tensor

        # Super call.
        node_embeds = GNNx2.forward(self, edge_tuples, edge_feats, node_feats)
        node_embeds = torch.cat((node_embeds, node_feats), dim=1)
        return node_embeds
    
class GNNx2(Model):
    R"""
    Graph neural network (2-layer).
    """
    def __init__(
        self,
        feat_input_size_edge: int, feat_input_size_node: int,
        feat_target_size: int, embed_inside_size: int,
        /,
        *,
        convolve: str, skip: bool, activate: str,
    ) -> None:
        R"""
        Initialize the class.
        """
        #
        Model.__init__(self)

        # TODO:
        # Given more than 2 layers, we can introduce dense connection.
        self.gnn1 = (
            self.graphicalize(
                convolve, feat_input_size_edge, feat_input_size_node,
                embed_inside_size,
                activate=activate,
            )
        )
        self.gnn2 = (
            self.graphicalize(
                convolve, feat_input_size_edge, embed_inside_size,
                feat_target_size,
                activate=activate,
            )
        )

        #
        self.edge_transform: torch.nn.Module
        self.skip: torch.nn.Module

        #
        if feat_input_size_edge > 1 and convolve in ("gcn", "gcnub", "cheb"):
            #
            self.edge_transform = torch.nn.Linear(feat_input_size_edge, 1)
            self.edge_activate = activatize("softplus")
        else:
            self.edge_transform = torch.nn.Identity()
            self.edge_activate = activatize("identity")

        #
        if feat_input_size_node == feat_target_size:
            #
            self.skip = torch.nn.Identity()
        else:
            #
            self.skip = (
                torch.nn.Linear(feat_input_size_node, feat_target_size)
            )

        #
        self.activate = activatize(activate)

        # Use a 0-or-1 integer to mask skip connection.
        self.doskip = int(skip)

    def graphicalize(
        self,
        name: str, feat_input_size_edge: int, feat_input_size_node: int,
        feat_target_size: int,
        /,
        *,
        activate: str,
    ) -> torch.nn.Module:
        R"""
        Get unit graphical module.
        """
        # TODO:
        # Wait for Pytorch Geometric type annotation supporting.
        if name == "gcn":
            #
            module = (
                geo_nn.GCNConv(feat_input_size_node, feat_target_size)
            )
        elif name == "gcnub":
            #
            module = (
                geo_nn.GCNConv(
                    feat_input_size_node, feat_target_size,
                    bias=False,
                )
            )
        elif name == "gatedgcn":
            #
            module = (
                geo_nn.ResGatedGraphConv(
                    feat_input_size_node, feat_target_size,
                )
            )
        elif name == "gat":
            #
            heads = auto_num_heads(feat_target_size)
            module = (
                geo_nn.GATConv(
                    feat_input_size_node, feat_target_size // heads,
                    heads=heads, edge_dim=feat_input_size_edge,
                )
            )
        elif name == "cheb":
            #
            module = (
                geo_nn.ChebConv(feat_input_size_node, feat_target_size, 2)
            )
        elif name == "gin":
            #
            module = (
                geo_nn.GINEConv(
                    torch.nn.Sequential(
                        torch.nn.Linear(
                            feat_input_size_node, feat_target_size,
                        ),
                        activatize(activate),
                        torch.nn.Linear(feat_target_size, feat_target_size),
                    ),
                    edge_dim=feat_input_size_edge,
                )
            )
        else:
            # EXPECT:
            # It is possible to require unsupporting sequential model.
            raise RuntimeError(
                "Graphical module identifier \"{:s}\" is not supported."
                .format(name),
            )
        return cast(torch.nn.Module, module)

    def reset(self, rng: torch.Generator, /) -> int:
        R"""
        Reset model parameters by given random number generator.
        """
        #
        resetted = 0
        resetted = resetted + glorot(self.gnn1, rng)
        resetted = resetted + glorot(self.gnn2, rng)
        resetted = resetted + glorot(self.edge_transform, rng)
        resetted = resetted + glorot(self.skip, rng)
        return resetted

    def convolve(
        self,
        edge_tuples: torch.Tensor, edge_feats: torch.Tensor,
        node_feats: torch.Tensor,
        /,
    ) -> torch.Tensor:
        R"""
        Convolve.
        """
        # TODO:
        # Wait for Pytorch Geometric type annotation supporting.
        node_embeds: torch.Tensor

        #
        node_embeds = (
            self.gnn1.forward(node_feats, edge_tuples, edge_feats.squeeze())
        )
        node_embeds = (
            self.gnn2.forward(
                self.activate(node_embeds), edge_tuples, edge_feats.squeeze(),
            )
        )
        return node_embeds

    def forward(
        self,
        edge_tuples: torch.Tensor, edge_feats: torch.Tensor,
        node_feats: torch.Tensor,
        /,
    ) -> torch.Tensor:
        R"""
        Forward.
        """
        # TODO:
        # Wait for Pytorch Geometric type annotation supporting.
        edge_embeds: torch.Tensor
        node_embeds: torch.Tensor
        node_residuals: torch.Tensor

        edge_embeds = edge_feats
        node_embeds = self.convolve(edge_tuples, edge_embeds, node_feats)
        node_residuals = self.skip.forward(node_feats)
        return node_embeds + self.doskip * node_residuals
    


class Linear(torch.nn.Module):
    R"""
    Linear but recurrent module.
    """
    def __init__(self, feat_input_size: int, feat_target_size: int, /) -> None:
        R"""
        Initialize the class.
        """
        #
        torch.nn.Module.__init__(self)

        #
        self.feat_input_size = feat_input_size
        self.feat_target_size = feat_target_size
        self.lin = torch.nn.Linear(self.feat_input_size, self.feat_target_size)

    def forward(
        self,
        tensor: torch.Tensor,
        /,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        R"""
        Forward.
        """
        #
        (num_times, num_samples, _) = tensor.shape
        return (
            torch.reshape(
                self.lin.forward(
                    torch.reshape(
                        tensor,
                        (num_times * num_samples, self.feat_input_size),
                    ),
                ),
                (num_times, num_samples, self.feat_target_size),
            ),
            tensor[-1],
        )


class Static(torch.nn.Module):
    R"""
    Treate static feature as dynamic.
    """
    def forward(
        self,
        tensor: torch.Tensor,
        /,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        R"""
        Forward.
        """
        #
        return (torch.reshape(tensor, (1, *tensor.shape)), tensor)


class MultiheadAttention(torch.nn.Module):
    R"""
    Multi-head attention with recurrent-like forward.
    """
    def __init__(self, feat_input_size: int, feat_target_size: int, /) -> None:
        R"""
        Initialize the class.
        """
        #
        torch.nn.Module.__init__(self)

        #
        embed_size = feat_target_size
        self.num_heads = auto_num_heads(embed_size)
        self.mha = torch.nn.MultiheadAttention(embed_size, self.num_heads)

        #
        self.transform: torch.nn.Module

        #
        if feat_input_size != embed_size:
            #
            self.transform = (
                torch.nn.Linear(feat_input_size, embed_size, bias=False)
            )
        else:
            #
            self.transform = torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        R"""
        Forward.
        """
        #
        x = self.transform(x)
        (y, attn) = self.mha.forward(x, x, x)
        return (y, cast(torch.Tensor, attn))
    
def sequentialize(
    name: str, feat_input_size: int, feat_target_size: int,
    /,
) -> torch.nn.Module:
    R"""
    Get sequential module.
    """
    #
    print("snn: ", name)
    if name == "mamba":
        return nn.Sequential(
            nn.Linear(feat_input_size, feat_target_size),
            Mamba(
                d_model=feat_target_size, # Model dimension d_model
                d_state=16,  # SSM state expansion factor
                d_conv=4,    # Local convolution width
                expand=2,    # Block expansion factor
            ),
        )
    elif name == "linear":
        #
        return Linear(feat_input_size, feat_target_size)
    elif name == "gru":
        #
        return torch.nn.GRU(feat_input_size, feat_target_size)
    elif name == "lstm":
        #
        return torch.nn.LSTM(feat_input_size, feat_target_size)
    elif name == "gru[]":
        #
        return torch.nn.GRUCell(feat_input_size, feat_target_size)
    elif name == "lstm[]":
        #
        return torch.nn.LSTMCell(feat_input_size, feat_target_size)
    elif name == "mha":
        #
        return MultiheadAttention(feat_input_size,feat_target_size)
    elif name == "static":
        #
        return Static()
    else:
        # EXPECT:
        # It is possible to require unsupporting sequential model.
        raise RuntimeError(
            "Sequential module identifier is not supported."
        )


class EvoBrain(nn.Module):
    """
    Sequential neural network then graph neural network (2-layer).
    """
    def __init__(self, feat_input_size_edge, feat_input_size_node, feat_target_size, embed_inside_size, convolve, reduce_edge, reduce_node, skip, activate, concat, neo_gnn, num_eigenvectors=8):
        super(EvoBrain, self).__init__()
        feat_input_size_edge =1

        # Add Laplacian Eigenvector PE
        if num_eigenvectors > 0:
            self.laplacian_pe = AddLaplacianEigenvectorPE(k=num_eigenvectors)
        self.num_eigenvectors = num_eigenvectors

        self.reduce_edge = reduce_edge
        self.reduce_node = reduce_node
        self.snn_edge = sequentialize(reduce_edge, feat_input_size_edge, embed_inside_size)
        self.snn_node = sequentialize(reduce_node, feat_input_size_node, embed_inside_size)
        self.gnnx2 = graphicalize(convolve, feat_input_size_edge if reduce_edge == "static" else embed_inside_size, embed_inside_size + num_eigenvectors, feat_target_size, embed_inside_size, skip=skip, activate=activate, concat=concat)
        self.activate = activatize(activate)
        self.SIMPLEST = False

        self.edge_transform = torch.nn.Linear(embed_inside_size, 1)
        self.edge_activate = activatize("softplus")
        
        self.feat_target_size = feat_target_size + int(concat) * embed_inside_size

        self.neo_gnn = neo_gnn
        
    def reset(self, rng):
        resetted = 0
        resetted = resetted + glorot(self.snn_edge, rng)
        resetted = resetted + glorot(self.snn_node, rng)
        resetted = resetted + self.gnnx2.reset(rng)
        return resetted

    
    def prepare_edge_weights(self, edge_embeds):
        """
        Prepare edge weights from edge embeddings for Laplacian computation.
        """
        # If edge embeddings are multi-dimensional, compute a scalar weight
        if edge_embeds.dim() > 1 and edge_embeds.size(-1) > 1:
            # Use the norm of the edge embedding as weight
            weights = torch.norm(edge_embeds, p=2, dim=-1)
        else:
            # If already scalar, just squeeze
            weights = edge_embeds.squeeze(-1)
        
        # Ensure positive weights
        weights = torch.abs(weights)
        
        # Normalize weights
        weights = weights / (weights.max() + 1e-6)
        
        return weights
    
    def forward(self, inputs, supports):
        timestep, b, node, dim = inputs.shape
        inputs = inputs.reshape(timestep, b*node, dim)
        
        if self.reduce_node == "mingru" or self.reduce_node == "mamba":
            inputs = inputs.permute(1, 0, 2)
            node_embeds = self.snn_node.forward(inputs)
            node_embeds =  node_embeds.permute(1, 0, 2)
        else:
            node_embeds, _ = self.snn_node.forward(inputs)
        node_embeds = node_embeds.reshape(timestep, b, node, dim)

        if supports.shape[2] == 1:
            supports = torch.squeeze(supports, dim=2)
        edge_tuples, edge_features = self.create_edge_tuples_and_features(supports)

        edge_features = edge_features.reshape(timestep, -1, 1)

        if self.reduce_edge == "mingru" or self.reduce_edge == "mamba":
            edge_features =  edge_features.permute(1, 0, 2)
            edge_embeds = self.snn_edge.forward(edge_features)
            edge_embeds =  edge_embeds.permute(1, 0, 2)
        else:
            edge_embeds, _ = self.snn_edge.forward(edge_features)
        edge_embeds = edge_embeds.reshape(timestep, b, -1, dim)

        all_node_embeds = node_embeds
        all_edge_embeds = edge_embeds

        device = next(self.parameters()).device

        edge_tuples = edge_tuples.to(device)

        node_last = all_node_embeds[-1].to(device)   # [b, node, dim]
        edge_last = all_edge_embeds[-1].to(device)   # [b, E, dim]
        edge_weights = self.edge_activate(self.edge_transform(edge_last))  # [b, E, 1]

        node_with_pe_list = []
        edge_index_list = []
        edge_weight_list = []

        for i in range(b):
            ew = edge_weights[i]
            ew = ew / (ew.max() + 1e-6)

            data = Data(
                x=node_last[i],
                edge_index=edge_tuples,
                edge_weight=ew.squeeze(-1)
            )

            if self.num_eigenvectors > 0:
                data = self.laplacian_pe(data.detach())
                pe = data.laplacian_eigenvector_pe.to(device)
                x_i = torch.cat([node_last[i], pe], dim=-1)
            else:
                x_i = node_last[i]

            node_with_pe_list.append(x_i)
            
            edge_index_list.append(edge_tuples + i * node)
            edge_weight_list.append(edge_weights[i].squeeze(-1))

        # [b, node, dim(+k)]
        node_with_pe = torch.stack(node_with_pe_list, dim=0)

        x = self.activate(node_with_pe)
        x = x.view(b * node_last.size(1), -1)

        edge_index = torch.cat(edge_index_list, dim=1)   # [2, b*E]
        edge_weight = torch.cat(edge_weight_list, dim=0) # [b*E]

        out = self.gnnx2.forward(
            edge_index,
            edge_weight,
            x
        )

        outputs = out.view(b, node_last.size(1), -1)
        return outputs


    
    def create_edge_tuples_and_features(self, adj):
        batch_size, timesteps, num_nodes, _ = adj.shape
        node_indices = torch.arange(num_nodes)
        edge_tuples = torch.stack(torch.meshgrid(node_indices, node_indices)).reshape(2, -1).to(next(self.parameters()).device)
        edge_features = adj.reshape(timesteps, batch_size, -1).unsqueeze(-1) # (timesteps, batch_size, num_nodes num_nodes, 1)
        non_zero_indices = torch.nonzero((edge_features.sum(dim=(0, 1)) > 0.0001) | (edge_features.sum(dim=(0, 1)) < -0.0001)).to(next(self.parameters()).device)
        non_zero_edge_tuples = edge_tuples[:, non_zero_indices[:,0]]

        non_zero_edge_features = edge_features[:, :, non_zero_indices[:, 0], non_zero_indices[:, 1]]
        non_zero_edge_features = non_zero_edge_features.reshape(timesteps, batch_size, -1, 1)

        return non_zero_edge_tuples, non_zero_edge_features

    
    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_rnn_layers):
            init_states.append(
                torch.zeros(batch_size, self.rnn_units).to(next(self.parameters()).device)
            )
        return torch.stack(init_states, dim=0)
    


class EvoBrain_classification(nn.Module):
    """
    Sequential neural network then graph neural network (2-layer) adapted for classification.
    """
    def __init__(self, args, num_classes, device=None, gnn="ssg", num_eigenvectors=16):
        super(EvoBrain_classification, self).__init__()
        
        self.num_nodes = args.num_nodes
        self.num_classes = num_classes
        self.device = device
        
        self.evobrain = EvoBrain(feat_input_size_edge=args.input_dim, 
                               feat_input_size_node=args.input_dim,
                               feat_target_size=args.rnn_units, 
                               embed_inside_size=args.input_dim,
                               convolve=gnn, 
                               reduce_edge="mamba",
                               reduce_node="mamba", 
                               skip=False,
                               activate="tanh", 
                               concat=True,
                               neo_gnn=True,
                                num_eigenvectors=num_eigenvectors
                               )
        if args.agg != "concat":
            self.fc = nn.Linear(self.evobrain.feat_target_size + num_eigenvectors, num_classes)
        else:
            self.fc = nn.Linear(self.evobrain.feat_target_size * self.num_nodes, num_classes)
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()
        self.agg = args.agg
        self.numeigenvectors = num_eigenvectors
        
    def forward(self, input_seq, seq_lengths, adj):
        """
        Args:
            input_seq: input sequence, shape (batch, seq_len, num_nodes, input_dim)
            seq_lengths: actual seq lengths w/o padding, shape (batch,)
            supports: list of supports from laplacian or dual_random_walk filters
        Returns:
            pool_logits: logits from last FC layer (before sigmoid/softmax)
        """
        batch_size, max_seq_len = input_seq.shape[0], input_seq.shape[1]
        
        # (max_seq_len, batch, num_nodes, input_dim)
        input_seq = torch.transpose(input_seq, dim0=0, dim1=1)
        
        final_hidden = self.evobrain(input_seq, adj)
        if self.agg == "concat":
            final_hidden = final_hidden.view(batch_size, -1)  # (batch_size, num_nodes * num_features)
        logits = self.fc(self.relu(self.dropout(final_hidden)))
        
        if self.agg == "max":
            # max pooling over nodes
            pool_logits, _ = torch.max(logits, dim=1)  # (batch_size, num_features)
        elif self.agg == "mean":
            # mean pooling over nodes
            pool_logits = torch.mean(logits, dim=1)  # (batch_size, num_features)
        elif self.agg == "sum":
            # sum pooling over nodes
            pool_logits = torch.sum(logits, dim=1)  # (batch_size, num_features)
        elif self.agg == "concat":
            pool_logits = logits
        else:
            raise ValueError(f"Unsupported aggregation method: {self.agg}")
        
        return pool_logits, final_hidden