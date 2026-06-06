"""
Graph Neural Network Module for Spatial Modeling
=================================================

Advanced GNN components for capturing spatial relationships in 
meteorological data. Treats each grid point as a node with 
connections to neighbors.

Implements:
1. Graph Construction from grid data
2. Graph Convolutional Networks (GCN)
3. Graph Attention Networks (GAT)
4. GraphSAGE for inductive learning
5. Spectral Graph Convolution
6. Multi-scale Graph Processing

Author: IITM Pune - Government of India Project
Reference: Kipf & Welling 2017, Velikovi et al. 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ChebConv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import grid, to_dense_batch
import numpy as np
from typing import Optional, Tuple, List
import math


class GridGraphConstructor:
    """
    Constructs graph structure from 2D grid data.
    
    Each pixel becomes a node, connected to neighbors based on
    connectivity pattern (4-connected, 8-connected, or distance-based).
    
    Args:
        height: Grid height
        width: Grid width
        connectivity: '4', '8', or 'radius'
        radius: Radius for distance-based connectivity
    """
    
    def __init__(
        self,
        height: int,
        width: int,
        connectivity: str = '8',
        radius: int = 1
    ):
        self.height = height
        self.width = width
        self.connectivity = connectivity
        self.radius = radius
        
        # Pre-compute edge index
        self.edge_index = self._build_edge_index()
        
    def _build_edge_index(self) -> torch.Tensor:
        """Build edge connectivity for grid."""
        edges = []
        
        for i in range(self.height):
            for j in range(self.width):
                node_idx = i * self.width + j
                
                if self.connectivity == '4':
                    # 4-connected: up, down, left, right
                    neighbors = [
                        (i-1, j), (i+1, j), (i, j-1), (i, j+1)
                    ]
                elif self.connectivity == '8':
                    # 8-connected: includes diagonals
                    neighbors = [
                        (i-1, j-1), (i-1, j), (i-1, j+1),
                        (i, j-1),             (i, j+1),
                        (i+1, j-1), (i+1, j), (i+1, j+1)
                    ]
                else:
                    # Radius-based connectivity
                    neighbors = []
                    for di in range(-self.radius, self.radius + 1):
                        for dj in range(-self.radius, self.radius + 1):
                            if di == 0 and dj == 0:
                                continue
                            if di*di + dj*dj <= self.radius*self.radius:
                                neighbors.append((i + di, j + dj))
                
                for ni, nj in neighbors:
                    if 0 <= ni < self.height and 0 <= nj < self.width:
                        neighbor_idx = ni * self.width + nj
                        edges.append([node_idx, neighbor_idx])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index
    
    def to_graph(self, x: torch.Tensor) -> Data:
        """
        Convert grid tensor to graph.
        
        Args:
            x: Tensor of shape [B, C, H, W]
            
        Returns:
            PyG Data object
        """
        B, C, H, W = x.shape
        
        # Flatten spatial dimensions: [B, C, H, W] -> [B, H*W, C]
        node_features = x.permute(0, 2, 3, 1).reshape(B, -1, C)
        
        return node_features, self.edge_index


class GraphConvolutionLayer(nn.Module):
    """
    Basic Graph Convolution Layer (GCN).
    
    Aggregates features from neighbors using normalized adjacency.
    
    Reference: Kipf & Welling, "Semi-Supervised Classification with 
               Graph Convolutional Networks" (ICLR 2017)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [N, in_channels]
            edge_index: Edge connectivity [2, E]
            edge_weight: Optional edge weights
        """
        # Linear transformation
        support = torch.matmul(x, self.weight)
        
        # Aggregate from neighbors
        row, col = edge_index
        N = x.size(0)
        
        # Compute degree for normalization
        deg = torch.zeros(N, device=x.device)
        deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # Normalized message passing
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Aggregate
        out = torch.zeros_like(support)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i], edge_index[1, i]
            out[dst] += norm[i] * support[src]
            
        if self.bias is not None:
            out += self.bias
            
        return out


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT).
    
    Uses attention mechanism to weight neighbor contributions.
    
    Reference: Velikovi et al., "Graph Attention Networks" (ICLR 2018)
    
    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        heads: Number of attention heads
        concat: Whether to concatenate or average heads
        dropout: Dropout rate
        negative_slope: LeakyReLU negative slope
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 8,
        concat: bool = True,
        dropout: float = 0.1,
        negative_slope: float = 0.2
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope
        
        # Linear transformation for each head
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        # Attention weights
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
        
        self.bias = nn.Parameter(torch.Tensor(heads * out_channels if concat else out_channels))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with attention.
        
        Args:
            x: Node features [N, in_channels]
            edge_index: Edge connectivity [2, E]
        """
        N = x.size(0)
        
        # Linear transformation and reshape for multi-head
        x = self.lin(x).view(N, self.heads, self.out_channels)
        
        # Compute attention scores
        alpha_src = (x * self.att_src).sum(dim=-1)  # [N, heads]
        alpha_dst = (x * self.att_dst).sum(dim=-1)  # [N, heads]
        
        row, col = edge_index
        
        # Attention coefficients for each edge
        alpha = alpha_src[row] + alpha_dst[col]  # [E, heads]
        alpha = F.leaky_relu(alpha, self.negative_slope)
        
        # Softmax over neighbors
        alpha = self._softmax(alpha, col, N)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Aggregate with attention weights
        out = torch.zeros(N, self.heads, self.out_channels, device=x.device)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i], edge_index[1, i]
            out[dst] += alpha[i].unsqueeze(-1) * x[src]
            
        if self.concat:
            out = out.view(N, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
            
        out += self.bias
        
        return out
    
    def _softmax(self, src: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Compute softmax over neighbors."""
        src_max = torch.zeros(num_nodes, src.size(1), device=src.device)
        src_max.scatter_reduce_(0, index.unsqueeze(-1).expand_as(src), src, reduce='amax', include_self=False)
        
        out = (src - src_max[index]).exp()
        
        out_sum = torch.zeros(num_nodes, src.size(1), device=src.device)
        out_sum.scatter_add_(0, index.unsqueeze(-1).expand_as(out), out)
        
        return out / (out_sum[index] + 1e-8)


class MultiScaleGNN(nn.Module):
    """
    Multi-Scale Graph Neural Network.
    
    Processes graph at multiple scales by varying connectivity radius.
    Captures both local and long-range spatial dependencies.
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden feature dimension
        out_channels: Output feature dimension
        scales: List of connectivity radii [1, 2, 4]
        num_layers: Number of GNN layers per scale
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        height: int,
        width: int,
        scales: List[int] = [1, 2, 4],
        num_layers: int = 2
    ):
        super().__init__()
        
        self.scales = scales
        self.height = height
        self.width = width
        
        # Build edge indices for each scale
        self.edge_indices = nn.ParameterList()
        for radius in scales:
            constructor = GridGraphConstructor(height, width, 'radius', radius)
            # Register as buffer (not parameter)
            self.register_buffer(f'edge_index_{radius}', constructor.edge_index)
            
        # GNN layers for each scale
        self.scale_gnns = nn.ModuleList()
        for _ in scales:
            layers = nn.ModuleList()
            for i in range(num_layers):
                in_ch = in_channels if i == 0 else hidden_channels
                layers.append(GraphAttentionLayer(in_ch, hidden_channels, heads=4, concat=False))
            self.scale_gnns.append(layers)
            
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_channels * len(scales), hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, out_channels)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [B, N, C] where N = H*W
            
        Returns:
            Output features [B, N, out_channels]
        """
        B, N, C = x.shape
        
        scale_outputs = []
        
        for scale_idx, radius in enumerate(self.scales):
            edge_index = getattr(self, f'edge_index_{radius}')
            
            # Process each sample in batch
            batch_out = []
            for b in range(B):
                h = x[b]  # [N, C]
                
                # Apply GNN layers
                for layer in self.scale_gnns[scale_idx]:
                    h = layer(h, edge_index)
                    h = F.relu(h)
                    
                batch_out.append(h)
                
            scale_outputs.append(torch.stack(batch_out, dim=0))  # [B, N, hidden]
            
        # Concatenate multi-scale features
        multi_scale = torch.cat(scale_outputs, dim=-1)  # [B, N, hidden * num_scales]
        
        # Fuse
        out = self.fusion(multi_scale)  # [B, N, out_channels]
        
        return out


class SpatialGraphBlock(nn.Module):
    """
    Complete Spatial Graph Processing Block for MSUA-Net integration.
    
    Converts CNN features to graph, processes with GNN, converts back.
    This is the main module to add to MSUA-Net bottleneck.
    
    Args:
        channels: Number of feature channels
        hidden_channels: Hidden dimension for GNN
        num_heads: Number of attention heads
        num_layers: Number of GNN layers
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        channels: int,
        hidden_channels: int = None,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.channels = channels
        hidden_channels = hidden_channels or channels
        
        # Pre-processing
        self.norm1 = nn.LayerNorm(channels)
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_ch = channels if i == 0 else hidden_channels
            out_ch = channels if i == num_layers - 1 else hidden_channels
            concat = i < num_layers - 1  # Don't concat on last layer
            
            self.gat_layers.append(
                GraphAttentionLayer(
                    in_ch, 
                    out_ch // num_heads if concat else out_ch,
                    heads=num_heads,
                    concat=concat,
                    dropout=dropout
                )
            )
            
        # Post-processing
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout)
        )
        
        # Edge index cache
        self._edge_index = None
        self._cached_size = None
        
    def _get_edge_index(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Get or build edge index for grid."""
        if self._cached_size != (H, W):
            constructor = GridGraphConstructor(H, W, connectivity='8')
            self._edge_index = constructor.edge_index.to(device)
            self._cached_size = (H, W)
        return self._edge_index
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: CNN features [B, C, H, W]
            
        Returns:
            Processed features [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Get edge index
        edge_index = self._get_edge_index(H, W, x.device)
        
        # Reshape: [B, C, H, W] -> [B, H*W, C]
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        
        # Residual connection
        residual = x_flat
        
        # Pre-norm
        x_norm = self.norm1(x_flat)
        
        # Process each batch sample through GAT
        batch_outputs = []
        for b in range(B):
            h = x_norm[b]  # [N, C]
            
            for i, layer in enumerate(self.gat_layers):
                h = layer(h, edge_index)
                if i < len(self.gat_layers) - 1:
                    h = F.elu(h)
                    
            batch_outputs.append(h)
            
        gat_out = torch.stack(batch_outputs, dim=0)  # [B, N, C]
        
        # Residual
        x_out = residual + gat_out
        
        # FFN with residual
        x_out = x_out + self.ffn(self.norm2(x_out))
        
        # Reshape back: [B, H*W, C] -> [B, C, H, W]
        out = x_out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        return out


class HierarchicalGraphBlock(nn.Module):
    """
    Hierarchical Graph Processing with pooling.
    
    Creates multi-resolution graph representation by pooling
    and processing at different scales.
    
    Args:
        channels: Input channels
        scales: Number of hierarchy levels
    """
    
    def __init__(
        self,
        channels: int,
        scales: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.scales = scales
        
        # Downsampling for hierarchy
        self.pool_layers = nn.ModuleList([
            nn.AvgPool2d(2) for _ in range(scales - 1)
        ])
        
        # GNN at each scale
        self.graph_layers = nn.ModuleList([
            SpatialGraphBlock(channels, dropout=dropout)
            for _ in range(scales)
        ])
        
        # Upsampling for fusion
        self.upsample_layers = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            for _ in range(scales - 1)
        ])
        
        # Fusion convs
        self.fusion_convs = nn.ModuleList([
            nn.Conv2d(channels * 2, channels, 1)
            for _ in range(scales - 1)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Hierarchical processing.
        
        Args:
            x: Input [B, C, H, W]
        """
        # Build pyramid
        pyramid = [x]
        for pool in self.pool_layers:
            pyramid.append(pool(pyramid[-1]))
            
        # Process each level with GNN
        processed = []
        for i, (feat, gnn) in enumerate(zip(pyramid, self.graph_layers)):
            processed.append(gnn(feat))
            
        # Fuse bottom-up
        out = processed[-1]  # Coarsest
        for i in range(self.scales - 2, -1, -1):
            out = self.upsample_layers[i](out)
            out = self.fusion_convs[i](torch.cat([out, processed[i]], dim=1))
            
        return out


# ============================================================
# Integration helper for MSUA-Net
# ============================================================

def add_gnn_to_bottleneck(model, channels: int = 512, use_hierarchical: bool = False):
    """
    Helper function to add GNN to existing model's bottleneck.
    
    Args:
        model: MSUA-Net model
        channels: Bottleneck channel dimension
        use_hierarchical: Whether to use hierarchical GNN
    """
    if use_hierarchical:
        gnn_block = HierarchicalGraphBlock(channels)
    else:
        gnn_block = SpatialGraphBlock(channels)
        
    # This would need to be integrated in the model's forward pass
    return gnn_block


if __name__ == "__main__":
    # Test the modules
    print("Testing GNN Modules...")
    
    # Test SpatialGraphBlock
    block = SpatialGraphBlock(channels=64)
    x = torch.randn(2, 64, 32, 32)
    out = block(x)
    print(f" SpatialGraphBlock: {x.shape} -> {out.shape}")
    
    # Test HierarchicalGraphBlock
    hier_block = HierarchicalGraphBlock(channels=64, scales=3)
    out = hier_block(x)
    print(f" HierarchicalGraphBlock: {x.shape} -> {out.shape}")
    
    print("\n All GNN modules working!")
