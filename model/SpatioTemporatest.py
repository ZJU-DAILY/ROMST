import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
import numpy as np
from dgl.nn import ChebConv
import dgl
from mamba_ssm import Mamba


class STGCNLayer(nn.Module):
 
    def __init__(self, in_channels, out_channels, K):
        super(STGCNLayer, self).__init__()

        self.graph_conv = ChebConv(in_channels, out_channels, K)
        self.linear = nn.Linear(out_channels, in_channels)

    def forward(self, g, x):
        batch_size = x.size(0)
        outputs = []

        for i in range(batch_size):
            node_features = x[i]
            
            out_graph = self.graph_conv(g, node_features)
            out_graph = torch.relu(out_graph)
            out_graph = self.linear(out_graph)
            outputs.append(out_graph.unsqueeze(0))

        return torch.cat(outputs, dim=0)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, temporal_features):
        embedded_features = self.value_embedding(temporal_features)
        return self.dropout(embedded_features)


class EncoderLayer(nn.Module):
    def __init__(self, mamba, mamba_r, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        
        self.mamba = mamba    
        self.mamba_r = mamba_r  
        
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        
    def forward(self, temporal_features):
        mamba_forward = self.mamba(temporal_features)
        mamba_backward = self.mamba_r(temporal_features.flip(dims=[1])).flip(dims=[1])
        mamba_output = mamba_forward + mamba_backward
        attn = 1 

        residual_1 = temporal_features + mamba_output
        
        ff_input = residual_1 = self.norm1(residual_1)
        ff_output = self.dropout(self.activation(self.conv1(ff_input.transpose(-1, 1))))
        ff_output = self.dropout(self.conv2(ff_output).transpose(-1, 1))

        return self.norm2(residual_1 + ff_output), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, temporal_features):
        attns = []

        for attn_layer in self.attn_layers:
            temporal_features, attn = attn_layer(temporal_features)
            attns.append(attn)

        if self.norm is not None:
            temporal_features = self.norm(temporal_features)

        return temporal_features, attns


class MGCN_block(nn.Module):
    def __init__(self, device, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, len_input):
        super(MGCN_block, self).__init__()

        self.enc_embedding = DataEmbedding_inverted(len_input, 512, 0.1)
        
        self.stgcn_layer = STGCNLayer(len_input, nb_chev_filter, K)
        
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, 
                                      kernel_size=(1, 1), stride=(1, time_strides))

        self.ln = nn.LayerNorm(nb_time_filter)
        
        self.encoder = Encoder(
            [
                EncoderLayer(
                    Mamba(
                        d_model=512,
                        d_state=32,
                        d_conv=2,
                        expand=1,
                    ),
                    Mamba(
                        d_model=512,
                        d_state=32,
                        d_conv=2,
                        expand=1,
                    ),
                    512,
                    2048,
                    dropout=0.1,
                    activation='gelu'
                ) for l in range(1)
            ],
            norm_layer=torch.nn.LayerNorm(512)
        )
        
        self.projector = nn.Linear(512, len_input, bias=True)
        self.projector1 = nn.Linear(nb_time_filter, in_channels, bias=True)
        self.projector2 = nn.Linear(in_channels, nb_time_filter, bias=True)
        
        self.device = device

    def forward(self, temporal_data, adj_matrix):
        original_input = temporal_data

        batch_size, num_of_vertices, num_of_features, num_of_timesteps = original_input.shape
        
        squeezed_input = torch.squeeze(original_input, dim=2).to(self.device)

        adj_tensor = torch.from_numpy(adj_matrix) if isinstance(adj_matrix, np.ndarray) else adj_matrix
        edge_index = adj_tensor.nonzero(as_tuple=False).T
        dgl_graph = dgl.graph((edge_index[0], edge_index[1]))
        dgl_graph = dgl_graph.to(self.device)

        # (B, N, T) -> (B, N, T)
        gcn_features = self.stgcn_layer(dgl_graph, squeezed_input)

        #(B, N, T) -> (B, N, 512)
        embedded_features = self.enc_embedding(gcn_features)

        # (B, N, 512) -> (B, N, 512)
        mamba_features, attention_weights = self.encoder(embedded_features)
        
        # (B, N, 512) -> (B, N, T)
        projected_features = self.projector(mamba_features).permute(0, 2, 1)[:, :, :num_of_vertices]
        projected_features = projected_features.permute(0, 2, 1)

        # (B, N, T) -> (B, N, T, 1)
        expanded_features = torch.unsqueeze(projected_features, dim=3)

        #(B, N, T, 1) -> (B, N, T, nb_time_filter)
        time_features = self.projector2(expanded_features)

        residual_features = self.residual_conv(original_input.permute(0, 2, 1, 3))
        residual_features = residual_features.permute(0, 2, 3, 1)

        fused_features = self.ln(F.relu(residual_features + time_features))

        # (B, N, T, nb_time_filter) -> (B, N, F, S)
        prediction = self.projector1(fused_features)
        prediction = prediction.permute(0, 1, 3, 2)

        return fused_features,prediction


class SpatioTemporalModel(nn.Module):

    
    def __init__(self, temporal_input_dim, output_dim, seq_len=12, pred_len=12):
        super().__init__()
        self.temporal_input_dim = temporal_input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.mgcn_block = MGCN_block(
            device=self.device,
            in_channels=temporal_input_dim,
            K=3,
            nb_chev_filter=64,
            nb_time_filter=64,
            time_strides=1,
            len_input=seq_len
        )

    def forward(self, temporal_data, adj_matrix, return_prediction=True):
        batch_size, seq_len, num_nodes, feature_dim = temporal_data.shape
        
        # (B, S, N, F) -> (B, N, F, S)
        mgcn_input = temporal_data.permute(0, 2, 3, 1)

        fused_features, final_prediction = self.mgcn_block(mgcn_input, adj_matrix)
       
        # (B, N, T, nb_time_filter) -> (B, S, N, nb_time_filter)
        fused_features = fused_features.permute(0, 2, 1, 3)  # (B, N, T, 64) -> (B, T, N, 64)
        final_prediction = final_prediction.permute(0, 3, 1, 2)  # (B, N, F, S) -> (B, S, N, F)
        
        if return_prediction:
            return final_prediction, fused_features
        
        return fused_features
