"""
STGAN network for GARN (Group-Aware Robot Navigation, Lu et al. RA-L 2025).
Implemented as a comparison baseline for TAGA.

Architecture (paper Section IV-C):
  1. Attention Extraction: parallel MLPs (f_r, f_ind, f_grp) per agent type
     → bilinear attention matrix A_t (Eq. 9)
  2. Spatial GCN: C^(l+1) = ReLU(A_t · C^l · W_g) (Eq. 10)
  3. Temporal LSTM: captures temporal dependencies across timesteps
  4. Value Estimation: actor-critic MLP heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from rl.networks.network_utils import init


def reshapeT(T, seq_length, nenv):
    shape = T.size()[1:]
    return T.unsqueeze(0).reshape((seq_length, nenv, *shape))


class AttentionExtraction(nn.Module):
    """
    Attention extraction module with parallel MLPs for different agent types.
    Each agent type has its own MLP that maps raw features to a shared
    embedding space. Attention weights are computed via bilinear attention.
    (GARN paper Section IV-C-2, Eq. 9)

    Paper MLP sizes: f_r=(256,128), f_ind=(256,128), f_grp=(128,128)
    """
    def __init__(self, robot_input_dim, human_input_dim, group_input_dim, embed_dim):
        super(AttentionExtraction, self).__init__()
        # f_r: robot embedding MLP
        self.robot_mlp = nn.Sequential(
            nn.Linear(robot_input_dim, 256), nn.ReLU(),
            nn.Linear(256, embed_dim), nn.ReLU()
        )
        # f_ind: individual (human) embedding MLP
        self.human_mlp = nn.Sequential(
            nn.Linear(human_input_dim, 256), nn.ReLU(),
            nn.Linear(256, embed_dim), nn.ReLU()
        )
        # f_grp: group embedding MLP
        self.group_mlp = nn.Sequential(
            nn.Linear(group_input_dim, 128), nn.ReLU(),
            nn.Linear(128, embed_dim), nn.ReLU()
        )
        # W_r: learnable bilinear attention weight (paper: dim 128)
        self.W_r = nn.Parameter(torch.empty(embed_dim, embed_dim))
        nn.init.orthogonal_(self.W_r)
        self.scale = embed_dim ** 0.5

    def forward(self, robot_state, human_states, group_states, node_mask):
        """
        Args:
            robot_state:  [seq, nenv, 1, robot_dim]
            human_states: [seq, nenv, H, human_dim]
            group_states: [seq, nenv, G, group_dim]
            node_mask:    [seq, nenv, N] bool (True = valid node)
        Returns:
            E_t: [seq, nenv, N, embed_dim] node embeddings
            A_t: [seq, nenv, N, N] attention matrix (row-normalized)
        """
        robot_embed = self.robot_mlp(robot_state)    # [seq, nenv, 1, embed]
        human_embed = self.human_mlp(human_states)   # [seq, nenv, H, embed]
        group_embed = self.group_mlp(group_states)   # [seq, nenv, G, embed]

        # Stack all node embeddings: [robot, human_0, ..., human_H-1, grp_0, ..., grp_G-1]
        E_t = torch.cat([robot_embed, human_embed, group_embed], dim=2)

        # Bilinear attention: A[i,j] = x_i^T W_r x_j / sqrt(d)
        E_W = torch.matmul(E_t, self.W_r)                    # [seq, nenv, N, embed]
        A_t = torch.matmul(E_W, E_t.transpose(-1, -2))       # [seq, nenv, N, N]
        A_t = A_t / self.scale

        # Mask invalid columns (don't attend TO invalid nodes)
        col_mask = node_mask.unsqueeze(-2).expand_as(A_t)     # [seq, nenv, N, N]
        A_t = A_t.masked_fill(~col_mask, -1e9)

        A_t = F.softmax(A_t, dim=-1)

        # Zero out rows of invalid nodes (their representations won't be used)
        row_mask = node_mask.unsqueeze(-1).float()            # [seq, nenv, N, 1]
        A_t = A_t * row_mask

        return E_t, A_t


class SpatialGCN(nn.Module):
    """
    Spatial GCN with layer-wise propagation.
    C^(l+1) = ReLU(A_t · C^l · W_g^l)
    (GARN paper Section IV-C-1, Eq. 10)
    """
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(SpatialGCN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            self.layers.append(nn.Linear(in_d, hidden_dim, bias=False))

    def forward(self, C, A_t):
        """
        Args:
            C:   [seq, nenv, N, d] initial node features (C^0 = E_t)
            A_t: [seq, nenv, N, N] attention/adjacency matrix
        Returns:
            C:   [seq, nenv, N, hidden_dim] output node features
        """
        for linear in self.layers:
            AC = torch.matmul(A_t, C)       # aggregate neighbor features
            C = F.relu(linear(AC))           # transform
        return C


class STGAN(nn.Module):
    """
    Full STGAN network for GARN baseline.

    Pipeline:
      obs → AttentionExtraction(MLPs + bilinear attn) → A_t, E_t
        → SpatialGCN(A_t, E_t) → C^L
        → extract robot node → LSTM → project → actor/critic

    Interface matches the existing codebase (model.py Policy wrapper):
      __init__(obs_space_dict, args)
      forward(inputs, rnn_hxs, masks, infer=False) → (value, actor_feat, rnn_hxs)
    """

    def __init__(self, obs_space_dict, args):
        super(STGAN, self).__init__()
        self.is_recurrent = True
        self.args = args

        self.human_num = obs_space_dict['spatial_edges'].shape[0]
        self.seq_length = args.num_steps
        self.nenv = args.num_processes
        self.nminibatch = args.num_mini_batch

        # RNN sizes (must match storage allocation)
        self.human_node_rnn_size = args.human_node_rnn_size
        self.human_human_edge_rnn_size = args.human_human_edge_rnn_size
        self.output_size = args.human_node_output_size   # 256

        # GARN-specific hyperparameters
        self.embed_dim = args.garn_gcn_hidden             # 128
        self.gcn_hidden = args.garn_gcn_hidden            # 128
        self.lstm_hidden = args.garn_lstm_hidden           # 128
        self.gcn_num_layers = args.garn_gcn_layers         # 2

        # Input dimensions
        robot_input_dim = 9   # temporal_edges(2) + robot_node(7)
        spatial_dim = obs_space_dict['spatial_edges'].shape[1]  # 12 for pred envs

        # Group dimensions
        if 'group_centroids' in obs_space_dict:
            self.num_groups = obs_space_dict['group_centroids'].shape[0]
        else:
            self.num_groups = 0
        group_input_dim = 3   # centroid_x, centroid_y, radius

        # Total graph nodes: 1 robot + H humans + G groups
        self.total_nodes = 1 + self.human_num + self.num_groups

        init_ = lambda m: init(m, nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        # --- Attention Extraction Module ---
        self.attention_extraction = AttentionExtraction(
            robot_input_dim, spatial_dim, group_input_dim, self.embed_dim
        )

        # --- Spatial GCN ---
        self.spatial_gcn = SpatialGCN(
            self.embed_dim, self.gcn_hidden, self.gcn_num_layers
        )

        # --- Temporal LSTM ---
        self.lstm = nn.LSTM(
            input_size=self.gcn_hidden,
            hidden_size=self.lstm_hidden,
            num_layers=1,
            batch_first=False
        )

        # Project LSTM output → output_size (256)
        self.output_linear = init_(nn.Linear(self.lstm_hidden, self.output_size))

        # --- Actor-Critic heads ---
        self.actor = nn.Sequential(
            init_(nn.Linear(self.output_size, self.output_size)), nn.Tanh(),
            init_(nn.Linear(self.output_size, self.output_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(self.output_size, self.output_size)), nn.Tanh(),
            init_(nn.Linear(self.output_size, self.output_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(self.output_size, 1))

    @property
    def recurrent_hidden_state_size(self):
        return self.human_node_rnn_size

    def _build_node_mask(self, detected_human_num, seq_length, nenv, device):
        """
        Build boolean validity mask for all graph nodes.
        Returns: [seq_length, nenv, total_nodes] with True for valid nodes.
        """
        N = self.total_nodes
        mask = torch.zeros(seq_length, nenv, N, dtype=torch.bool, device=device)

        # Robot node is always valid
        mask[:, :, 0] = True

        # Human nodes: valid for indices < detected_human_num (humans sorted to front)
        det = detected_human_num.to(device)                       # [seq*nenv] or [seq*nenv, 1]
        det = det.view(seq_length, nenv)                          # [seq, nenv]
        human_idx = torch.arange(self.human_num, device=device)   # [H]
        human_valid = human_idx.view(1, 1, -1) < det.unsqueeze(-1)  # [seq, nenv, H]
        mask[:, :, 1:1 + self.human_num] = human_valid

        # Group nodes: mark valid when at least one group exists
        # (groups with zero radius are effectively empty but the GCN
        #  attention will learn to ignore them)
        if self.num_groups > 0:
            mask[:, :, 1 + self.human_num:] = True

        return mask

    def forward(self, inputs, rnn_hxs, masks, infer=False):
        if infer:
            seq_length = 1
            nenv = self.nenv
        else:
            seq_length = self.seq_length
            nenv = self.nenv // self.nminibatch

        # --- Extract observations and reshape to [seq, nenv, ...] ---
        robot_node = reshapeT(inputs['robot_node'], seq_length, nenv)
        temporal_edges = reshapeT(inputs['temporal_edges'], seq_length, nenv)
        spatial_edges = reshapeT(inputs['spatial_edges'], seq_length, nenv)

        device = robot_node.device

        # Robot state: [seq, nenv, 1, 9]
        robot_state = torch.cat((temporal_edges, robot_node), dim=-1)

        # Human states: [seq, nenv, H, spatial_dim]
        human_states = spatial_edges

        # Group states: [seq, nenv, G, 3]  (centroid_x, centroid_y, radius)
        if self.num_groups > 0 and 'group_centroids' in inputs:
            group_centroids = reshapeT(inputs['group_centroids'], seq_length, nenv)
            group_radii = reshapeT(inputs['group_radii'], seq_length, nenv)
            group_states = torch.cat(
                [group_centroids, group_radii.unsqueeze(-1)], dim=-1
            )
        else:
            group_states = torch.zeros(
                seq_length, nenv, max(self.num_groups, 1), 3,
                device=device, dtype=robot_node.dtype
            )
            # If num_groups is 0, we still need 1 dummy group for cat to work
            if self.num_groups == 0:
                group_states = group_states[:, :, :0, :]  # empty tensor

        # --- Node validity mask ---
        detected_human_num = inputs['detected_human_num'].squeeze(-1)
        node_mask = self._build_node_mask(
            detected_human_num, seq_length, nenv, device
        )

        # --- Attention Extraction ---
        E_t, A_t = self.attention_extraction(
            robot_state, human_states, group_states, node_mask
        )

        # --- Spatial GCN ---
        C = self.spatial_gcn(E_t, A_t)   # [seq, nenv, N, gcn_hidden]

        # Extract robot node representation (node index 0)
        robot_repr = C[:, :, 0, :]       # [seq, nenv, gcn_hidden]

        # --- Temporal LSTM ---
        # Recover LSTM hidden state (h) from human_node_rnn
        hidden_states_node = reshapeT(rnn_hxs['human_node_rnn'], 1, nenv)
        # hidden_states_node: [1, nenv, 1, human_node_rnn_size]
        h_0 = hidden_states_node[:, :, 0, :self.lstm_hidden].contiguous()

        # Recover LSTM cell state (c) from human_human_edge_rnn (first row)
        edge_rnn = reshapeT(rnn_hxs['human_human_edge_rnn'], 1, nenv)
        # edge_rnn: [1, nenv, edge_num, edge_rnn_size]
        c_0 = edge_rnn[:, :, 0, :self.lstm_hidden].contiguous()

        # Apply masks: reset hidden state on episode boundaries.
        # masks_reshaped: [seq, nenv, 1] broadcasts to h/c [1, nenv, lstm_hidden]
        masks_reshaped = reshapeT(masks, seq_length, nenv)

        if seq_length == 1:
            # Rollout path: single step, just mask initial h/c
            h_0 = h_0 * masks_reshaped
            c_0 = c_0 * masks_reshaped
            lstm_out, (h_n, c_n) = self.lstm(robot_repr, (h_0, c_0))
        else:
            # Training path: iterate timesteps to correctly reset h/c at
            # episode boundaries within the unrolled sequence.
            outputs = []
            h, c = h_0, c_0
            for t in range(seq_length):
                m_t = masks_reshaped[t:t + 1]                # [1, nenv, 1]
                h = h * m_t
                c = c * m_t
                out_t, (h, c) = self.lstm(robot_repr[t:t + 1], (h, c))
                outputs.append(out_t)
            lstm_out = torch.cat(outputs, dim=0)             # [seq, nenv, lstm_hidden]
            h_n, c_n = h, c
        # lstm_out: [seq, nenv, lstm_hidden]

        # --- Project to output size ---
        x = self.output_linear(lstm_out)  # [seq, nenv, output_size]

        # --- Actor-Critic ---
        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        # --- Update rnn_hxs ---
        # Store LSTM h in human_node_rnn
        new_node_rnn = torch.zeros(
            1, nenv, 1, self.human_node_rnn_size,
            device=device, dtype=x.dtype
        )
        new_node_rnn[:, :, 0, :self.lstm_hidden] = h_n
        rnn_hxs['human_node_rnn'] = new_node_rnn.squeeze(0)

        # Store LSTM c in human_human_edge_rnn (first row)
        edge_num = 1 + self.human_num
        new_edge_rnn = torch.zeros(
            nenv, edge_num, self.human_human_edge_rnn_size,
            device=device, dtype=x.dtype
        )
        new_edge_rnn[:, 0, :self.lstm_hidden] = c_n.squeeze(0)
        rnn_hxs['human_human_edge_rnn'] = new_edge_rnn

        if infer:
            return (self.critic_linear(hidden_critic).squeeze(0),
                    hidden_actor.squeeze(0),
                    rnn_hxs)
        else:
            return (self.critic_linear(hidden_critic).view(-1, 1),
                    hidden_actor.view(-1, self.output_size),
                    rnn_hxs)
