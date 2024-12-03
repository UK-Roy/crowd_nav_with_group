import torch.nn.functional as F

from .srnn_model import *

class GroupAttention_M(nn.Module):
    """
    Class for the robot-group attention module.
    """
    def __init__(self, args):
        super(GroupAttention_M, self).__init__()
        self.args = args
        self.group_attention_size = args.group_attention_size
        self.attention_size = args.attention_size
        self.robot_embed_dim = args.human_node_embedding_size 
        self.group_embed_dim = args.group_node_embedding_size  

        # Linear layers for embedding robot and group states
        self.robot_embedding_layer = nn.Linear(256, self.robot_embed_dim)
        self.group_embedding_layer = nn.Linear(self.group_embed_dim, self.attention_size)

        # Layer for computing attention scores
        self.attention_layer = nn.Linear(self.group_attention_size + self.robot_embed_dim, 1)

    def forward(self, robot_states, group_embeddings):
        """
        Forward pass for robot-group interaction.
        params:
        - robot_state: Robot state features (seq_len, nenv, robot_dim)
        - group_embeddings: Group features (seq_len, nenv, num_groups, group_dim)
        """
        seq_len, nenv, num_groups, group_dim = group_embeddings.shape
        
        # Check for no groups detected
        if num_groups == 0:
            # if infer_phase:
            # Create dummy group embeddings for inference
            dummy_group_embedding = torch.zeros(
                (seq_len, nenv, 1, self.group_embed_dim), device=robot_states.device, dtype=robot_states.dtype
            )
            group_embeddings = dummy_group_embedding
            num_groups = 1  # Update number of groups to 1

        # Embed robot state
        robot_embed = self.robot_embedding_layer(robot_states)  # (seq_len, nenv, robot_embed_dim)

        # Embed group states
        group_embed = self.group_embedding_layer(group_embeddings)  # (seq_len, nenv, num_groups, attention_size)

        # Compute attention scores
        robot_embed = robot_embed.repeat_interleave(num_groups, dim=2)  # (seq_len, nenv, num_groups, robot_embed_dim)
        combined_features = torch.cat([robot_embed, group_embed], dim=-1)  # Concatenate robot and group embeddings
        attention_scores = self.attention_layer(combined_features).squeeze(-1)  # (seq_len, nenv, num_groups)

        # Normalize attention scores
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (seq_len, nenv, num_groups)

        # Compute weighted group interaction
        weighted_group_interaction = torch.sum(
            attention_weights.unsqueeze(-1) * group_embeddings, dim=2
        )  # (seq_len, nenv, group_dim)

        return weighted_group_interaction, attention_weights


class SpatialEdgeSelfAttn(nn.Module):
    """
    Class for the human-human attention,
    uses a multi-head self attention proposed by https://arxiv.org/abs/1706.03762
    """
    def __init__(self, args):
        super(SpatialEdgeSelfAttn, self).__init__()
        self.args = args

        # Store required sizes
        # todo: hard-coded for now
        # with human displacement: + 2
        # pred 4 steps + disp: 12
        # pred 4 steps + no disp: 10
        # pred 5 steps + no disp: 12
        # pred 5 steps + no disp + probR: 17
        # Gaussian pred 5 steps + no disp: 27
        # pred 8 steps + no disp: 18
        if args.env_name in ['CrowdSimPred-v0', 'CrowdSimPredRealGST-v0']:
            self.input_size = 12
        elif args.env_name == 'CrowdSimVarNum-v0':
            self.input_size = 2 # 4
        else:
            raise NotImplementedError
        self.num_attn_heads=8
        self.attn_size=512


        # Linear layer to embed input
        self.embedding_layer = nn.Sequential(nn.Linear(self.input_size, 128), nn.ReLU(),
                                             nn.Linear(128, self.attn_size), nn.ReLU()
                                             )

        self.q_linear = nn.Linear(self.attn_size, self.attn_size)
        self.v_linear = nn.Linear(self.attn_size, self.attn_size)
        self.k_linear = nn.Linear(self.attn_size, self.attn_size)

        # multi-head self attention
        self.multihead_attn=torch.nn.MultiheadAttention(self.attn_size, self.num_attn_heads)


    # Given a list of sequence lengths, create a mask to indicate which indices are padded
    # e.x. Input: [3, 1, 4], max_human_num = 5
    # Output: [[1, 1, 1, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 0]]
    def create_attn_mask(self, each_seq_len, seq_len, nenv, max_human_num):
        # mask with value of False means padding and should be ignored by attention
        # why +1: use a sentinel in the end to handle the case when each_seq_len = 18
        if self.args.no_cuda:
            mask = torch.zeros(seq_len * nenv, max_human_num + 1).cpu()
        else:
            mask = torch.zeros(seq_len*nenv, max_human_num+1).cuda()
        mask[torch.arange(seq_len*nenv), each_seq_len.long()] = 1.
        mask = torch.logical_not(mask.cumsum(dim=1))
        # remove the sentinel
        mask = mask[:, :-1].unsqueeze(-2) # seq_len*nenv, 1, max_human_num
        return mask


    def forward(self, inp, each_seq_len):
        '''
        Forward pass for the model
        params:
        inp : input edge features
        each_seq_len:
        if self.args.sort_humans is True, the true length of the sequence. Should be the number of detected humans
        else, it is the mask itself
        '''
        # inp is padded sequence [seq_len, nenv, max_human_num, 2]
        seq_len, nenv, max_human_num, _ = inp.size()
        if self.args.sort_humans:
            attn_mask = self.create_attn_mask(each_seq_len, seq_len, nenv, max_human_num)  # [seq_len*nenv, 1, max_human_num]
            attn_mask = attn_mask.squeeze(1)  # if we use pytorch builtin function
        else:
            # combine the first two dimensions
            attn_mask = each_seq_len.reshape(seq_len*nenv, max_human_num)


        input_emb=self.embedding_layer(inp).view(seq_len*nenv, max_human_num, -1)
        input_emb=torch.transpose(input_emb, dim0=0, dim1=1) # if we use pytorch builtin function, v1.7.0 has no batch first option
        q=self.q_linear(input_emb)
        k=self.k_linear(input_emb)
        v=self.v_linear(input_emb)

        #z=self.multihead_attn(q, k, v, mask=attn_mask)
        z,_=self.multihead_attn(q, k, v, key_padding_mask=torch.logical_not(attn_mask)) # if we use pytorch builtin function
        z=torch.transpose(z, dim0=0, dim1=1) # if we use pytorch builtin function
        return z



class EdgeAttention_M(nn.Module):
    '''
    Class for the robot-human attention module
    '''
    def __init__(self, args):
        '''
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        '''
        super(EdgeAttention_M, self).__init__()

        self.args = args

        # Store required sizes
        self.human_human_edge_rnn_size = args.human_human_edge_rnn_size
        self.human_node_rnn_size = args.human_node_rnn_size
        self.attention_size = args.attention_size



        # Linear layer to embed temporal edgeRNN hidden state
        self.temporal_edge_layer=nn.ModuleList()
        self.spatial_edge_layer=nn.ModuleList()

        self.temporal_edge_layer.append(nn.Linear(self.human_human_edge_rnn_size, self.attention_size))

        # Linear layer to embed spatial edgeRNN hidden states
        self.spatial_edge_layer.append(nn.Linear(self.human_human_edge_rnn_size, self.attention_size))



        # number of agents who have spatial edges (complete graph: all 6 agents; incomplete graph: only the robot)
        self.agent_num = 1
        self.num_attention_head = 1

    def create_attn_mask(self, each_seq_len, seq_len, nenv, max_human_num):
        # mask with value of False means padding and should be ignored by attention
        # why +1: use a sentinel in the end to handle the case when each_seq_len = 18
        if self.args.no_cuda:
            mask = torch.zeros(seq_len * nenv, max_human_num + 1).cpu()
        else:
            mask = torch.zeros(seq_len * nenv, max_human_num + 1).cuda()
        mask[torch.arange(seq_len * nenv), each_seq_len.long()] = 1.
        mask = torch.logical_not(mask.cumsum(dim=1))
        # remove the sentinel
        mask = mask[:, :-1].unsqueeze(-2)  # seq_len*nenv, 1, max_human_num
        return mask

    def att_func(self, temporal_embed, spatial_embed, h_spatials, attn_mask=None):
        seq_len, nenv, num_edges, h_size = h_spatials.size()  # [1, 12, 30, 256] in testing,  [12, 30, 256] in training
        attn = temporal_embed * spatial_embed
        attn = torch.sum(attn, dim=3)

        # Variable length
        temperature = num_edges / np.sqrt(self.attention_size)
        attn = torch.mul(attn, temperature)

        # if we don't want to mask invalid humans, attn_mask is None and no mask will be applied
        # else apply attn masks
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, -1e9)

        # Softmax
        attn = attn.view(seq_len, nenv, self.agent_num, self.human_num)
        attn = torch.nn.functional.softmax(attn, dim=-1)
        # print(attn[0, 0, 0].cpu().numpy())

        # Compute weighted value
        # weighted_value = torch.mv(torch.t(h_spatials), attn)

        # reshape h_spatials and attn
        # shape[0] = seq_len, shape[1] = num of spatial edges (6*5 = 30), shape[2] = 256
        h_spatials = h_spatials.view(seq_len, nenv, self.agent_num, self.human_num, h_size)
        h_spatials = h_spatials.view(seq_len * nenv * self.agent_num, self.human_num, h_size).permute(0, 2,
                                                                                         1)  # [seq_len*nenv*6, 5, 256] -> [seq_len*nenv*6, 256, 5]

        attn = attn.view(seq_len * nenv * self.agent_num, self.human_num).unsqueeze(-1)  # [seq_len*nenv*6, 5, 1]
        weighted_value = torch.bmm(h_spatials, attn)  # [seq_len*nenv*6, 256, 1]

        # reshape back
        weighted_value = weighted_value.squeeze(-1).view(seq_len, nenv, self.agent_num, h_size)  # [seq_len, 12, 6 or 1, 256]
        return weighted_value, attn



    # h_temporal: [seq_len, nenv, 1, 256]
    # h_spatials: [seq_len, nenv, 5, 256]
    def forward(self, h_temporal, h_spatials, each_seq_len):
        '''
        Forward pass for the model
        params:
        h_temporal : Hidden state of the temporal edgeRNN
        h_spatials : Hidden states of all spatial edgeRNNs connected to the node.
        each_seq_len:
            if self.args.sort_humans is True, the true length of the sequence. Should be the number of detected humans
            else, it is the mask itself
        '''
        seq_len, nenv, max_human_num, _ = h_spatials.size()
        # find the number of humans by the size of spatial edgeRNN hidden state
        self.human_num = max_human_num // self.agent_num

        weighted_value_list, attn_list=[],[]
        for i in range(self.num_attention_head):

            # Embed the temporal edgeRNN hidden state
            temporal_embed = self.temporal_edge_layer[i](h_temporal)
            # temporal_embed = temporal_embed.squeeze(0)

            # Embed the spatial edgeRNN hidden states
            spatial_embed = self.spatial_edge_layer[i](h_spatials)

            # Dot based attention
            temporal_embed = temporal_embed.repeat_interleave(self.human_num, dim=2)

            if self.args.sort_humans:
                attn_mask = self.create_attn_mask(each_seq_len, seq_len, nenv, max_human_num)  # [seq_len*nenv, 1, max_human_num]
                attn_mask = attn_mask.squeeze(-2).view(seq_len, nenv, max_human_num)
            else:
                attn_mask = each_seq_len
            weighted_value,attn=self.att_func(temporal_embed, spatial_embed, h_spatials, attn_mask=attn_mask)
            weighted_value_list.append(weighted_value)
            attn_list.append(attn)

        if self.num_attention_head > 1:
            return self.final_attn_linear(torch.cat(weighted_value_list, dim=-1)), attn_list
        else:
            return weighted_value_list[0], attn_list[0]

class EndRNN(RNNBase):
    '''
    Class for the GRU
    '''
    def __init__(self, args):
        '''
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        '''
        super(EndRNN, self).__init__(args, edge=False)

        self.args = args

        # Store required sizes
        self.rnn_size = args.human_node_rnn_size
        self.output_size = args.human_node_output_size
        self.embedding_size = args.human_node_embedding_size
        self.input_size = args.human_node_input_size
        self.edge_rnn_size = args.human_human_edge_rnn_size
        self.group_embedd_size = args.group_attention_size

        # Linear layer to embed input
        self.encoder_linear = nn.Linear(256, self.embedding_size)

        # ReLU and Dropout layers
        self.relu = nn.ReLU()

        # Linear layer to embed attention module output
        self.edge_attention_embed = nn.Linear(self.edge_rnn_size + self.group_embedd_size, self.embedding_size)


        # Output linear layer
        self.output_linear = nn.Linear(self.rnn_size, self.output_size)



    def forward(self, robot_s, h_spatial_other, h, masks):
        '''
        Forward pass for the model
        params:
        pos : input position
        h_temporal : hidden state of the temporal edgeRNN corresponding to this node
        h_spatial_other : output of the attention module
        h : hidden state of the current nodeRNN
        c : cell state of the current nodeRNN
        '''
        # Encode the input position
        encoded_input = self.encoder_linear(robot_s)
        encoded_input = self.relu(encoded_input)

        h_edges_embedded = self.relu(self.edge_attention_embed(h_spatial_other))

        concat_encoded = torch.cat((encoded_input, h_edges_embedded), -1)

        x, h_new = self._forward_gru(concat_encoded, h, masks)

        outputs = self.output_linear(x)


        return outputs, h_new

class selfAttn_merge_SRNN(nn.Module):
    """
    Class for the proposed network
    """
    def __init__(self, obs_space_dict, args, infer=False):
        """
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        """
        super(selfAttn_merge_SRNN, self).__init__()
        self.infer = infer
        self.is_recurrent = True
        self.args=args

        self.human_num = obs_space_dict['spatial_edges'].shape[0]

        self.seq_length = args.seq_length
        self.nenv = args.num_processes
        self.nminibatch = args.num_mini_batch

        # Store required sizes
        self.human_node_rnn_size = args.human_node_rnn_size
        self.human_human_edge_rnn_size = args.human_human_edge_rnn_size
        self.output_size = args.human_node_output_size

        # Initialize the Node and Edge RNNs
        self.humanNodeRNN = EndRNN(args)

        # Initialize attention module
        self.attn = EdgeAttention_M(args)
        
        # Initialize the Group attention module
        self.group_attn = GroupAttention_M(args)
        self.group_input_size=12+6


        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        num_inputs = hidden_size = self.output_size

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())


        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        robot_size = 9
        self.robot_linear = nn.Sequential(init_(nn.Linear(robot_size, 256)), nn.ReLU()) # todo: check dim
        self.human_node_final_linear=init_(nn.Linear(self.output_size,2))

        if self.args.use_self_attn:
            self.spatial_attn = SpatialEdgeSelfAttn(args)
            self.spatial_linear = nn.Sequential(init_(nn.Linear(512, 256)), nn.ReLU())
        else:
            self.spatial_linear = nn.Sequential(init_(nn.Linear(obs_space_dict['spatial_edges'].shape[1], 128)), nn.ReLU(),
                                                init_(nn.Linear(128, 256)), nn.ReLU())


        self.temporal_edges = [0]
        self.spatial_edges = np.arange(1, self.human_num+1)

        dummy_human_mask = [0] * self.human_num
        dummy_human_mask[0] = 1
        if self.args.no_cuda:
            self.dummy_human_mask = Variable(torch.Tensor([dummy_human_mask]).cpu())
        else:
            self.dummy_human_mask = Variable(torch.Tensor([dummy_human_mask]).cuda())
        
        
        # Group Embedding
        self.grp_embed = args.group_node_embedding_size
        # Group Linear layer to embed group input
        self.group_embedding_layer = nn.Sequential(nn.Linear(self.group_input_size, self.grp_embed), nn.ReLU(),
                                             nn.Linear(64, 64), nn.ReLU()
                                             )
        # ReLU and Dropout layers
        self.relu = nn.ReLU()



    def forward(self, inputs, rnn_hxs, masks, infer=False):
        if infer:
            # Test/rollout time
            seq_length = 1
            nenv = self.nenv

        else:
            # Training time
            seq_length = self.seq_length
            nenv = self.nenv // self.nminibatch

        robot_node = reshapeT(inputs['robot_node'], seq_length, nenv)
        temporal_edges = reshapeT(inputs['temporal_edges'], seq_length, nenv)
        spatial_edges = reshapeT(inputs['spatial_edges'], seq_length, nenv)
        # group_embeddings = reshapeT(inputs['group_embeddings'], seq_length, nenv)
        group_embeddings = self.compute_group_embeddings(inputs, seq_length, nenv)

        # to prevent errors in old models that does not have sort_humans argument
        if not hasattr(self.args, 'sort_humans'):
            self.args.sort_humans = True
        if self.args.sort_humans:
            detected_human_num = inputs['detected_human_num'].squeeze(-1).cpu().int()
        else:
            human_masks = reshapeT(inputs['visible_masks'], seq_length, nenv).float() # [seq_len, nenv, max_human_num]
            # if no human is detected (human_masks are all False, set the first human to True)
            human_masks[human_masks.sum(dim=-1)==0] = self.dummy_human_mask


        hidden_states_node_RNNs = reshapeT(rnn_hxs['human_node_rnn'], 1, nenv)
        masks = reshapeT(masks, seq_length, nenv)


        if self.args.no_cuda:
            all_hidden_states_edge_RNNs = Variable(
                torch.zeros(1, nenv, 1+self.human_num, rnn_hxs['human_human_edge_rnn'].size()[-1]).cpu())
        else:
            all_hidden_states_edge_RNNs = Variable(
                torch.zeros(1, nenv, 1+self.human_num, rnn_hxs['human_human_edge_rnn'].size()[-1]).cuda())

        robot_states = torch.cat((temporal_edges, robot_node), dim=-1)
        robot_states = self.robot_linear(robot_states)


        # attention modules
        if self.args.sort_humans:
            # human-human attention
            if self.args.use_self_attn:
                spatial_attn_out=self.spatial_attn(spatial_edges, detected_human_num).view(seq_length, nenv, self.human_num, -1)
            else:
                spatial_attn_out = spatial_edges
            output_spatial = self.spatial_linear(spatial_attn_out)

            # robot-human attention
            hidden_attn_weighted, _ = self.attn(robot_states, output_spatial, detected_human_num)
        else:
            # human-human attention
            if self.args.use_self_attn:
                spatial_attn_out = self.spatial_attn(spatial_edges, human_masks).view(seq_length, nenv, self.human_num, -1)
            else:
                spatial_attn_out = spatial_edges
            output_spatial = self.spatial_linear(spatial_attn_out)

            # robot-human attention
            hidden_attn_weighted, _ = self.attn(robot_states, output_spatial, human_masks)

        # robot-group attension
        group_weighted_value, group_attention_weights = self.group_attn(robot_states, group_embeddings)

        # h_individual_scaled = hidden_attn_weighted * alpha
        # h_group_scaled = group_weighted_value * (1 - alpha)
        
        # Combine interactions
        combined_attention = torch.cat([hidden_attn_weighted, group_weighted_value.unsqueeze(2)], dim=-1)

        
        # Do a forward pass through GRU
        outputs, h_nodes \
            = self.humanNodeRNN(robot_states, combined_attention, hidden_states_node_RNNs, masks)
            # = self.humanNodeRNN(robot_states, hidden_attn_weighted, hidden_states_node_RNNs, masks)


        # Update the hidden and cell states
        all_hidden_states_node_RNNs = h_nodes
        outputs_return = outputs

        rnn_hxs['human_node_rnn'] = all_hidden_states_node_RNNs
        rnn_hxs['human_human_edge_rnn'] = all_hidden_states_edge_RNNs


        # x is the output and will be sent to actor and critic
        x = outputs_return[:, :, 0, :]

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        for key in rnn_hxs:
            rnn_hxs[key] = rnn_hxs[key].squeeze(0)

        if infer:
            return self.critic_linear(hidden_critic).squeeze(0), hidden_actor.squeeze(0), rnn_hxs
        else:
            return self.critic_linear(hidden_critic).view(-1, 1), hidden_actor.view(-1, self.output_size), rnn_hxs
    
    def compute_group_embeddings(self, input_dict, seq_length, nenv):
        """
        Compute group embeddings for a multi-environment setup, ignoring `-1` clusters.
        """
        # Reshape inputs to (seq_length, nenv, ...)
        clusters = reshapeT(input_dict['clusters'], seq_length, nenv)  # (seq_len, nenv, num_humans)
        spatial_edges = reshapeT(input_dict['spatial_edges'], seq_length, nenv)  # (seq_len, nenv, num_humans, spatial_dim)
        velocity_edges = reshapeT(input_dict['velocity_edges'], seq_length, nenv)  # (seq_len, nenv, num_humans, velocity_dim)
        direction_consistency = reshapeT(input_dict['direction_consistency'], seq_length, nenv)  # (seq_len, nenv, num_humans)
        group_centroids = reshapeT(input_dict['group_centroids'], seq_length, nenv)  # (seq_len, nenv, num_groups, centroid_dim)
        group_radii = reshapeT(input_dict['group_radii'], seq_length, nenv)  # (seq_len, nenv, num_groups)

        # Mask for valid clusters (ignores -1)
        valid_mask = clusters != -1
        valid_clusters = clusters[valid_mask].long()

        # Check for empty valid clusters
        if valid_clusters.numel() == 0:
            return torch.zeros((seq_length, nenv, 0, 0))  # Return empty embeddings if no valid clusters

        # Get unique group IDs (ignoring -1)
        unique_groups = torch.unique(valid_clusters)

        # Group embedding list
        group_embeddings = []
        max_velocity_dim = velocity_edges.size(-1)
        max_spatial_dim = spatial_edges.size(-1)
        max_centroid_dim = group_centroids.size(-1)

        # Loop over unique groups
        for group_id in unique_groups:
            # Mask for individuals in the current group
            group_mask = clusters == group_id  # (seq_len, nenv, num_humans)

            # Aggregate features for the group
            group_velocity = torch.where(group_mask.unsqueeze(-1), velocity_edges, torch.zeros_like(velocity_edges))
            group_spatial = torch.where(group_mask.unsqueeze(-1), spatial_edges, torch.zeros_like(spatial_edges))
            group_direction = torch.where(group_mask, direction_consistency, torch.zeros_like(direction_consistency))

            # Sum pooling across humans in the group
            sum_mask = group_mask.sum(dim=2, keepdim=True).clamp(min=1)  # Avoid division by zero
            avg_velocity = group_velocity.sum(dim=2) / sum_mask
            avg_spatial = group_spatial.sum(dim=2) / sum_mask
            avg_direction = group_direction.sum(dim=2) / sum_mask.squeeze(-1)

            # Fetch group-level features (centroid and radius)
            centroid = group_centroids[:, :, group_id]  # (seq_len, nenv, centroid_dim)
            radius = group_radii[:, :, group_id].unsqueeze(-1)  # (seq_len, nenv, 1)

            # Concatenate group features
            group_embedding = torch.cat([avg_velocity, avg_spatial, avg_direction.unsqueeze(-1), centroid, radius], dim=-1)
            group_embeddings.append(group_embedding)

        # print(f"Type of group_embeddings: {type(group_embeddings)}")
        # if isinstance(group_embeddings, list):
        #     # print(f"Number of elements in group_embeddings: {len(group_embeddings)}")
        #     for i, emb in enumerate(group_embeddings):
        #         print(f"Shape of group_embedding[{i}]: {emb.shape if isinstance(emb, torch.Tensor) else 'Invalid Tensor'}")
        # else:
        #     print("group_embeddings is not a list!")
        
        # Stack group embeddings along a new group dimension
        group_embeddings = torch.stack(group_embeddings, dim=2)  # (seq_len, nenv, num_groups, embedding_dim)
        
        # If I want to increase the group embedding vectors
        
        # Group_embeddings
        seq_len, nenv, num_groups, embedding_dim = group_embeddings.shape  # Assume embedding_dim=18

        # Flatten the input to (seq_len * nenv * num_groups, embedding_dim)
        group_embeddings_flat = group_embeddings.view(-1, embedding_dim) 
        
        # group_embeddings_flat = torch.clamp(group_embeddings_flat, min=-1e5, max=1e5)
        # group_embeddings_flat = torch.nn.functional.normalize(group_embeddings_flat, p=2, dim=-1)

        # Pass through the linear layer
        embeddings = self.relu(self.group_embedding_layer(group_embeddings_flat))  # Shape: (seq_len * nenv * num_groups, hidden_dim)
        
        # Check for NaN in output
        if torch.isnan(embeddings).any():
            raise ValueError("NaN detected in embeddings")

        # Reshape back to (seq_len, nenv, num_groups, hidden_dim)
        group_embeddings = embeddings.view(seq_len, nenv, num_groups, -1)
        
        # input_emb=self.embedding_layer(inp).view(seq_len*nenv, max_human_num, -1)

        return group_embeddings


def reshapeT(T, seq_length, nenv):
    shape = T.size()[1:]
    return T.unsqueeze(0).reshape((seq_length, nenv, *shape))