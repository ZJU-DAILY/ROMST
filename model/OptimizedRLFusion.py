import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from .SpatioTemporatest import SpatioTemporalModel
from .TextFeatureExtractor import TextFeatureExtractor
from mydatasets.text_encoder import BjTTTextFeatureExtractor
import os


class ImageFeatureExtractor(nn.Module):
    def __init__(self, image_channels=1, feature_dim=128):
        super().__init__()
        
        self.img_encoder = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    
        self.feature_proj = nn.Linear(128, feature_dim)
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        for param in self.parameters():
            if param.requires_grad:
                param.register_hook(lambda grad: torch.clamp(grad, -1.0, 1.0))
    
    def forward(self, images):
        if len(images.shape) == 5:
            batch_size, dim1, channels, height, width = images.shape
            single_batch = images[0]
            single_features = self.img_encoder(single_batch)
            single_features = single_features.view(dim1, 128)
            projected = self.feature_proj(single_features)
            features = projected.unsqueeze(0).expand(batch_size, -1, -1)
            
        elif len(images.shape) == 6:
            batch_size, seq_len, num_nodes, channels, height, width = images.shape
            single_batch = images[0]
            single_flat = single_batch.view(-1, channels, height, width)
            single_features = self.img_encoder(single_flat)
            single_features = single_features.view(-1, 128)
            projected = self.feature_proj(single_features)
            projected = projected.view(seq_len, num_nodes, -1)
            features = projected.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        return features


Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class PrioritizedExperienceReplay:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
        
    def push(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
            
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        experiences = [self.buffer[idx] for idx in indices]
        states = np.stack([e.state for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        next_states = np.stack([e.next_state for e in experiences])
        dones = np.array([e.done for e in experiences])
        
        return states, actions, rewards, next_states, dones, indices, weights
        
    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            priority = (abs(error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
            
    def __len__(self):
        return len(self.buffer)


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
        
    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
        
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.data = torch.outer(epsilon_out, epsilon_in)
        self.bias_epsilon.data = epsilon_out
        
    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())
        
    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)


class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        self.feature_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            NoisyLinear(hidden_dim // 2, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            NoisyLinear(hidden_dim // 2, action_dim)
        )
        
    def forward(self, state):
        features = self.feature_layers(state)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values
    
    def reset_noise(self):
        for layer in self.value_stream:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
        for layer in self.advantage_stream:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()


class AdaptiveDynamicFusionController(nn.Module):
    def __init__(self, hidden_dim, action_dim=4, dropout_rate=0.2, use_text=True, use_image=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.use_text = use_text
        self.use_image = use_image
        
        self.action_encoder = nn.Embedding(self.action_dim, hidden_dim)
        
        self.st_norm = nn.LayerNorm(hidden_dim)
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.image_norm = nn.LayerNorm(hidden_dim)
        
        if use_text and use_image:
            input_dim = hidden_dim * 4

        elif use_text or use_image:
            input_dim = hidden_dim * 3
        else:
            input_dim = hidden_dim * 2
            
        max_weight_dim = 3 if (use_text and use_image) else (2 if (use_text or use_image) else 1)
        self.action_aware_weight_gen = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, max_weight_dim),
            nn.Softmax(dim=-1)
        )
        
        self.modal_interaction = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
        
        self.residual_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self._init_fusion_weights()
   
        
    def forward(self, st_feat, text_feat, image_feat, action):
        batch_size = st_feat.shape[0]
        seq_len = st_feat.shape[1]
        num_nodes = st_feat.shape[2]
        
        has_text = text_feat is not None and not (text_feat == 0).all()
        has_image = image_feat is not None and not (image_feat == 0).all()

        st_feat = self.st_norm(st_feat)
        if text_feat is not None:
            text_feat = self.text_norm(text_feat)
        if image_feat is not None:
            image_feat = self.image_norm(image_feat)
        
        action_tensor = torch.tensor(action, dtype=torch.long, device=st_feat.device)
        action_encoded = self.action_encoder(action_tensor)
        action_encoded = action_encoded.unsqueeze(0).unsqueeze(1).unsqueeze(2).expand(
            batch_size, seq_len, num_nodes, -1
        )
        
        st_reshaped = st_feat.reshape(batch_size * num_nodes, seq_len, self.hidden_dim)
        text_reshaped = text_feat.reshape(batch_size * num_nodes, seq_len, self.hidden_dim) if has_text else None
        image_reshaped = image_feat.reshape(batch_size * num_nodes, seq_len, self.hidden_dim) if has_image else None
        
        if has_text:
            st_text_interacted, _ = self.modal_interaction(st_reshaped, text_reshaped, text_reshaped)
        else:
            st_text_interacted = st_reshaped
            
        if has_image:
            st_image_interacted, _ = self.modal_interaction(st_reshaped, image_reshaped, image_reshaped)
        else:
            st_image_interacted = st_reshaped
        
        st_text_interacted = st_text_interacted.reshape(batch_size, seq_len, num_nodes, self.hidden_dim)
        st_image_interacted = st_image_interacted.reshape(batch_size, seq_len, num_nodes, self.hidden_dim)
        
        if has_text and has_image:
            combined_features = torch.cat([st_feat, st_text_interacted, st_image_interacted, action_encoded], dim=-1)
        elif has_text:
            combined_features = torch.cat([st_feat, st_text_interacted, action_encoded], dim=-1)
        elif has_image:
            combined_features = torch.cat([st_feat, st_image_interacted, action_encoded], dim=-1)
        else:
            combined_features = torch.cat([st_feat, action_encoded], dim=-1)
        
        modal_weights = self.action_aware_weight_gen(combined_features)
        
        weights_full = torch.zeros(batch_size, seq_len, num_nodes, 3, device=st_feat.device)
        if has_text and has_image:
            used_weights = modal_weights
            weights_full = used_weights
            fused_features = torch.stack([st_feat, st_text_interacted, st_image_interacted], dim=3)
            gates = used_weights.unsqueeze(-1)
            st_enhanced = (fused_features * gates).sum(dim=3)
        elif has_text:
            used_weights = modal_weights
            weights_full[:, :, :, 0:2] = used_weights
            fused_features = torch.stack([st_feat, st_text_interacted], dim=3)
            gates = used_weights.unsqueeze(-1)
            st_enhanced = (fused_features * gates).sum(dim=3)
        elif has_image:
            used_weights = modal_weights
            weights_full[:, :, :, 0] = used_weights[:, :, :, 0]
            weights_full[:, :, :, 2] = used_weights[:, :, :, 1]
            fused_features = torch.stack([st_feat, st_image_interacted], dim=3)
            gates = used_weights.unsqueeze(-1)
            st_enhanced = (fused_features * gates).sum(dim=3)
        else:
            used_weights = modal_weights
            weights_full[:, :, :, 0] = used_weights.squeeze(-1)
            st_enhanced = st_feat
        
        gate_input = torch.cat([st_feat, st_enhanced], dim=-1)
        residual_weight = self.residual_gate(gate_input)
        final_fusion = st_feat * (1 - residual_weight) + st_enhanced * residual_weight
        
        fusion_weights = weights_full.mean(dim=(1, 2))
        
        return final_fusion,fusion_weights.detach()


class AdvancedRLAgent:
    def __init__(self, state_dim, action_dim=4, lr=1e-4, gamma=0.99, 
                 buffer_size=8000, batch_size=64, target_update_freq=500):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_network = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_network = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = torch.optim.AdamW(
            self.q_network.parameters(), lr=lr, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50000, eta_min=lr*0.1
        )
        
        self.replay_buffer = PrioritizedExperienceReplay(buffer_size)
        
        self.steps_done = 0
        self.training_step = 0
        
    def select_action(self, state, training=True):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if training:
                self.q_network.reset_noise()
                
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def compute_enhanced_reward(self, st_prediction, fused_pred, targets, fusion_weights, epoch=None, warmup_epochs=10, total_epochs=50):
        if st_prediction is None:
            return 0.0
        try:
            fusion_mae = F.l1_loss(fused_pred, targets).item()
            st_mae = F.l1_loss(st_prediction, targets).item()
            if not hasattr(self, 'historical_mae'):
                self.historical_mae = []
            self.historical_mae.append(fusion_mae)
            if len(self.historical_mae) > 10:
                self.historical_mae.pop(0)
            avg_hist_mae = sum(self.historical_mae) / len(self.historical_mae)
            base_reward = st_mae - fusion_mae
            self_reward = max(0, avg_hist_mae - fusion_mae)
            if fusion_weights is not None:
                w = fusion_weights.detach().cpu().numpy()
                w = w / (w.sum() + 1e-8)
                entropy = -np.sum(w * np.log(w + 1e-8))
                entropy_reward = entropy
            else:
                entropy_reward = 0.0
            reward = 10 * (base_reward + 0.5 * self_reward + 0.1 * entropy_reward)
            
            return reward
        except Exception:
            return -1.0

           
       
    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
            
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return None
            
        states, actions, rewards, next_states, dones, indices, is_weights = batch
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        is_weights = torch.FloatTensor(is_weights).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * ~dones.unsqueeze(1))
        
        td_errors = current_q_values - target_q_values
        
        
        loss = (is_weights.unsqueeze(1) * F.mse_loss(current_q_values, target_q_values, reduction='none')).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy().flatten())
        
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()


class CompactStateBuilder:
    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim
    def build_state(self, st_feat, text_feat=None, image_feat=None):
        epsilon = 1e-6
        st_mean = st_feat.mean(dim=(0, 1))
        st_var = st_feat.var(dim=(0, 1))
        st_core = st_mean / (st_var + epsilon)
        
        if text_feat is not None:
            text_mean = text_feat.mean(dim=(0, 1))
            has_text = 1.0
        else:
            text_mean = torch.zeros_like(st_core)
            has_text = 0.0
        if image_feat is not None:
            image_mean = image_feat.mean(dim=(0, 1))
            has_image = 1.0
        else:
            image_mean = torch.zeros_like(st_core)
            has_image = 0.0
        state_tensor = torch.cat([
            st_core,
            text_mean,
            image_mean,
            torch.tensor([has_text, has_image], device=st_feat.device)
        ], dim=0).detach()
        
        return state_tensor.cpu().numpy() if state_tensor.device.type == 'cuda' else state_tensor.numpy()


class UltimateMultiModalRLModel(nn.Module):
    def __init__(self, num_nodes, temporal_input_dim, hidden_dim, st_output_dim, 
                 num_mamba_layers, text_model, image_channels=1, pred_len=12, 
                 use_image=False, use_text=True, maxlen=966, text_feature_extractor=None,
                 baseline_model_path=None):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.st_model = SpatioTemporalModel(
            temporal_input_dim=temporal_input_dim,
            output_dim=st_output_dim,
            seq_len=pred_len,
            pred_len=pred_len
        )
        
        if baseline_model_path is not None and os.path.exists(baseline_model_path):
            checkpoint = torch.load(baseline_model_path, map_location=self.device)
            self.st_model.load_state_dict(checkpoint['model_state_dict'])
            self.st_model.eval()
        
        self.use_text = use_text
        self.use_image = use_image
        
        if use_text:
            if text_feature_extractor is not None:
                self.text_feature_extractor = text_feature_extractor
                text_dim = text_feature_extractor.text_encoder.feature_dim
                self.text_model = None
                if hasattr(text_feature_extractor, 'text_encoder') and hasattr(text_feature_extractor.text_encoder, 'max_timesteps'):
                    self.text_processor = nn.LSTM(text_dim, hidden_dim, batch_first=True)
                else:
                    self.text_processor = None
                    self.text_projection = nn.Linear(text_dim, hidden_dim)
            elif text_model is not None:
                self.text_feature_extractor = None
                self.text_model = text_model
                text_dim = text_model.get_feature_dim() if hasattr(text_model, 'get_feature_dim') else 2048
                self.text_processor = None
                self.text_projection = nn.Linear(text_dim, hidden_dim)
            
            self.text_cache = {}
            
        if use_image:
            self.image_extractor = ImageFeatureExtractor(image_channels, hidden_dim)
        if use_text and use_image:
            action_dim = 4
        elif use_text or use_image:
            action_dim = 3
        else:
            action_dim = 1
        self.fusion_controller = AdaptiveDynamicFusionController(hidden_dim, action_dim=action_dim, use_text=use_text, use_image=use_image)
        self.state_builder = CompactStateBuilder(hidden_dim)
        state_dim = hidden_dim * 3 + 2
        self.rl_agent = AdvancedRLAgent(state_dim, action_dim=action_dim)
        self.action_dim = action_dim
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, st_output_dim)
        )
        self.pred_len = pred_len
        self.maxlen = maxlen
        self.to(self.device)
        self.rl_agent.q_network.to(self.device)
        self.rl_agent.target_network.to(self.device)
        self.global_step = 0
        
        self.weights_history = []
        
        self.to(self.device)
        
    def forward(self, temporal_data, text_data=None, images=None, adj_matrix=None, 
                targets=None, training=True, epoch=None, text=None):
        batch_size = temporal_data.shape[0]
        seq_len = temporal_data.shape[1]
        num_nodes = temporal_data.shape[2]
        device = temporal_data.device
        with torch.no_grad():
            if training and targets is not None:
                st_prediction, raw_st_feat = self.st_model(
                    temporal_data, adj_matrix, return_prediction=True
                )
            else:
                raw_st_feat = self.st_model(temporal_data, adj_matrix, return_prediction=False)
                st_prediction = None
        raw_st_feat_flat = raw_st_feat.contiguous().reshape(-1, 64)
        aligned_st_flat = raw_st_feat_flat.float()
        aligned_st = aligned_st_flat.reshape(batch_size, seq_len, num_nodes, -1)
        aligned_text = None
        if self.use_text:
            if text is not None:
                if isinstance(text, list) and len(text) > 0:
                    first = text[0]
                    text_key = tuple(first) if isinstance(first, list) else first
                else:
                    text_key = text
                if text_key not in self.text_cache:
                    batch_texts = text[0] if isinstance(text, list) and len(text) > 0 else text
                    
                    with torch.no_grad():
                        if isinstance(batch_texts, list):
                            text_features_list = []
                            for i, node_text in enumerate(batch_texts):
                                try:
                                    node_features = self.text_model.extract_features(node_text)
                                    text_features_list.append(node_features)
                                except Exception as e:
                                    node_features = torch.zeros(self.text_model.get_feature_dim(), device=self.device)
                                    text_features_list.append(node_features)
                            text_features = torch.stack(text_features_list, dim=0)
                        else:
                            text_features = self.text_model.extract_features(batch_texts).unsqueeze(0)
                    
                    text_features = self.text_projection(text_features)
                    
                    self.text_cache[text_key] = text_features
                    
                    torch.cuda.empty_cache()
                else:
                    text_features = self.text_cache[text_key]
                
                aligned_text = text_features.unsqueeze(0).unsqueeze(1).expand(batch_size, seq_len, -1, -1)
                
            elif text_data is not None:
                batch_indices = text_data
                features = self.text_feature_extractor.extract_batch_features(
                    batch_indices, seq_len
                )
                
                if self.text_processor is not None:
                    batch_lstm_out, _ = self.text_processor(features)
                    aligned_text = batch_lstm_out.unsqueeze(2).expand(-1, -1, num_nodes, -1)
                else:
                    aligned_text = features.unsqueeze(2).expand(-1, -1, num_nodes, -1)
                    
            elif text_data is None and self.text_feature_extractor is not None:
                if hasattr(self.text_feature_extractor, 'text_encoder') and not hasattr(self.text_feature_extractor.text_encoder, 'max_timesteps'):
                    all_features = self.text_feature_extractor.text_encoder.get_all_features()
                    all_features = all_features.to(self.device, dtype=aligned_st.dtype, non_blocking=True)

                    if all_features.shape[-1] != aligned_st.shape[-1]:
                        if not hasattr(self, 'text_projection'):
                            feature_dim = all_features.shape[-1]
                            self.text_projection = nn.Linear(feature_dim, aligned_st.shape[-1]).to(self.device)
                            self.add_module('text_projection', self.text_projection)
                        all_features = self.text_projection(all_features)
                    all_features = all_features.contiguous()

                    torch.cuda.empty_cache()

                    aligned_text = all_features.unsqueeze(0).unsqueeze(1).expand(batch_size, seq_len, -1, -1)
        aligned_image = None
        if self.use_image and images is not None:
            if len(images.shape) == 5:
                batch_size, dim1, channels, height, width = images.shape
                
                if dim1 == num_nodes:
                    image_features = self.image_extractor(images)
                    aligned_image = image_features.unsqueeze(1).expand(-1, seq_len, -1, -1)
                    
                elif dim1 == seq_len:
                    image_features = self.image_extractor(images)
                    aligned_image = image_features.unsqueeze(2).expand(-1, -1, num_nodes, -1)
                    
            elif len(images.shape) == 6:
                image_features = self.image_extractor(images)
                aligned_image = image_features
                
        fused_features_list = []
        fusion_weights_list = []
        states = []
        actions = []
        prev_batch_weights = getattr(self, 'prev_batch_weights', None)
        for i in range(batch_size):
            text_feat_i = aligned_text[i] if aligned_text is not None else None
            image_feat_i = aligned_image[i] if aligned_image is not None else None
            state_i = self.state_builder.build_state(
                aligned_st[i], text_feat_i, image_feat_i
            )
            states.append(state_i)
            
            action = self.rl_agent.select_action(state_i, training)
            actions.append(action)
            
            fused_feat_i, weights_i = self.fusion_controller(
                aligned_st[i:i+1], 
                aligned_text[i:i+1] if aligned_text is not None else None,
                aligned_image[i:i+1] if aligned_image is not None else None,
                action
            )
            
            fused_features_list.append(fused_feat_i.squeeze(0))
            fusion_weights_list.append(weights_i.squeeze(0))
            
        fused_features = torch.stack(fused_features_list, dim=0)
        fusion_weights = torch.stack(fusion_weights_list, dim=0)
        
        batch_size, seq_len, num_nodes, hidden_dim = fused_features.shape
        fused_features_flat = fused_features.reshape(-1, hidden_dim)
        predictions_flat = self.predictor(fused_features_flat)
        
        final_prediction = predictions_flat.reshape(batch_size, seq_len, num_nodes, -1)
        
        if self.pred_len < final_prediction.shape[1]:
            final_prediction = final_prediction[:, -self.pred_len:, ...]
            
        rl_loss = 0.0
        rewards = []
        
        if targets is not None and training:
            
            for i in range(batch_size):
                st_pred_i = st_prediction[i:i+1] if st_prediction is not None else None
                fusion_pred_i = final_prediction[i:i+1]
                target_i = targets[i:i+1]
                reward = self.rl_agent.compute_enhanced_reward(
                    st_pred_i, fusion_pred_i, target_i, fusion_weights[i], epoch=epoch
                )
                if not hasattr(self, 'reward_history'):
                    self.reward_history = []
                if not hasattr(self, 'global_step'):
                    self.global_step = 0
                self.global_step += 1
                self.reward_history.append((self.global_step, float(reward)))
                
                if not hasattr(self, 'weights_history'):
                    self.weights_history = []
                try:
                    weights_vec = fusion_weights[i].detach().cpu().numpy()
                    self.weights_history.append((self.global_step, weights_vec.tolist()))
                except Exception:
                    pass
                
                rewards.append(reward)
                next_state = states[i + 1] if i + 1 < len(states) else states[i]
                
                self.rl_agent.store_experience(states[i], actions[i], reward, next_state, False)
                
            rl_loss = self.rl_agent.learn() or 0.0
            self.prev_batch_weights = fusion_weights.detach()

            return final_prediction, st_prediction, states, actions, rewards, rl_loss
        if not training:
            self.prev_batch_weights = fusion_weights.detach()
        return final_prediction, st_prediction, states, actions, None, None
    
    def update_target_network(self):
        self.rl_agent.target_network.load_state_dict(self.rl_agent.q_network.state_dict())
        
    def get_training_stats(self):
        return {
            'buffer_size': len(self.rl_agent.replay_buffer),
            'training_step': self.rl_agent.training_step,
            'learning_rate': self.rl_agent.optimizer.param_groups[0]['lr']
        }
    
    def get_reward_history(self):
        if hasattr(self, 'reward_history'):
            return self.reward_history
        return []
    
    def clear_reward_history(self):
        if hasattr(self, 'reward_history'):
            self.reward_history = []
        if hasattr(self, 'global_step'):
            self.global_step = 0
    
    def get_weights_history(self):
        if hasattr(self, 'weights_history'):
            return self.weights_history
        return []
    
    def clear_weights_history(self):
        if hasattr(self, 'weights_history'):
            self.weights_history = []
    
    def clear_text_cache(self):
        if hasattr(self, 'text_cache'):
            self.text_cache.clear()
