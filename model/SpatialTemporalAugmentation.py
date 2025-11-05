import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SpatialTemporalAugmentation(nn.Module):
 
    def __init__(self, alpha):

        super().__init__()
        self.alpha = alpha
        self.eps = 1e-6
        

        self.p_close = 0.5  
        self.p_mid = 0.3    
        self.p_far = 0.2  

    def to_probability(self, data):

        data_positive = data - data.min() + self.eps
        prob = data_positive / (data_positive.sum(dim=-1, keepdim=True) + self.eps)
        return prob

    def kl_divergence(self, P, Q):
        P = P + self.eps
        Q = Q + self.eps
        kl = torch.sum(P * torch.log(P / Q), dim=-1)
        return kl

    def jensen_shannon_divergence(self, P, Q):
        M = 0.5 * (P + Q)
        jsd = 0.5 * self.kl_divergence(P, M) + 0.5 * self.kl_divergence(Q, M)
        jsd = jsd / np.log(2)

        return jsd
    
    def calculate_stjsd(self, predictions, targets):
        batch_size = predictions.shape[0]
        
        pred_flat = predictions.reshape(batch_size, -1)
        target_flat = targets.reshape(batch_size, -1)
        
        pred_prob = self.to_probability(pred_flat)
        target_prob = self.to_probability(target_flat)
    
        stjsd = self.jensen_shannon_divergence(target_prob, pred_prob)
        
        return stjsd

    def generate_spatial_correlated_noise(self, shape, device, adj_matrix=None):

        batch_size, seq_len, num_nodes, num_features = shape
        base_noise = torch.randn(shape, device=device) 
        if adj_matrix is not None:
            adj_matrix = adj_matrix.to(device=device, dtype=torch.float32)
            is_binary = torch.all((adj_matrix == 0) | (adj_matrix == 1))

            if is_binary:
    
                eye = torch.eye(adj_matrix.shape[0], device=adj_matrix.device)
                
                A1 = (adj_matrix > 0).float()
                A1 = A1 * (1.0 - eye)
                A1 = A1 / (A1.sum(dim=1, keepdim=True) + self.eps)


                A2_raw = (adj_matrix @ adj_matrix) > 0
                A2 = A2_raw.float() * (1 - (A1 > 0).float())
                A2 = A2 * (1.0 - eye)
                A2 = A2 / (A2.sum(dim=1, keepdim=True) + self.eps)

    
                A3_raw = (A2_raw.float() @ adj_matrix) > 0
                A3 = A3_raw.float() * (1 - (A1 > 0).float()) * (1 - (A2_raw > 0).float())
                A3 = A3 * (1.0 - eye)
                A3 = A3 / (A3.sum(dim=1, keepdim=True) + self.eps)

                W = self.p_close * A1 + self.p_mid * A2 + self.p_far * A3

                W = W / (W.sum(dim=1, keepdim=True) + self.eps)
            else:
                
                D = adj_matrix.clamp(min=0.0)
                S = 1.0 - D  
                S = S.clamp(min=0.0)
                eye = torch.eye(S.shape[0], device=S.device, dtype=torch.bool)
                off_diag = S[~eye]
                q1 = torch.quantile(off_diag, 0.33)
                q2 = torch.quantile(off_diag, 0.66)

                M1 = (S >= q2).float()      
                M2 = ((S >= q1) & (S < q2)).float()  
                M3 = (S < q1).float()      

                # 去除自环权重
                one_minus_eye = (~eye).float()
                W1 = (M1 * S) * one_minus_eye; W1 = W1 / (W1.sum(dim=1, keepdim=True) + self.eps)
                W2 = (M2 * S) * one_minus_eye; W2 = W2 / (W2.sum(dim=1, keepdim=True) + self.eps)
                W3 = (M3 * S) * one_minus_eye; W3 = W3 / (W3.sum(dim=1, keepdim=True) + self.eps)

                W = self.p_close * W1 + self.p_mid * W2 + self.p_far * W3
        
                W = W / (W.sum(dim=1, keepdim=True) + self.eps)

            spatial_noise = torch.einsum('ij,btjf->btif', W, base_noise)
            
        else:
            spatial_noise = torch.zeros_like(base_noise)
            
        
            for i in range(num_nodes):
                if i > 0:
                    noise_close_left = base_noise[:, :, i-1, :]
                else:
                    noise_close_left = base_noise[:, :, i, :]  
                    
                if i < num_nodes - 1:
                    noise_close_right = base_noise[:, :, i+1, :]
                else:
                    noise_close_right = base_noise[:, :, i, :]  
                
        
                noise_close = (noise_close_left + noise_close_right) / 2
                
        
                mid_idx = (i + num_nodes // 4) % num_nodes  
                noise_mid = base_noise[:, :, mid_idx, :]
                
                far_idx = (i + num_nodes // 2) % num_nodes  
                noise_far = base_noise[:, :, far_idx, :]
    
                spatial_noise[:, :, i, :] = (
                    self.p_close * noise_close + 
                    self.p_mid * noise_mid + 
                    self.p_far * noise_far
                )
        
        return spatial_noise

    def augment_with_adaptive_noise(self, temporal_data, predictions, targets,adj_matrix=None):
        device = temporal_data.device
        batch_size, seq_len, num_nodes, num_features = temporal_data.shape
        
        with torch.no_grad():
            stjsd = self.calculate_stjsd(predictions, targets)  
            noise_intensity = stjsd.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  
            noise_intensity = noise_intensity.expand(batch_size, seq_len, num_nodes, num_features)
        
    
        spatial_noise = self.generate_spatial_correlated_noise(
            temporal_data.shape, device,adj_matrix
        )
        
        adaptive_noise = noise_intensity * spatial_noise
        
        augmented_data = temporal_data + adaptive_noise

        return augmented_data, stjsd
    
    def compute_contrastive_loss(self, original_pred, augmented_pred, targets):

        batch_size = original_pred.shape[0]
        
        aug_flat = augmented_pred.reshape(batch_size, -1)
        targets_flat = targets.reshape(batch_size, -1)
        aug_prob = self.to_probability(aug_flat) + self.eps
        targets_prob = self.to_probability(targets_flat)
        consistency_loss = -torch.sum(targets_prob * torch.log(aug_prob), dim=-1).mean()


        dims = tuple(range(1, original_pred.dim()))  
        distance = torch.sqrt(torch.sum((original_pred - augmented_pred) ** 2, dim=dims) + self.eps)
        diversity_loss = -distance.mean()

        total_loss = consistency_loss + self.alpha * diversity_loss

        return total_loss, {
            'consistency_loss': float(consistency_loss.detach().cpu()),
            'diversity_loss': float(diversity_loss.detach().cpu()),
            'total_loss': float(total_loss.detach().cpu())
        }
    
    def forward(self, temporal_data, predictions, targets, model_forward_fn, adj_matrix=None):


        augmented_data, stjsd_values = self.augment_with_adaptive_noise(
            temporal_data, predictions, targets, adj_matrix
        )
        

        augmented_predictions = model_forward_fn(augmented_data)
        
    
        contrastive_loss, loss_dict = self.compute_contrastive_loss(
            predictions, augmented_predictions, targets
        )
        

        loss_dict['stjsd_mean'] = stjsd_values.mean().item()
        
        return contrastive_loss, loss_dict
