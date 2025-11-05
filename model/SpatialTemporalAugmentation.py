import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SpatialTemporalAugmentation(nn.Module):
 
    def __init__(self, alpha):
        """
        Args:
            alpha: 多样性损失权重
        """
        super().__init__()
        self.alpha = alpha
        self.eps = 1e-6
        
        # 空间距离概率
        self.p_close = 0.5  # 近距离权重
        self.p_mid = 0.3    # 中距离权重
        self.p_far = 0.2    # 远距离权重

    def to_probability(self, data):
        # 确保数据非负
        data_positive = data - data.min() + self.eps
        # 归一化到概率分布
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
        
        # 将预测和目标展平为 (batch, -1)
        pred_flat = predictions.reshape(batch_size, -1)
        target_flat = targets.reshape(batch_size, -1)
        
        # 转换为概率分布
        pred_prob = self.to_probability(pred_flat)
        target_prob = self.to_probability(target_flat)
        
        # 计算JSD
        stjsd = self.jensen_shannon_divergence(target_prob, pred_prob)
        
        return stjsd

    def generate_spatial_correlated_noise(self, shape, device, adj_matrix=None):

        batch_size, seq_len, num_nodes, num_features = shape
        
        # 首先生成所有节点的基础噪声
        base_noise = torch.randn(shape, device=device)  # (batch, seq_len, nodes, features)
        
        # 如果提供了邻接矩阵，使用真实拓扑
        if adj_matrix is not None:
            adj_matrix = adj_matrix.to(device=device, dtype=torch.float32)

            # 判断是二值邻接还是距离矩阵（含自环）
            is_binary = torch.all((adj_matrix == 0) | (adj_matrix == 1))

            if is_binary:
                # 不将自环视为近邻，先移除对角线
                eye = torch.eye(adj_matrix.shape[0], device=adj_matrix.device)
                # 1-hop 权重：行归一化的邻接（去对角）
                A1 = (adj_matrix > 0).float()
                A1 = A1 * (1.0 - eye)
                A1 = A1 / (A1.sum(dim=1, keepdim=True) + self.eps)

                # 2-hop（去除 1-hop 与自环）
                A2_raw = (adj_matrix @ adj_matrix) > 0
                A2 = A2_raw.float() * (1 - (A1 > 0).float())
                A2 = A2 * (1.0 - eye)
                A2 = A2 / (A2.sum(dim=1, keepdim=True) + self.eps)

                # 3-hop 及以上（去除前两圈与自环）
                A3_raw = (A2_raw.float() @ adj_matrix) > 0
                A3 = A3_raw.float() * (1 - (A1 > 0).float()) * (1 - (A2_raw > 0).float())
                A3 = A3 * (1.0 - eye)
                A3 = A3 / (A3.sum(dim=1, keepdim=True) + self.eps)

                W = self.p_close * A1 + self.p_mid * A2 + self.p_far * A3
                # 归一化三圈合成后的权重，确保不含自环且各圈占比正确
                W = W / (W.sum(dim=1, keepdim=True) + self.eps)
            else:
                # 距离矩阵（已归一化，越小越近，含自环）→ 相似度
                D = adj_matrix.clamp(min=0.0)
                S = 1.0 - D  # 转为相似度，越大越近
                S = S.clamp(min=0.0)
                eye = torch.eye(S.shape[0], device=S.device, dtype=torch.bool)

                # 依据分位数划分近/中/远（明确排除对角线）
                off_diag = S[~eye]
                q1 = torch.quantile(off_diag, 0.33)
                q2 = torch.quantile(off_diag, 0.66)

                M1 = (S >= q2).float()      # 近
                M2 = ((S >= q1) & (S < q2)).float()  # 中
                M3 = (S < q1).float()       # 远

                # 去除自环权重
                one_minus_eye = (~eye).float()
                W1 = (M1 * S) * one_minus_eye; W1 = W1 / (W1.sum(dim=1, keepdim=True) + self.eps)
                W2 = (M2 * S) * one_minus_eye; W2 = W2 / (W2.sum(dim=1, keepdim=True) + self.eps)
                W3 = (M3 * S) * one_minus_eye; W3 = W3 / (W3.sum(dim=1, keepdim=True) + self.eps)

                W = self.p_close * W1 + self.p_mid * W2 + self.p_far * W3
                # 合成后再次行归一化
                W = W / (W.sum(dim=1, keepdim=True) + self.eps)

            spatial_noise = torch.einsum('ij,btjf->btif', W, base_noise)
            
        else:
            spatial_noise = torch.zeros_like(base_noise)
            
            # 对每个节点，混合不同空间距离的噪声
            for i in range(num_nodes):
                # 近邻节点噪声（假设相邻节点）
                if i > 0:
                    noise_close_left = base_noise[:, :, i-1, :]
                else:
                    noise_close_left = base_noise[:, :, i, :]  # 边界情况
                    
                if i < num_nodes - 1:
                    noise_close_right = base_noise[:, :, i+1, :]
                else:
                    noise_close_right = base_noise[:, :, i, :]  # 边界情况
                
                # 近邻噪声：左右邻居的平均
                noise_close = (noise_close_left + noise_close_right) / 2
                
                # 中距离节点噪声（假设距离2-3的节点）
                mid_idx = (i + num_nodes // 4) % num_nodes  # 中等距离的节点
                noise_mid = base_noise[:, :, mid_idx, :]
                
                # 远距离节点噪声（假设距离较远的节点）
                far_idx = (i + num_nodes // 2) % num_nodes  # 远距离的节点
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
        
        # 计算预测质量（STJSD）
        with torch.no_grad():
            stjsd = self.calculate_stjsd(predictions, targets)  # (batch,)
            noise_intensity = stjsd.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (batch,1,1,1)
            noise_intensity = noise_intensity.expand(batch_size, seq_len, num_nodes, num_features)
        
        
        # 生成空间相关噪声
        spatial_noise = self.generate_spatial_correlated_noise(
            temporal_data.shape, device,adj_matrix
        )
        

        # 自适应噪声 = α(STJSD) * spatial_noise
        adaptive_noise = noise_intensity * spatial_noise
        
        # 数据增强
        augmented_data = temporal_data + adaptive_noise

        return augmented_data, stjsd
    
    def compute_contrastive_loss(self, original_pred, augmented_pred, targets):

        batch_size = original_pred.shape[0]
        
        # 一致性项：-Σ y ln p'(x)（目标分布与增强预测分布的交叉熵）
        aug_flat = augmented_pred.reshape(batch_size, -1)
        targets_flat = targets.reshape(batch_size, -1)
        aug_prob = self.to_probability(aug_flat) + self.eps
        targets_prob = self.to_probability(targets_flat)
        consistency_loss = -torch.sum(targets_prob * torch.log(aug_prob), dim=-1).mean()

        # 多样性项：鼓励原预测与增强预测相互远离
        dims = tuple(range(1, original_pred.dim()))  # 聚合除 batch 外的维度
        distance = torch.sqrt(torch.sum((original_pred - augmented_pred) ** 2, dim=dims) + self.eps)
        diversity_loss = -distance.mean()

        total_loss = consistency_loss + self.alpha * diversity_loss

        return total_loss, {
            'consistency_loss': float(consistency_loss.detach().cpu()),
            'diversity_loss': float(diversity_loss.detach().cpu()),
            'total_loss': float(total_loss.detach().cpu())
        }
    
    def forward(self, temporal_data, predictions, targets, model_forward_fn, adj_matrix=None):

        # 基于预测质量生成自适应增强数据
        augmented_data, stjsd_values = self.augment_with_adaptive_noise(
            temporal_data, predictions, targets, adj_matrix
        )
        
        # 获取增强数据的预测
        augmented_predictions = model_forward_fn(augmented_data)
        
        # 计算对比学习损失
        contrastive_loss, loss_dict = self.compute_contrastive_loss(
            predictions, augmented_predictions, targets
        )
        
        # 添加STJSD信息用于监控
        loss_dict['stjsd_mean'] = stjsd_values.mean().item()
        
        return contrastive_loss, loss_dict