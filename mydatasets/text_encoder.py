

import os
import torch
import numpy as np
from typing import List, Optional, Dict
import pickle
from pathlib import Path


class BjTTTextEncoder:
    
    def __init__(self, text_model, cache_dir: str = "./text_cache", max_length: int = 966, max_timesteps: int = 3200, index: int = -1):
        self.text_model = text_model
        self.cache_dir = Path(cache_dir)
        self.max_length = max_length
        self.max_timesteps = max_timesteps
        self.index = index
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.cache_dir.mkdir(exist_ok=True)
        index_tag = f"_index_{self.index}" if self.index != -1 else ""
        self.cache_file = self.cache_dir / f"bjtt_text_features{index_tag}.pkl"
        self.metadata_file = self.cache_dir / f"bjtt_text_metadata{index_tag}.pkl"
        
        self.text_features: Optional[torch.Tensor] = None
        self.feature_dim: Optional[int] = None
        
    def precompute_features(self, data_dir: str, force_recompute: bool = False, index: Optional[int] = None) -> torch.Tensor:
        if index is not None:
            self.index = index
            index_tag = f"_index_{self.index}" if self.index != -1 else ""
            self.cache_file = self.cache_dir / f"bjtt_text_features{index_tag}.pkl"
            self.metadata_file = self.cache_dir / f"bjtt_text_metadata{index_tag}.pkl"
        if not force_recompute and self.cache_file.exists():
            with open(self.cache_file, 'rb') as f:
                self.text_features = pickle.load(f)
            with open(self.metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            self.feature_dim = metadata['feature_dim']
            if self.text_features is not None:
                return self.text_features
        
        
        
        all_texts = []
        timestep_to_file = {}
        current_timestep = 0
        
        for month in ['1', '2', '3']:
            data_folder = os.path.join(data_dir, 'data', month)
            text_folder = os.path.join(data_dir, 'text', month)
            
            if not os.path.exists(data_folder):
                continue
            if not os.path.exists(text_folder):
                continue
                
            npy_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.npy')])
            
            for npy_file in npy_files:
                txt_file = os.path.splitext(npy_file)[0] + '.txt'
                txt_path = os.path.join(text_folder, txt_file)
                
                if os.path.exists(txt_path):
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    all_texts.append(text)
                    timestep_to_file[current_timestep] = (month, npy_file, txt_file)
                else:
                    all_texts.append("")
                    timestep_to_file[current_timestep] = (month, npy_file, "missing.txt")
                
                current_timestep += 1
        
        if len(all_texts) > self.max_timesteps:
            all_texts = all_texts[:self.max_timesteps]
            timestep_to_file = {k: v for k, v in timestep_to_file.items() if k < self.max_timesteps}
        
        if len(all_texts) == 0:
            self.text_features = torch.empty(0, 0)
            self.feature_dim = 0
            return self.text_features
        
        total_timesteps = len(all_texts)
        if self.index != -1:
            if self.index == 1:
                cut = int(total_timesteps * 0.3)
                all_texts = all_texts[:cut]
                timestep_to_file = {i: timestep_to_file[i] for i in range(cut) if i in timestep_to_file}
            else:
                start_idx = int(total_timesteps * (0.3 + (self.index - 2) * 0.175))
                end_idx = int(total_timesteps * (0.3 + (self.index - 1) * 0.175))
                all_texts = all_texts[start_idx:end_idx]
                timestep_to_file = {i - start_idx: timestep_to_file[i] for i in range(start_idx, min(end_idx, total_timesteps)) if i in timestep_to_file}
            

        all_features = []
        
        for i, text in enumerate(all_texts):
            feature = self.text_model.extract_features(
                text,
                max_length=self.max_length
            ).float()
            all_features.append(feature.cpu())
            
            
        self.text_features = torch.stack(all_features, dim=0)
        self.feature_dim = self.text_features.shape[1]
        
        
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.text_features, f)
        
        metadata = {
            'feature_dim': self.feature_dim,
            'total_timesteps': len(all_texts),
            'max_timesteps': self.max_timesteps,
            'timestep_to_file': timestep_to_file,
            'index': self.index
        }
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        return self.text_features
    
    def get_sample_features(self, start_idx: int, seq_len: int) -> torch.Tensor:
        if self.text_features is None:
            return torch.empty(0, 0)
        
        end_idx = start_idx + seq_len
        if end_idx > self.text_features.shape[0]:
            return torch.empty(0, self.text_features.shape[1] if self.text_features is not None else 0)
        
        return self.text_features[start_idx:end_idx]
    
    def get_batch_features(self, start_indices: List[int], seq_len: int) -> torch.Tensor:
        batch_features = []
        for start_idx in start_indices:
            sample_features = self.get_sample_features(start_idx, seq_len)
            batch_features.append(sample_features)
        
        return torch.stack(batch_features, dim=0)
    
    def check_cache_exists(self) -> bool:
        if not self.cache_file.exists() or not self.metadata_file.exists():
            return False
        
        try:
            with open(self.cache_file, 'rb') as f:
                features = pickle.load(f)
            with open(self.metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            
            if metadata.get('max_timesteps', 0) != self.max_timesteps:
                return False
            
            return features is not None and features.shape[0] > 0
        except Exception as e:
            return False
    
    def load_cached_features(self) -> torch.Tensor:
        if not self.check_cache_exists():
            return None
        
        with open(self.cache_file, 'rb') as f:
            self.text_features = pickle.load(f)
        with open(self.metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        self.feature_dim = metadata['feature_dim']
        
        if self.text_features is None:
            return None
        
        return self.text_features
    
    def clear_cache(self):
        if self.cache_file.exists():
            self.cache_file.unlink()
        if self.metadata_file.exists():
            self.metadata_file.unlink()
        self.text_features = None
        self.feature_dim = None
        


class BjTTTextFeatureExtractor:
    
    def __init__(self, text_encoder: BjTTTextEncoder):
        self.text_encoder = text_encoder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def extract_batch_features(self, batch_indices: List[int], seq_len: int) -> torch.Tensor:
        features = self.text_encoder.get_batch_features(batch_indices, seq_len)
        return features.to(self.device) 