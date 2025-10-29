import torch
import torch.nn as nn
import numpy as np
import os
import json
import pickle
from typing import List, Dict, Optional
from model.TextFeatureExtractor import TextFeatureExtractor

 

class TerraTextEncoder:
    
    def __init__(self, text_model: TextFeatureExtractor, cache_dir: str = "./terra_text_cache", 
                 max_length: int = 966, num_nodes: int = 100, index: int = -1):
        self.text_model = text_model
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.num_nodes = num_nodes
        self.index = index
        self.device = text_model.device
        self.feature_dim = text_model.get_feature_dim()
        
        os.makedirs(cache_dir, exist_ok=True)
        index_tag = f"_index_{self.index}" if self.index != -1 else ""
        self.cache_file = os.path.join(cache_dir, f"terra_text_features{index_tag}.pkl")
        self.metadata_file = os.path.join(cache_dir, f"terra_text_metadata{index_tag}.json")
        self.text_features = None
        self.node_texts = None
        
    def check_cache_exists(self) -> bool:
        return os.path.exists(self.cache_file) and os.path.exists(self.metadata_file)
    
    def load_cached_features(self) -> torch.Tensor:
        if not self.check_cache_exists():
            return None
        
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        if metadata.get('num_nodes') != self.num_nodes:
            return None
        
        with open(self.cache_file, 'rb') as f:
            self.text_features = pickle.load(f)
        
        return self.text_features
    
    def precompute_features(self, text_dir: str, force_recompute: bool = False, index: int = None) -> torch.Tensor:
        if index is not None:
            self.index = index
            index_tag = f"_index_{self.index}" if self.index != -1 else ""
            self.cache_file = os.path.join(self.cache_dir, f"terra_text_features{index_tag}.pkl")
            self.metadata_file = os.path.join(self.cache_dir, f"terra_text_metadata{index_tag}.json")
        if not force_recompute and self.check_cache_exists():
            return self.load_cached_features()
        
        self._generate_geo_index()
        text_files = self._parse_text_files(text_dir)
        
        text_features_list = []
        node_texts = []
        
        for node_idx in range(self.num_nodes):
            lat, lon = self.index_to_geo[node_idx]
            
            text_file = None
            for file_path in text_files:
                filename = os.path.basename(file_path)
                import re
                pattern = r"meta_(\d+)([NS])_(\d+)([EW])\.txt"
                match = re.match(pattern, filename)
                if match:
                    file_lat, lat_dir, file_lon, lon_dir = match.groups()
                    file_lat = int(file_lat) * (-1 if lat_dir == "S" else 1)
                    file_lon = int(file_lon) * (-1 if lon_dir == "W" else 1)
                    
                    if file_lat == lat and file_lon == lon:
                        text_file = file_path
                        break
            
            if text_file is not None and os.path.exists(text_file):
                try:
                    with open(text_file, 'r', encoding='utf-8') as f:
                        text_content = f.read().strip()
                    node_texts.append(text_content)
                    
                    with torch.no_grad():
                        node_features = self.text_model.extract_features(text_content)
                    
                except Exception as e:
                    continue
            else:
                continue
            
            text_features_list.append(node_features)
            
            if node_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        if not text_features_list:
            self.text_features = torch.empty(0, self.feature_dim, device=self.device)
            self.node_texts = []
            self._save_cache()
            return self.text_features
        
        self.text_features = torch.stack(text_features_list, dim=0)
        self.node_texts = node_texts
        self._save_cache()
        return self.text_features
    
    def _parse_text_files(self, text_dir: str):
        text_files = []
        import re
        
        if not os.path.exists(text_dir):
            return text_files
        
        lat_range = (50, 60)
        lon_range = (-8, 2)
        
        pattern = r"meta_(\d+)([NS])_(\d+)([EW])\.txt"
        regex = re.compile(pattern)
        
        matched_files = []
        for file in os.listdir(text_dir):
            match = regex.match(file)
            if match:
                lat, lat_dir, lon, lon_dir = match.groups()
                lat = int(lat) * (-1 if lat_dir == "S" else 1)
                lon = int(lon) * (-1 if lon_dir == "W" else 1)
                
                if lat_range[0] <= lat < lat_range[1] and lon_range[0] <= lon < lon_range[1]:
                    matched_files.append((lat, lon, os.path.join(text_dir, file)))
        
        matched_files.sort(key=lambda x: (x[0], x[1]))
        
        for lat, lon, file_path in matched_files:
            text_files.append(file_path)
        
        return text_files
    
    def _generate_geo_index(self):
        self.geo_to_index = {}
        self.index_to_geo = {}
        
        lat_range = (50, 60)
        lon_range = (-8, 2)
        
        latitudes = np.linspace(lat_range[0], lat_range[1] - 1, lat_range[1] - lat_range[0])
        longitudes = np.linspace(lon_range[0], lon_range[1] - 1, lon_range[1] - lon_range[0])
        count = 0
        for lat in latitudes:
            for lon in longitudes:
                self.geo_to_index[(int(lat), int(lon))] = count
                self.index_to_geo[count] = (int(lat), int(lon))
                count += 1
        
        self.num_nodes = len(self.geo_to_index)
    
    def _save_cache(self):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.text_features, f)
        
        metadata = {
            'num_nodes': self.num_nodes,
            'feature_dim': self.text_features.shape[1],
            'max_length': self.max_length,
            'node_texts': self.node_texts,
            'index': self.index
        }
        
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
    
    def get_node_features(self, node_indices: List[int]) -> torch.Tensor:
        if self.text_features is None:
            return torch.empty(0, self.feature_dim, device=self.device)
        return self.text_features[node_indices]
    
    def get_all_features(self) -> torch.Tensor:
        if self.text_features is None:
            return torch.empty(0, self.feature_dim, device=self.device)
        return self.text_features


class TerraTextFeatureExtractor:
    
    def __init__(self, text_encoder: TerraTextEncoder, hidden_dim: int = 32):
        self.text_encoder = text_encoder
        self.hidden_dim = hidden_dim
        
        feature_dim = text_encoder.text_model.get_feature_dim()
        self.feature_projection = nn.Linear(feature_dim, hidden_dim)
        
        self.device = text_encoder.device
        self.feature_projection.to(self.device)
    
    def extract_batch_features(self, batch_indices: List[int], seq_len: int) -> torch.Tensor:
        node_features = self.text_encoder.get_node_features(batch_indices)
        projected_features = self.feature_projection(node_features)
        batch_features = projected_features.unsqueeze(1).expand(-1, seq_len, -1)
        return batch_features
    
    def get_feature_dim(self) -> int:
        return self.hidden_dim 