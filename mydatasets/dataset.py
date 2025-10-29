import os
import sys
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
import json
import warnings
import cv2
warnings.filterwarnings('ignore')
import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from PIL import Image
    import torchvision.transforms as transforms
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

class BjTTDataset(Dataset):
    def __init__(self, data_dir, seq_len, pred_len, num_nodes, split='train', train_ratio=0.8, val_ratio=0.1, index=-1):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self._cache = {}
        self._cache_size = 100

        all_grid_data = []
        all_text_data = []
        
        def natural_key(s):
            return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]
        
        for month in ['1', '2', '3']:
            data_folder = os.path.join(data_dir, 'data', month)
            text_folder = os.path.join(data_dir, 'text', month)
            
            if not os.path.exists(data_folder):
                continue
                
            npy_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.npy')], key=natural_key)
            
            month_grid_data = []
            for file in npy_files:
                grid = np.load(os.path.join(data_folder, file)).astype(np.float32)
                month_grid_data.append(grid)
            
            all_grid_data.extend(month_grid_data)

            month_text_data = []
            if os.path.exists(text_folder):
                txt_files = sorted([f for f in os.listdir(text_folder) if f.endswith('.txt')], key=natural_key)
                if len(txt_files) != len(npy_files):
                    raise ValueError(
                        f"month={month} 文本与数据数量不一致: txt={len(txt_files)}, npy={len(npy_files)}"
                    )
                for file in txt_files:
                    with open(os.path.join(text_folder, file), 'r', encoding='utf-8') as fh:
                        month_text_data.append(fh.read())
            else:
                raise FileNotFoundError(f"Text folder not found: {text_folder}")

            if len(month_text_data) != len(month_grid_data):
                raise ValueError(
                    f"month={month} 文本与时序数据数量不一致: text={len(month_text_data)}, data={len(month_grid_data)}"
                )

            all_text_data.extend(month_text_data)
        
        self.all_grid_data = np.array(all_grid_data, dtype=np.float32)
        num_timesteps = len(self.all_grid_data)

        max_timesteps = 3200
        if num_timesteps > max_timesteps:
            self.all_grid_data = self.all_grid_data[:max_timesteps]
            num_timesteps = len(self.all_grid_data)
            all_text_data = all_text_data[:max_timesteps]
        else:
            if len(all_text_data) > num_timesteps:
                all_text_data = all_text_data[:num_timesteps]
            elif len(all_text_data) < num_timesteps:
                all_text_data.extend([""] * (num_timesteps - len(all_text_data)))

        current_data = self.all_grid_data[:, :, :, 1:2]
        current_data = current_data.reshape(current_data.shape[0], -1, 1)[:, :self.num_nodes, :]

        if index != -1:
            if index == 1:
                cut = int(num_timesteps * 0.3)
                current_data = current_data[:cut]
                all_text_data = all_text_data[:cut]
                num_timesteps = len(current_data)
            else:
                start_idx = int(num_timesteps * (0.3 + (index - 2) * 0.175))
                end_idx = int(num_timesteps * (0.3 + (index - 1) * 0.175))
                current_data = current_data[start_idx:end_idx]
                all_text_data = all_text_data[start_idx:end_idx]
                num_timesteps = len(current_data)

        train_end = int(num_timesteps * train_ratio)
        val_end = int(num_timesteps * (train_ratio + val_ratio))
        
        if split == 'train':
            self.start_idx = 0
            self.end_idx = train_end
        elif split == 'val':
            self.start_idx = train_end
            self.end_idx = val_end
        else:
            self.start_idx = val_end
            self.end_idx = num_timesteps
            
        train_slice = current_data[:train_end]
        reshaped_train = train_slice.reshape(-1, 1)
        self.data_min = np.min(reshaped_train).astype(np.float32)
        self.data_max = np.max(reshaped_train).astype(np.float32)
        self.normalized_data = ((current_data - self.data_min) / (self.data_max - self.data_min)).astype(np.float32)

        self.num_timesteps = self.end_idx - self.start_idx
        self.num_samples = self.num_timesteps - seq_len - pred_len + 1
        
        if self.num_samples <= 0:
            raise ValueError(f"Time series too short: total timesteps ({self.num_timesteps}) < sequence length ({seq_len}) + prediction length ({pred_len})")
            

        self.normalized_data = self.normalized_data[self.start_idx:self.end_idx]
        self.text_data = all_text_data[self.start_idx:self.end_idx]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx in self._cache:
            cached_result = self._cache[idx].copy()
            cached_result['sample_idx'] = idx
            return cached_result
        
        his = self.normalized_data[idx:idx+self.seq_len]
        temporal_data = his
        
        fut = self.normalized_data[idx+self.seq_len:idx+self.seq_len+self.pred_len]
        target = fut

        text_seq = self.text_data[idx:idx+self.seq_len]
        
        result = {
            'temporal_data': torch.tensor(temporal_data, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32),
            'text': text_seq,
            'sample_idx': idx
        }
        
        if len(self._cache) < self._cache_size:
            self._cache[idx] = result.copy()
            
        return result


class TerraDataset(Dataset):
    def __init__(self, image_dir, text_dir, time_series_path, seq_len, pred_len, 
                 image_size=(128, 128), channels=1, split='train', train_ratio=0.8, 
                 val_ratio=0.1, 
                 lat_range=(50, 60), lon_range=(-8, 2), transform=None, index=-1):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.image_size = image_size
        self.channels = channels
        self.split = split
        self._cache = {}
        self._cache_size = 100
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.index = index
        
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.time_series = np.load(time_series_path).astype(np.float32)
        
        if self.time_series.ndim == 2:
            total_timesteps, original_num_nodes = self.time_series.shape
            self.num_features = 1
        else:
            total_timesteps, original_num_nodes, self.num_features = self.time_series.shape
            
        
        self.image_dir = image_dir
        self.text_dir = text_dir
        
        self._load_modality_files()
        
        total_timesteps = self.time_series.shape[0]
        
        if self.index != -1:
            if self.index == 1:
                self.time_series = self.time_series[:int(total_timesteps * 0.3)]
                
                total_timesteps = len(self.time_series)
            else:
                start_idx = int(total_timesteps * (0.3 + (self.index - 2) * 0.175))
                end_idx = int(total_timesteps * (0.3 + (self.index - 1) * 0.175))
                self.time_series = self.time_series[start_idx:end_idx]
               
                total_timesteps = len(self.time_series)

        train_end = int(total_timesteps * train_ratio)
        val_end = int(total_timesteps * (train_ratio + val_ratio))
        
        if split == 'train':
            self.start_idx = 0
            self.end_idx = train_end
        elif split == 'val':
            self.start_idx = train_end
            self.end_idx = val_end
        else:
            self.start_idx = val_end
            self.end_idx = total_timesteps
        
        train_slice = self.time_series[:train_end]
        reshaped_train = train_slice.reshape(-1, 1)
        self.data_min = np.min(reshaped_train).astype(np.float32)
        self.data_max = np.max(reshaped_train).astype(np.float32)
        normalized_all = (self.time_series - self.data_min) / (self.data_max - self.data_min)
        self.normalized_data = normalized_all.astype(np.float32)

        self.num_timesteps = self.end_idx - self.start_idx
        
        self.normalized_data = self.normalized_data[self.start_idx:self.end_idx]

        self.num_samples = self.num_timesteps - seq_len - pred_len + 1
        
        if self.num_samples <= 0:
            raise ValueError(f"Time series too short: total timesteps ({self.num_timesteps}) < sequence length ({seq_len}) + prediction length ({pred_len})")
            
        

    def _load_modality_files(self):
        
        self.image_files = self._parse_files(self.image_dir, r"relief_(\d+)([NS])_(\d+)([EW])\.(png|jpg)")
        
        self.text_files = self._parse_files(self.text_dir, r"meta_(\d+)([NS])_(\d+)([EW])\.txt")
        
        
        self._generate_geo_index()
        
        self.matched_samples = self._match_samples()

    def _parse_files(self, directory, pattern):
        files = {}
        import re
        regex = re.compile(pattern)
        if os.path.exists(directory):
            for file in os.listdir(directory):
                match = regex.match(file)
                if match:
                    if pattern.endswith('\.(png|jpg)'):
                        lat, lat_dir, lon, lon_dir, ext = match.groups()
                    else:
                        lat, lat_dir, lon, lon_dir = match.groups()
                    lat = int(lat) * (-1 if lat_dir == "S" else 1)
                    lon = int(lon) * (-1 if lon_dir == "W" else 1)
                    
                    if self.lat_range[0] <= lat < self.lat_range[1] and self.lon_range[0] <= lon < self.lon_range[1]:
                        files[(lat, lon)] = os.path.join(directory, file)
        return files
        
    def _match_samples(self):
        matched_samples = []
        for coord in self.image_files.keys():
            if coord in self.text_files and coord in self.geo_to_index:
                matched_samples.append(coord)
        return matched_samples
    
    def _generate_geo_index(self):
        self.geo_to_index = {}
        self.index_to_geo = {}

        latitudes = np.linspace(self.lat_range[0], self.lat_range[1] - 1, self.lat_range[1] - self.lat_range[0])
        longitudes = np.linspace(self.lon_range[0], self.lon_range[1] - 1, self.lon_range[1] - self.lon_range[0])
        count = 0
        for lat in latitudes:
            for lon in longitudes:
                self.geo_to_index[(int(lat), int(lon))] = count
                self.index_to_geo[count] = (int(lat), int(lon))
                count += 1
        
        self.num_nodes = len(self.geo_to_index)
        

    def __len__(self):
        return len(self.matched_samples)

    def __getitem__(self, idx):
        lat, lon = self.matched_samples[idx]
        image_path = self.image_files[(lat, lon)]
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))
        img_tensor = self.transform(img)
        text_path = self.text_files[(lat, lon)]
        with open(text_path, 'r', encoding='utf-8') as f:
            text_data = f.read()
        text_tensor = text_data
        
        node_idx = self.geo_to_index[(lat, lon)]
        node_series = self.normalized_data[:, node_idx]
        
        x_samples = []
        y_samples = []
        for start_idx in range(self.num_timesteps - self.seq_len - self.pred_len + 1):
            x_samples.append(node_series[start_idx:start_idx + self.seq_len])
            y_samples.append(node_series[start_idx + self.seq_len:start_idx + self.seq_len + self.pred_len])
        
        x_samples = np.array(x_samples)
        y_samples = np.array(y_samples)
        
        return {
            'temporal_data': torch.tensor(x_samples, dtype=torch.float32),
            'images': img_tensor,
            'text': text_tensor,
            'target': torch.tensor(y_samples, dtype=torch.float32),
            'sample_idx': idx
        }


class PEMS04Dataset(Dataset):
    def __init__(self, data_dir, seq_len, pred_len, num_nodes=307, 
                 split='train', train_ratio=0.8, val_ratio=0.1, index=-1):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.index = index
        self._cache = {}
        self._cache_size = 100

        data_path = os.path.join(data_dir, 'data.npy')
        if os.path.exists(data_path):
            raw_data = np.load(data_path).astype(np.float32)
            
            self.time_series = raw_data[:, :, 0:1]
        else:
            raise FileNotFoundError(f"Data file not found: {data_path}")

        total_timesteps = self.time_series.shape[0]
        
        if self.index != -1:
            if self.index == 1:
                self.time_series = self.time_series[:int(total_timesteps * 0.3)]
                total_timesteps = len(self.time_series)
            else:
                start_idx = int(total_timesteps * (0.3 + (self.index - 2) * 0.175))
                end_idx = int(total_timesteps * (0.3 + (self.index - 1) * 0.175))
                self.time_series = self.time_series[start_idx:end_idx]
                total_timesteps = len(self.time_series)

        train_end = int(total_timesteps * train_ratio)
        val_end = int(total_timesteps * (train_ratio + val_ratio))
        
        if split == 'train':
            self.start_idx = 0
            self.end_idx = train_end
        elif split == 'val':
            self.start_idx = train_end
            self.end_idx = val_end
        else:
            self.start_idx = val_end
            self.end_idx = total_timesteps
        
        train_slice = self.time_series[:train_end]
        reshaped_train = train_slice.reshape(-1, 1)
        self.data_min = np.min(reshaped_train).astype(np.float32)
        self.data_max = np.max(reshaped_train).astype(np.float32)
        normalized_all = (self.time_series - self.data_min) / (self.data_max - self.data_min)
        self.normalized_data = normalized_all.astype(np.float32)

        self.num_timesteps = self.end_idx - self.start_idx
        self.num_samples = self.num_timesteps - seq_len - pred_len + 1
        
        if self.num_samples <= 0:
            raise ValueError(f"Time series too short: total timesteps ({self.num_timesteps}) < sequence length ({seq_len}) + prediction length ({pred_len})")
        
        

        self.normalized_data = self.normalized_data[self.start_idx:self.end_idx]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx in self._cache:
            cached_result = self._cache[idx].copy()
            cached_result['sample_idx'] = idx
            return cached_result
        
        temporal_data = self.normalized_data[idx:idx+self.seq_len]
        
        target = self.normalized_data[idx+self.seq_len:idx+self.seq_len+self.pred_len]
        
        result = {
            'temporal_data': torch.tensor(temporal_data, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32),
            'sample_idx': idx
        }
        
        if len(self._cache) < self._cache_size:
            self._cache[idx] = result.copy()
            
        return result


class GreenEarthNetDataset(Dataset):
    def __init__(self, data_dir, seq_len, pred_len, num_nodes, temporal_path=None, image_path=None, 
                 split='train', train_ratio=0.8, val_ratio=0.1, index=-1):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.index = index
        self._cache = {}
        self._cache_size = 100


        if temporal_path is None:
            temporal_path = os.path.join(data_dir, 'time_series.npy')
        
        if os.path.exists(temporal_path):
            raw_temporal = np.load(temporal_path).astype(np.float32)
            
            self.time_series = raw_temporal[:, :, np.newaxis]
        else:
            raise FileNotFoundError(f"Temporal data file not found: {temporal_path}")

        if image_path is None:
            image_path = os.path.join(data_dir, 'image_rgb.npy')
        
        if os.path.exists(image_path):
            self.image_data = np.load(image_path).astype(np.float32)
        else:
            self.image_data = None

        total_timesteps = self.time_series.shape[0]
        
        if self.index != -1:
            if self.index == 1:
                self.time_series = self.time_series[:int(total_timesteps * 0.3)]
                if self.image_data is not None:
                    self.image_data = self.image_data[:int(total_timesteps * 0.3)]
                total_timesteps = len(self.time_series)
            else:
                start_idx = int(total_timesteps * (0.3 + (self.index - 2) * 0.175))
                end_idx = int(total_timesteps * (0.3 + (self.index - 1) * 0.175))
                self.time_series = self.time_series[start_idx:end_idx]
                if self.image_data is not None:
                    self.image_data = self.image_data[start_idx:end_idx]
                total_timesteps = len(self.time_series)

        train_end = int(total_timesteps * train_ratio)
        val_end = int(total_timesteps * (train_ratio + val_ratio))
        
        if split == 'train':
            self.start_idx = 0
            self.end_idx = train_end
        elif split == 'val':
            self.start_idx = train_end
            self.end_idx = val_end
        else:
            self.start_idx = val_end
            self.end_idx = total_timesteps
        
        train_slice = self.time_series[:train_end]
        reshaped_train = train_slice.reshape(-1, 1)
        self.data_min = np.min(reshaped_train).astype(np.float32)
        self.data_max = np.max(reshaped_train).astype(np.float32)
        normalized_all = (self.time_series - self.data_min) / (self.data_max - self.data_min)
        self.normalized_data = normalized_all.astype(np.float32)
        
        self.num_timesteps = self.end_idx - self.start_idx
        self.num_samples = self.num_timesteps - seq_len - pred_len + 1
        
        if self.num_samples <= 0:
            raise ValueError(f"Time series too short: total timesteps ({self.num_timesteps}) < sequence length ({seq_len}) + prediction length ({pred_len})")
        
        

        self.normalized_data = self.normalized_data[self.start_idx:self.end_idx]
        if self.image_data is not None:
            self.image_data = self.image_data[self.start_idx:self.end_idx]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx in self._cache:
            cached_result = self._cache[idx].copy()
            cached_result['sample_idx'] = idx
            return cached_result
        
        temporal_data = self.normalized_data[idx:idx+self.seq_len]
        
        target = self.normalized_data[idx+self.seq_len:idx+self.seq_len+self.pred_len]
        
        result = {
            'temporal_data': torch.tensor(temporal_data, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32),
            'sample_idx': idx
        }
        
        if self.image_data is not None:
            image_sequence = self.image_data[idx:idx+self.seq_len]
            image_tensor = torch.tensor(image_sequence, dtype=torch.float32)
            result['images'] = image_tensor
        
        if len(self._cache) < self._cache_size:
            self._cache[idx] = result.copy()
            
        return result


class UniversalTrainDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        
        sample = original_dataset[0]
        self.has_images = 'images' in sample
        self.has_text = 'text' in sample
        self.has_adj_matrix = 'adj_matrix' in sample
        
        self.seq_len = sample['temporal_data'].shape[0]
        self.num_nodes = sample['temporal_data'].shape[1] 
        
        if sample['temporal_data'].dim() == 2:
            self.num_features = 1
        else:
            self.num_features = sample['temporal_data'].shape[2]
            
        self.pred_len = sample['target'].shape[0]
        
        if self.has_images:
            self.image_shape = sample['images'].shape[2:]
        
        if self.has_text:
            pass
        
        if self.has_adj_matrix:
            self.adj_matrix_shape = sample['adj_matrix'].shape
            
        
        
        
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        sample = self.original_dataset[idx]
        
        temporal_data = sample['temporal_data']
        if temporal_data.dim() == 2:
            temporal_data = temporal_data.unsqueeze(-1)
        
        target = sample['target']
        if target.dim() == 2:
            target = target.unsqueeze(-1)
        
        result = {
            'temporal_data': temporal_data,
            'target': target
        }
        
        if 'sample_idx' in sample:
            result['sample_idx'] = sample['sample_idx']
        
        if self.has_images:
            result['images'] = sample['images']
        
        if self.has_text:
            result['text'] = sample['text']
        
        if self.has_adj_matrix:
            result['adj_matrix'] = sample['adj_matrix']
            
        return result


class UniversalValDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        
        sample = original_dataset[0]
        self.has_images = 'images' in sample
        self.has_text = 'text' in sample
        self.has_adj_matrix = 'adj_matrix' in sample
        
        self.seq_len = sample['temporal_data'].shape[0]
        self.num_nodes = sample['temporal_data'].shape[1] 
        
        if sample['temporal_data'].dim() == 2:
            self.num_features = 1
        else:
            self.num_features = sample['temporal_data'].shape[2]
            
        self.pred_len = sample['target'].shape[0]
        
        
        
        if self.has_images:
            self.image_shape = sample['images'].shape[2:]
        
        if self.has_text:
            pass
        
        if self.has_adj_matrix:
            self.adj_matrix_shape = sample['adj_matrix'].shape
            
        
        
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        sample = self.original_dataset[idx]
        
        temporal_data = sample['temporal_data']
        if temporal_data.dim() == 2:
            temporal_data = temporal_data.unsqueeze(-1)
        
        target = sample['target']
        if target.dim() == 2:
            target = target.unsqueeze(-1)
        
        result = {
            'temporal_data': temporal_data,
            'target': target
        }
        
        if self.has_images:
            result['images'] = sample['images']
        
        if self.has_text:
            result['text'] = sample['text']
        
        if self.has_adj_matrix:
            result['adj_matrix'] = sample['adj_matrix']
            
        return result



class UniversalTestDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        
        sample = original_dataset[0]
        self.has_images = 'images' in sample
        self.has_text = 'text' in sample
        self.has_adj_matrix = 'adj_matrix' in sample
        
        self.seq_len = sample['temporal_data'].shape[0]
        self.num_nodes = sample['temporal_data'].shape[1] 
        
        if sample['temporal_data'].dim() == 2:
            self.num_features = 1
        else:
            self.num_features = sample['temporal_data'].shape[2]
            
        self.pred_len = sample['target'].shape[0]
        
        
        
        if self.has_images:
            self.image_shape = sample['images'].shape[2:]
        
        if self.has_text:
            pass
        
        if self.has_adj_matrix:
            self.adj_matrix_shape = sample['adj_matrix'].shape
            
        
        
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        sample = self.original_dataset[idx]
        
        temporal_data = sample['temporal_data']
        if temporal_data.dim() == 2:
            temporal_data = temporal_data.unsqueeze(-1)
        
        target = sample['target']
        if target.dim() == 2:
            target = target.unsqueeze(-1)
        
        result = {
            'temporal_data': temporal_data,
            'target': target
        }
        
        if self.has_images:
            result['images'] = sample['images']
        
        if self.has_text:
            result['text'] = sample['text']
        
        if self.has_adj_matrix:
            result['adj_matrix'] = sample['adj_matrix']
            
        return result


def collate_fn(batch):
    temporal_data = torch.stack([item['temporal_data'] for item in batch], dim=0)
    target = torch.stack([item['target'] for item in batch], dim=0)
    
    out = {
        'temporal_data': temporal_data,
        'target': target
    }
    
    if 'sample_idx' in batch[0]:
        sample_indices = [item['sample_idx'] for item in batch]
        out['sample_indices'] = torch.tensor(sample_indices, dtype=torch.long)
    
    if 'text' in batch[0]:
        text_data = [item['text'] for item in batch]
        out['text'] = text_data
    
    if 'images' in batch[0]:
        images = torch.stack([item['images'] for item in batch], dim=0)
        out['images'] = images
    
    if 'adj_matrix' in batch[0]:
        adj_matrix = batch[0]['adj_matrix']
        out['adj_matrix'] = adj_matrix
        
    return out


def create_dataloader(dataset, batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True, generator=None, worker_init_fn=None):
    dataloader_kwargs = {
        'dataset': dataset,
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'collate_fn': collate_fn,
        'drop_last': drop_last,
    }
    
    if num_workers > 0:
        dataloader_kwargs['persistent_workers'] = True
        dataloader_kwargs['prefetch_factor'] = 4
        if worker_init_fn is not None:
            dataloader_kwargs['worker_init_fn'] = worker_init_fn
    if generator is not None:
        dataloader_kwargs['generator'] = generator
    
    return DataLoader(**dataloader_kwargs)


def create_dataloaders(args, index=-1):
    
    
    if args.dataset == 'BjTT':
        train_dataset_raw = BjTTDataset(
            data_dir=args.data_dir,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            num_nodes=args.num_nodes,
            split='train',
            index=index
        )
        
        val_dataset_raw = BjTTDataset(
            data_dir=args.data_dir,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            num_nodes=args.num_nodes,
            split='val',
            index=index
        )
        
        test_dataset_raw = BjTTDataset(
            data_dir=args.data_dir,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            num_nodes=args.num_nodes,
            split='test',
            index=index
        )
        
        train_dataset = UniversalTrainDataset(train_dataset_raw)
        val_dataset = UniversalValDataset(val_dataset_raw)
        test_dataset = UniversalTestDataset(test_dataset_raw)
        
    elif args.dataset == 'Terra':
        lat_range = getattr(args, 'lat_range', (50, 60))
        lon_range = getattr(args, 'lon_range', (-8, 2))
        
        train_dataset_raw = TerraDataset(
            image_dir=args.image_dir,
            text_dir=args.text_dir,
            time_series_path=args.time_series_path,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            split='train',
            image_size=getattr(args, 'image_size', (128, 128)),
            channels=getattr(args, 'channels', 1),
            lat_range=lat_range,
            lon_range=lon_range,
            index=index
        )
        
        val_dataset_raw = TerraDataset(
            image_dir=args.image_dir,
            text_dir=args.text_dir,
            time_series_path=args.time_series_path,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            split='val',
            image_size=getattr(args, 'image_size', (128, 128)),
            channels=getattr(args, 'channels', 1),
            lat_range=lat_range,
            lon_range=lon_range,
            index=index
        )
        test_dataset_raw = TerraDataset(
            image_dir=args.image_dir,
            text_dir=args.text_dir,
            time_series_path=args.time_series_path,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            split='test',
            image_size=getattr(args, 'image_size', (128, 128)),
            channels=getattr(args, 'channels', 1),
            lat_range=lat_range,
            lon_range=lon_range,
            index=index
        )
        
        train_dataset = TerraTrainDataset(train_dataset_raw)
        val_dataset = TerraValDataset(val_dataset_raw)
        test_dataset = TerraTestDataset(test_dataset_raw)
        
    elif args.dataset == 'PEMS04':
        train_dataset_raw = PEMS04Dataset(
            data_dir=args.data_dir,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            num_nodes=307,
            split='train',
            train_ratio=0.8,
            val_ratio=0.1,
            index=index
        )
        
        val_dataset_raw = PEMS04Dataset(
            data_dir=args.data_dir,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            num_nodes=307,
            split='val',
            train_ratio=0.8,
            val_ratio=0.1,
            index=index
        )
        
        test_dataset_raw = PEMS04Dataset(
            data_dir=args.data_dir,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            num_nodes=307,
            split='test',
            train_ratio=0.8,
            val_ratio=0.1,
            index=index
        )
        
        train_dataset = UniversalTrainDataset(train_dataset_raw)
        val_dataset = UniversalValDataset(val_dataset_raw)
        test_dataset = UniversalTestDataset(test_dataset_raw)
        
    elif args.dataset == 'GreenEarthNet':
        train_dataset_raw = GreenEarthNetDataset(
            data_dir=args.data_dir,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            num_nodes=args.num_nodes,
            temporal_path=getattr(args, 'temporal_path', None),
            image_path=getattr(args, 'image_path', None),
            split='train',
            train_ratio=0.8,
            val_ratio=0.1,
            index=index
        )
        
        val_dataset_raw = GreenEarthNetDataset(
            data_dir=args.data_dir,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            num_nodes=args.num_nodes,
            temporal_path=getattr(args, 'temporal_path', None),
            image_path=getattr(args, 'image_path', None),
            split='val',
            train_ratio=0.8,
            val_ratio=0.1,
            index=index
        )
        
        test_dataset_raw = GreenEarthNetDataset(
            data_dir=args.data_dir,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            num_nodes=args.num_nodes,
            temporal_path=getattr(args, 'temporal_path', None),
            image_path=getattr(args, 'image_path', None),
            split='test',
            train_ratio=0.8,
            val_ratio=0.1,
            index=index
        )
        
        train_dataset = UniversalTrainDataset(train_dataset_raw)
        val_dataset = UniversalValDataset(val_dataset_raw)
        test_dataset = UniversalTestDataset(test_dataset_raw)
        
    
    
    seed = getattr(args, 'seed', 42)
    g = torch.Generator()
    g.manual_seed(seed)
    def _worker_init_fn(worker_id):
        worker_seed = (seed + worker_id) % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=getattr(args, 'num_workers', 0),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        generator=g,
        worker_init_fn=_worker_init_fn
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=getattr(args, 'num_workers', 0),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=g,
        worker_init_fn=_worker_init_fn
    )
    
    test_loader = create_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=getattr(args, 'num_workers', 0),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=g,
        worker_init_fn=_worker_init_fn
    )
    return train_loader, val_loader, test_loader


class TerraTrainDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.num_nodes = len(original_dataset)
        
        self._reorganize_data()

    def _reorganize_data(self):
        img_list = []
        text_list = []
        x_samples_list = []
        y_samples_list = []

        for i in range(self.num_nodes):
            try:
                sample = self.original_dataset[i]
                img_tensor = sample['images']
                img_list.append(img_tensor)
                text_data = sample['text']
                text_list.append(text_data)
                x_samples_tensor = sample['temporal_data']
                x_samples_list.append(x_samples_tensor)
                y_samples_tensor = sample['target']
                y_samples_list.append(y_samples_tensor)
            except Exception as e:
                continue
        self.img_data = torch.stack(img_list, dim=0)
        self.text_data = text_list
        self.x_samples_data = torch.stack(x_samples_list, dim=0)
        self.x_samples_data = self.x_samples_data.permute(1, 0, 2)
        self.y_samples_data = torch.stack(y_samples_list, dim=0)
        self.y_samples_data = self.y_samples_data.permute(1, 0, 2)

    def __len__(self):
        return self.x_samples_data.shape[0]

    def __getitem__(self, idx):
        img_tensor = self.img_data
        text_data = self.text_data
        x_samples_tensor = self.x_samples_data[idx]
        y_samples_tensor = self.y_samples_data[idx]
        temporal_data = x_samples_tensor.permute(1, 0).unsqueeze(-1)
        target = y_samples_tensor.permute(1, 0).unsqueeze(-1)
        
        images = img_tensor
        
        text = text_data

        return {
            'temporal_data': temporal_data,
            'images': images,
            'text': text,
            'target': target,
            'sample_idx': idx
        }


class TerraValDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.num_nodes = len(original_dataset)
        self._reorganize_data()

    def _reorganize_data(self):
        img_list = []
        text_list = []
        x_samples_list = []
        y_samples_list = []
        for i in range(self.num_nodes):
            try:
                sample = self.original_dataset[i]
                img_tensor = sample['images']
                img_list.append(img_tensor)
                text_data = sample['text']
                text_list.append(text_data)
                x_samples_tensor = sample['temporal_data']
                if x_samples_tensor.dim() == 3 and x_samples_tensor.shape[-1] == 1:
                    x_samples_tensor = x_samples_tensor.squeeze(-1)
                x_samples_list.append(x_samples_tensor)
                y_samples_tensor = sample['target']
                if y_samples_tensor.dim() == 3 and y_samples_tensor.shape[-1] == 1:
                    y_samples_tensor = y_samples_tensor.squeeze(-1)
                y_samples_list.append(y_samples_tensor)
            except Exception as e:
                continue
        self.img_data = torch.stack(img_list, dim=0)
        self.text_data = text_list
        self.x_samples_data = torch.stack(x_samples_list, dim=0)
        self.x_samples_data = self.x_samples_data.permute(1, 0, 2)
        self.y_samples_data = torch.stack(y_samples_list, dim=0)
        self.y_samples_data = self.y_samples_data.permute(1, 0, 2)

    def __len__(self):
        return self.x_samples_data.shape[0]

    def __getitem__(self, idx):
        img_tensor = self.img_data
        text_data = self.text_data
        x_samples_tensor = self.x_samples_data[idx]
        y_samples_tensor = self.y_samples_data[idx]
        temporal_data = x_samples_tensor.permute(1, 0).unsqueeze(-1)
        target = y_samples_tensor.permute(1, 0).unsqueeze(-1)
        images = img_tensor
        text = text_data

        return {
            'temporal_data': temporal_data,
            'images': images,
            'text': text,
            'target': target,
            'sample_idx': idx
        }


class TerraTestDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.num_nodes = len(original_dataset)
        self._reorganize_data()

    def _reorganize_data(self):
        img_list = []
        text_list = []
        x_samples_list = []
        y_samples_list = []
        for i in range(self.num_nodes):
            try:
                sample = self.original_dataset[i]
                img_tensor = sample['images']
                img_list.append(img_tensor)
                text_data = sample['text']
                text_list.append(text_data)
                x_samples_tensor = sample['temporal_data']
                if x_samples_tensor.dim() == 3 and x_samples_tensor.shape[-1] == 1:
                    x_samples_tensor = x_samples_tensor.squeeze(-1)
                x_samples_list.append(x_samples_tensor)
                y_samples_tensor = sample['target']
                if y_samples_tensor.dim() == 3 and y_samples_tensor.shape[-1] == 1:
                    y_samples_tensor = y_samples_tensor.squeeze(-1)
                y_samples_list.append(y_samples_tensor)
            except Exception as e:
                continue
        self.img_data = torch.stack(img_list, dim=0)
        self.text_data = text_list
        self.x_samples_data = torch.stack(x_samples_list, dim=0)
        self.x_samples_data = self.x_samples_data.permute(1, 0, 2)
        self.y_samples_data = torch.stack(y_samples_list, dim=0)
        self.y_samples_data = self.y_samples_data.permute(1, 0, 2)

    def __len__(self):
        return self.x_samples_data.shape[0]

    def __getitem__(self, idx):
        img_tensor = self.img_data
        text_data = self.text_data
        x_samples_tensor = self.x_samples_data[idx]
        y_samples_tensor = self.y_samples_data[idx]
        temporal_data = x_samples_tensor.permute(1, 0).unsqueeze(-1)
        target = y_samples_tensor.permute(1, 0).unsqueeze(-1)
        images = img_tensor
        text = text_data

        return {
            'temporal_data': temporal_data,
            'images': images,
            'text': text,
            'target': target,
            'sample_idx': idx
        }





