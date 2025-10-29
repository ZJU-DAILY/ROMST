import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import sys
import os
import random
import numpy as np
import torch.nn.functional as F
import logging
from datetime import datetime
from transformers.models import FalconForQuestionAnswering
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mydatasets.dataset import create_dataloaders
from mydatasets.loss import get_loss, InverseNormalizer
from mydatasets.text_encoder import BjTTTextEncoder, BjTTTextFeatureExtractor
from model.OptimizedRLFusion import UltimateMultiModalRLModel
from model.TextFeatureExtractor import TextFeatureExtractor
from model.SpatioTemporatest import SpatioTemporalModel
from model.SpatialTemporalAugmentation import SpatialTemporalAugmentation
from mydatasets.text_encoder import BjTTTextEncoder, BjTTTextFeatureExtractor
from mydatasets.terra_text_encoder import TerraTextEncoder, TerraTextFeatureExtractor
from tqdm import tqdm
import argparse
from test6 import test as test_fn
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import thop


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':16:8')
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass


def setup_logging(save_dir, dataset=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(save_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_log_{dataset + '_' if dataset else ''}{timestamp}.log")
    logger = logging.getLogger("stage2")
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    logger.info(f"log file: {log_file}")
    return logger

def compute_loss(predictions, targets, inverse_normalizer=None):
    if inverse_normalizer is not None:
        pred_denorm = inverse_normalizer.inverse_normalize_tensor(predictions)
        y_denorm = inverse_normalizer.inverse_normalize_tensor(targets)
    else:
        pred_denorm = predictions
        y_denorm = targets
    return nn.functional.mse_loss(pred_denorm, y_denorm)

def setup_text_precoding(args, index=-1):
   
    if not args.use_text:
        return None, 0.0
    
    if args.dataset == 'BjTT':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        import time
        time.sleep(1)
        text_model = TextFeatureExtractor(
            model_name='llama 1b',
            device=args.device,
            max_length=args.maxlen
        )
        text_model.prune_model(pruning_ratio=args.pruning_ratio, method="custom_synflow")
        print(f"Text model pruned (ratio: {args.pruning_ratio})")
        text_encoder = BjTTTextEncoder(
            text_model=text_model,
            cache_dir=f"./bjtt_text_cache_{args.pruning_ratio}",
            max_length=args.maxlen,
            max_timesteps=3200,
            index=index
        )
        
        cache_exists = text_encoder.check_cache_exists()
        precompute_secs = 0.0
        if cache_exists:
            print("Found cached text features, loading...")
            text_features = text_encoder.load_cached_features()
            print(f"Cache loaded: {text_features.shape}")
        else:
            print("No cache found, precomputing text features...")
            t0 = time.time()
            text_features = text_encoder.precompute_features(
                data_dir=args.data_dir, 
                force_recompute=getattr(args, 'force_recompute_text', False),
                index=index
            )
            precompute_secs = time.time() - t0
            print(f"Text pre-encoding finished: {text_features.shape}")
        text_feature_extractor = BjTTTextFeatureExtractor(text_encoder)
        return text_feature_extractor, precompute_secs
        
    elif args.dataset == 'Terra':

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        import time
        time.sleep(1)
        text_model = TextFeatureExtractor(
            model_name='llama 1b',
            device=args.device,
            max_length=args.maxlen
        )
    
        text_model.prune_model(pruning_ratio=args.pruning_ratio, method="custom_synflow")
        print(f"Text model pruned (ratio: {args.pruning_ratio})")
        text_encoder = TerraTextEncoder(
            text_model=text_model,
            cache_dir=f"./terra_text_cache_{args.pruning_ratio}",
            max_length=args.maxlen,
            num_nodes=args.num_nodes,
            index=index
        )
        
        cache_exists = text_encoder.check_cache_exists()
        precompute_secs = 0.0
        if cache_exists:
            print("Found Terra text feature cache, loading...")
            text_features = text_encoder.load_cached_features()
            print(f"Cache loaded: {text_features.shape}")
        else:
            print("No cache found, precomputing Terra text features...")
            t0 = time.time()
            text_features = text_encoder.precompute_features(
                text_dir=args.text_dir,
                force_recompute=getattr(args, 'force_recompute_text', False),
                index=index
            )
            precompute_secs = time.time() - t0
            print(f"Terra text pre-encoding finished: {text_features.shape}")
        
        text_feature_extractor = TerraTextFeatureExtractor(text_encoder, hidden_dim=args.hidden_dim)
        
        return text_feature_extractor, precompute_secs
        
    else:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        import time
        time.sleep(1)
        text_model = TextFeatureExtractor(
            model_name='llama 1b',
            device=args.device,
            max_length=args.maxlen
        )
        text_model.prune_model(pruning_ratio=args.pruning_ratio, method="custom_synflow")
        print(f"Text model pruned (ratio: {args.pruning_ratio})")
 
        return text_model, 0.0


def train_spatiotemporal_model(train_loader, val_loader, inverse_normalizer, args, current_index=None, init_model_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_start_time = time.time()
    logger = logging.getLogger(__name__)
    
    model = SpatioTemporalModel(
        temporal_input_dim=args.temporal_input_dim,
        output_dim=args.output_dim,
        seq_len=args.seq_len,
        pred_len=args.pred_len
    ).to(device)
    
    if init_model_path is not None:
        if not os.path.exists(init_model_path):
            raise FileNotFoundError(f"Previous stage spatiotemporal model not found: {init_model_path}")
        try:
            ckpt_prev = torch.load(init_model_path, map_location=device)
            model.load_state_dict(ckpt_prev['model_state_dict'], strict=True)
            msg = f"Loaded weights from previous spatiotemporal model: {init_model_path}"
            print(msg)
            logger.info(msg)
        except Exception as e:
            raise RuntimeError(f"Failed to load previous spatiotemporal model: {init_model_path}. Error: {e}")
    
    prior_matrix_path = os.path.join(args.prior_matrix_path, f'{args.dataset}_matrix_prior.npy')
    adj_matrix = np.load(prior_matrix_path)
    logger.info(f"Loaded initial prior matrix, shape: {adj_matrix.shape}")
    
    scaler_amp = None
    augmentation_module = SpatialTemporalAugmentation(alpha=getattr(args, 'aug_alpha', 0.7)) if getattr(args, 'use_augmentation', False) else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, eps=1e-8)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=1e-6)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    logger.info("="*60)
    logger.info("Stage 1: Train SpatioTemporalModel")
    logger.info("="*60)
    
    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        
        matrix = adj_matrix
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Stage1 Epoch {epoch+1}/{args.epochs}", dynamic_ncols=True, mininterval=0.5, file=sys.stdout, leave=True)):
            temporal_data = batch['temporal_data'].to(device, non_blocking=True)
            targets = batch['target'].to(device, non_blocking=True)
            
            adj_matrix_tensor = torch.tensor(matrix, dtype=torch.float32, device=device)
            
            optimizer.zero_grad()
            
            try:
                predictions, _ = model(temporal_data, adj_matrix_tensor, return_prediction=True)
                base_loss = compute_loss(predictions, targets, inverse_normalizer)
                contrastive_loss = None
                if augmentation_module is not None:
                    def augmented_forward(aug_temporal_data):
                        preds_aug, _ = model(aug_temporal_data, adj_matrix_tensor, return_prediction=True)
                        return preds_aug
                    contrastive_loss, _ = augmentation_module(
                        temporal_data, predictions, targets, augmented_forward, adj_matrix_tensor
                    )
                loss = base_loss + (getattr(args, 'contrastive_weight', 0.2) * contrastive_loss if contrastive_loss is not None else 0.0)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"[Train Warning] Batch {batch_idx}: loss is nan/inf, skip this batch")
                    optimizer.zero_grad()
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss_sum += loss.item()
                train_batches += 1
                
                if batch_idx % 50 == 0:
                    mae, rmse, mape = get_loss(predictions, targets, inverse_normalizer)
                    log_msg = f"  Batch {batch_idx}: Loss={base_loss.item():.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%"
                    logger.info(log_msg)
                    
            except Exception as e:
                logger.error(f"[Train Exception] Batch {batch_idx}: {e}")
                optimizer.zero_grad()
                continue
                
        avg_train_loss = train_loss_sum / max(train_batches, 1)
        
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                temporal_data = batch['temporal_data'].to(device, non_blocking=True)
                targets = batch['target'].to(device, non_blocking=True)
                
                adj_matrix_tensor = torch.tensor(matrix, dtype=torch.float32, device=device)
                predictions, _ = model(temporal_data, adj_matrix_tensor, return_prediction=True)
                loss = compute_loss(predictions, targets, inverse_normalizer)
                val_loss_sum += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss_sum / max(val_batches, 1)
        scheduler.step(avg_val_loss)
        
        logger.info(f"Epoch [{epoch+1:3d}/{args.epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            dataset_save_dir = os.path.join(args.save_dir, args.dataset)
            os.makedirs(dataset_save_dir, exist_ok=True)
            if current_index is not None:
                best_model_path = os.path.join(dataset_save_dir, f'best_spatiotemporal_model_index_{current_index}_aug_{args.contrastive_weight}_prune_{args.pruning_ratio}.pth')
            else:
                best_model_path = os.path.join(dataset_save_dir, 'best_spatiotemporal_model_aug_{args.contrastive_weight}_prune_{args.pruning_ratio}.pth')
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler_amp.state_dict() if scaler_amp else None,
                'loss': best_val_loss,
                'args': args
            }, best_model_path)
            logger.info(f"✓ New best spatiotemporal model saved: {best_model_path}")
            best_model_path_saved = best_model_path
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 2 == 0:
            torch.cuda.empty_cache()
    
    total_secs = int(time.time() - train_start_time)
    h = total_secs // 3600
    m = (total_secs % 3600) // 60
    s = total_secs % 60
    logger.info(f"Stage 1 training time: {h:02d}:{m:02d}:{s:02d}")
    logger.info("Stage 1 training finished")
    
    if 'best_model_path_saved' not in locals():
        raise RuntimeError("Stage 1 did not produce a best model file")
    return best_model_path_saved, total_secs


def get_batch_indices(batch, dataset_info):
    
    if 'sample_indices' in batch:
        indices = batch['sample_indices']
        if torch.is_tensor(indices):
            indices = indices.tolist()
        return indices
    else:
        batch_size = batch['temporal_data'].shape[0]
        return list(range(batch_size))

def measure_eval_efficiency(eval_model, data_loader, args, text_feature_extractor=None):
    logger = logging.getLogger(__name__)
    if not getattr(args, 'use_text', True):
        logger.debug("[FLOPs] skip: args.use_text is False")
        return float('nan')
    try:
        model_device = next(eval_model.parameters()).device
    except Exception:
        model_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = model_device
    logger.debug(f"[FLOPs] model_device={device}")
    try:
        batch = next(iter(data_loader))
    except Exception as e:
        logger.warning(f"[FLOPs] unable to fetch batch from data_loader: {e}")
        return float('nan')

    temporal_data = batch['temporal_data'].to(device, non_blocking=True)
    images = batch.get('images')
    if images is not None:
        images = images.to(device, non_blocking=True)
    prior_matrix_path = os.path.join(args.prior_matrix_path, f'{args.dataset}_matrix_prior.npy')
    try:
        matrix = np.load(prior_matrix_path)
    except Exception as e:
        logger.warning(f"[FLOPs] failed to load prior matrix: {prior_matrix_path}, err={e}")
        return float('nan')
    adj_matrix = torch.tensor(matrix, dtype=torch.float32, device=device)

    dataset_info = {'seq_len': args.seq_len, 'pred_len': args.pred_len}
    if args.use_text and text_feature_extractor is not None:
        if args.dataset == 'BjTT':
            batch_indices = get_batch_indices(batch, dataset_info)
            text_data = batch_indices
            text = None
        elif args.dataset == 'Terra':
            text_data = None
            text = None
        else:
            text_data = None
            text = batch.get('text', None)
    else:
        text_data = None
        text = None

    was_training = eval_model.training
    eval_model.eval()
    gflops = float('nan')
    try:
        with torch.no_grad():
            macs, _ = thop.profile(
                eval_model,
                inputs=(temporal_data, text_data, images, adj_matrix, None, False, None, text),
                verbose=False
            )
        gflops = float(macs) / 1e9
        logger.debug(
            f"[FLOPs] computed: gflops={gflops:.4f}, temporal={tuple(temporal_data.shape)}, "
            f"images={(tuple(images.shape) if images is not None else None)}, adj={tuple(adj_matrix.shape)}"
        )
    except Exception as e:
        logger.warning(f"[FLOPs] thop.profile failed: {e}")
        gflops = float('nan')

    if was_training:
        eval_model.train()

    del temporal_data, images, adj_matrix
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return gflops

    

def train_multimodal_model(train_loader, val_loader, scaler, args, baseline_model_path, text_feature_extractor=None, current_index=None, init_model_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_start_time = time.time()
    logger = logging.getLogger(__name__)
    
    
    if args.use_text and text_feature_extractor is not None:
        if args.dataset == 'BjTT':
            text_model = None
            text_feature_extractor_param = text_feature_extractor
        elif args.dataset == 'Terra':
            text_model = None
            text_feature_extractor_param = text_feature_extractor
        else:
            text_model = text_feature_extractor
            text_feature_extractor_param = None
    else:
        text_model = None
        text_feature_extractor_param = None
    
    internal_baseline = baseline_model_path
    model = UltimateMultiModalRLModel(
        num_nodes=args.num_nodes,
        temporal_input_dim=args.temporal_input_dim,
        hidden_dim=args.hidden_dim,
        st_output_dim=args.output_dim,
        num_mamba_layers=args.num_mamba_layers,
        text_model=text_model,
        image_channels=args.image_channels,
        pred_len=args.pred_len,
        use_image=args.use_image,
        use_text=args.use_text,
        maxlen=args.maxlen,
        text_feature_extractor=text_feature_extractor_param,
        baseline_model_path=internal_baseline
    ).to(device)

    
    if init_model_path is not None:
        if not os.path.exists(init_model_path):
            raise FileNotFoundError(f"Previous best model not found: {init_model_path}")
        try:
            ckpt_prev = torch.load(init_model_path, map_location=device)
            model.load_state_dict(ckpt_prev['model_state_dict'], strict=True)
            msg = f"Strictly loaded weights from previous best model: {init_model_path}"
            print(msg)
            logger.info(msg)
        except Exception as e:
            raise RuntimeError(f"Failed to strictly load previous weights: {init_model_path}. Error: {e}")

   
    prior_matrix_path = os.path.join(args.prior_matrix_path, f'{args.dataset}_matrix_prior.npy')
    adj_matrix = np.load(prior_matrix_path)
    print(f"Loaded initial prior matrix, shape: {adj_matrix.shape}")
    
    scaler_amp = None
    
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, eps=1e-8)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-8)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    torch.cuda.empty_cache()
    dataset_info = {
        'seq_len': args.seq_len,
        'pred_len': args.pred_len
    }
    
    best_model_path_saved = None
    measured_train_peak = False
    train_gpu_peak_mb = float('nan')
    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        
        matrix = adj_matrix
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{args.epochs}")):
            temporal_data = batch['temporal_data'].to(device, non_blocking=True)
            targets = batch['target'].to(device, non_blocking=True)
            images = batch.get('images')
            if images is not None:
                images = images.to(device, non_blocking=True)
            
            adj_matrix = torch.tensor(matrix, dtype=torch.float32, device=device)
            
            if args.use_text and text_feature_extractor is not None:
                if args.dataset == 'BjTT':
                    batch_indices = get_batch_indices(batch, dataset_info)
                    text_data = batch_indices
                    text = None
                elif args.dataset == 'Terra':
                    text_data = None
                    text = None
                else:
                    text_data = None
                    text = batch.get('text', None)
            else:
                text_data = None
                text = None
            
            try:
                if batch_idx % 5 == 0:
                    torch.cuda.empty_cache()
                if args.use_text and torch.cuda.is_available() and (batch_idx == 0) and (not measured_train_peak):
                    torch.cuda.reset_peak_memory_stats()
                
                if True:
                    outputs = model(temporal_data, text_data, images, adj_matrix, targets, training=True, epoch=epoch, text=text)
                    predictions = outputs[0] if isinstance(outputs, tuple) else outputs
                    loss = compute_loss(predictions, targets, scaler) / args.gradient_accumulation_steps
                    loss.backward()

                    if args.use_text and (batch_idx == 0) and (not measured_train_peak):
                        try:
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                                train_gpu_peak_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                        except Exception:
                            train_gpu_peak_mb = float('nan')
                        measured_train_peak = True
                
                train_loss_sum += loss.item() * args.gradient_accumulation_steps
                train_batches += 1
                
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    torch.cuda.empty_cache()
                
                if batch_idx % 50 == 0:
                    mae, rmse, mape = get_loss(predictions, targets, scaler)
                    msg1 = f"  Batch {batch_idx}: Loss={loss.item() * args.gradient_accumulation_steps:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%"
                    msg2 = (f"  Pred range: [{predictions.min().item():.2f}, {predictions.max().item():.2f}], "
                           f"Target range: [{targets.min().item():.2f}, {targets.max().item():.2f}]")
                    print(msg1)
                    print(msg2)
                    logger.info(msg1)
                    logger.info(msg2)
                         
            except Exception as e:
                print(f"[Train Exception] Batch {batch_idx}: {e}")
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue
                
        avg_train_loss = train_loss_sum / train_batches
        
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                temporal_data = batch['temporal_data'].to(device, non_blocking=True)
                targets = batch['target'].to(device, non_blocking=True)
                images = batch.get('images')
                if images is not None:
                    images = images.to(device, non_blocking=True)
                
                adj_matrix = torch.tensor(matrix, dtype=torch.float32, device=device)
                
                if args.use_text and text_feature_extractor is not None:
                    if args.dataset == 'BjTT':
                        batch_indices = get_batch_indices(batch, dataset_info)
                        text_data = batch_indices
                        text = None
                    elif args.dataset == 'Terra':
                        text_data = None
                        text = None
                    else:
                        text_data = None
                        text = batch.get('text', None)
                else:
                    text_data = None
                    text = None
                
                outputs = model(temporal_data, text_data, images, adj_matrix, targets=None, training=False, text=text)
                predictions = outputs[0] if isinstance(outputs, tuple) else outputs
                loss = compute_loss(predictions, targets, scaler)
                val_loss_sum += loss.item()
                val_batches += 1
                
        avg_val_loss = val_loss_sum / max(val_batches, 1)
        scheduler.step(avg_val_loss)
        
        print(f"Epoch [{epoch+1:3d}/{args.epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        logger.info(f"Epoch [{epoch+1:3d}/{args.epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            os.makedirs(args.save_dir, exist_ok=True)
            dataset_save_dir = os.path.join(args.save_dir, args.dataset)
            os.makedirs(dataset_save_dir, exist_ok=True)
            if current_index is not None:
                best_model_path = os.path.join(dataset_save_dir, f'best_multimodal_model_index_{current_index}_aug_{args.contrastive_weight}_prune_{args.pruning_ratio}.pth')
            else:
                best_model_path = os.path.join(dataset_save_dir, f'best_multimodal_model_aug_{args.contrastive_weight}_prune_{args.pruning_ratio}.pth')
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler_amp.state_dict() if scaler_amp else None,
                'loss': best_val_loss,
                'args': args
            }, best_model_path)
            print(f"✓ New best model saved: {best_model_path}")
            logger.info(f"New best model saved: {best_model_path}")
            best_model_path_saved = best_model_path
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
        if (epoch + 1) % 2 == 0:
            torch.cuda.empty_cache()
    
    total_secs = int(time.time() - train_start_time)
    h = total_secs // 3600
    m = (total_secs % 3600) // 60
    s = total_secs % 60
    print(f"Total training time: {h:02d}:{m:02d}:{s:02d}")
    logger.info(f"Total training time: {h:02d}:{m:02d}:{s:02d}")
    print("Stage 2 training finished")
    logger.info("Stage 2 training finished")
   
    if getattr(args, 'enable_reward_viz', False):
        try:
            log_dir = os.path.join(args.save_dir, 'reward')
            os.makedirs(log_dir, exist_ok=True)
            idx_tag = current_index if current_index is not None else 'all'
            
            reward_history = model.get_reward_history()
            
            if len(reward_history) > 0:
                steps = np.array([int(x[0]) for x in reward_history], dtype=int)
                rewards = np.array([float(x[1]) for x in reward_history], dtype=float)

                csv_path = os.path.join(log_dir, f"reward_trend_{args.dataset}_index_{idx_tag}_aug_{args.contrastive_weight}_prune_{args.pruning_ratio}.csv")
                with open(csv_path, 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(['step', 'reward'])
                    for s, r in zip(steps.tolist(), rewards.tolist()):
                        w.writerow([s, f"{r:.6f}"])
                
                plt.figure(figsize=(7, 4))
                ax = plt.gca()
                ax.plot(steps, rewards, linewidth=1.5, alpha=0.6, color='blue')
                for side in ('top', 'right', 'bottom', 'left'):
                    ax.spines[side].set_visible(True)
                ax.tick_params(axis='x', which='both', bottom=True, top=False, direction='in')
                ax.tick_params(axis='y', which='both', left=True, right=False, direction='in')
                ax.set_xlabel('Steps', fontname='Times New Roman')
                ax.set_ylabel('Reward', fontname='Times New Roman')
                plt.tight_layout()
                
                png_path = os.path.join(log_dir, f"reward_trend_{args.dataset}_index_{idx_tag}_aug_{args.contrastive_weight}_prune_{args.pruning_ratio}.png")
                plt.savefig(png_path, dpi=150)
                plt.close()
                
                logger.info(f"Saved reward trend figure: {png_path}")
                logger.info(f"Saved reward trend CSV: {csv_path}")
                
                model.clear_reward_history()
            else:
                logger.warning("Empty reward history, skip visualization")
        except Exception as e:
            logger.warning(f"Failed to save reward visualization: {e}")
   
    epochs_run = epoch + 1
    return best_model_path_saved, total_secs, epochs_run, train_gpu_peak_mb

def parse_arguments():
    parser = argparse.ArgumentParser(description='Stage 2 Training: UltimateMultiModalRLModel')
    parser.add_argument('--dataset', type=str, default='Terra')
    parser.add_argument('--data_dir', type=str, default='mydatasets/Terra')
    parser.add_argument('--prior_matrix_path', type=str, default='mydatasets/prior_matrix')
    parser.add_argument('--index', type=int, default=-1, help='Data split index (-1=all, 1-5=continual)')
    parser.add_argument('--time_series_path', type=str, default='mydatasets/Terra/time_series/wind_daily.npy')
    parser.add_argument('--image_dir', type=str, default='mydatasets/Terra/image')
    parser.add_argument('--text_dir', type=str, default='mydatasets/Terra/texts')
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--maxlen', type=int, default=966)
    parser.add_argument('--num_nodes', type=int, default=100)
    parser.add_argument('--temporal_input_dim', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=64)   
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--num_mamba_layers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--early_stop_patience', type=int, default=10)
    parser.add_argument('--use_image', action='store_true', default=True)
    parser.add_argument('--image_channels', type=int, default=1, help='Image channels')
    parser.add_argument('--use_text', action='store_true', default=True)
    parser.add_argument('--force_recompute_text', action='store_true', default=False, help='Force recompute text features')
    parser.add_argument('--baseline_model_path', type=str, default=None, help='Stage 1 model path')
    parser.add_argument('--pruning_ratio', type=float, default=0.3, help='Text model pruning ratio (0.0-1.0)')
    parser.add_argument('--use_augmentation', action='store_true', default=True, help='Enable contrastive augmentation')
    parser.add_argument('--contrastive_weight', type=float, default=0.6, help='Contrastive loss weight')
    parser.add_argument('--aug_alpha', type=float, default=0.6, help='Augmentation alpha')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='./checkpoint')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--enable_reward_viz', action='store_true', default=True, help='Enable RL reward step logging and visualization')
    #special parameters for Terra dataset
    parser.add_argument('--image_size', type=int, nargs=2, default=[128, 128], help='Image size (h, w)')
    parser.add_argument('--channels', type=int, default=1)
    parser.add_argument('--lat_range', type=int, nargs=2, default=[50, 60], help='Latitude range')
    parser.add_argument('--lon_range', type=int, nargs=2, default=[-8, 2], help='Longitude range')
    return parser.parse_args()

def main():
    set_seed(42)
    args = parse_arguments()
    
 
    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logging(args.save_dir, f"{args.dataset}_p{args.pruning_ratio}_c{args.contrastive_weight}")
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    hdr = [
        f"Dataset: {args.dataset}",
        f"Epochs: {args.epochs}",
        f"Learning rate: {args.learning_rate}",
        f"Batch size: {args.batch_size}",
        f"Grad accumulation steps: {args.gradient_accumulation_steps}",
        f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}",
        f"Use text: {args.use_text}",
        f"Text pruning ratio: {args.pruning_ratio}",
        f"Use image: {args.use_image}",
        f"Use augmentation: {args.use_augmentation}",
        f"Contrastive weight: {args.contrastive_weight}"
    ]
    for line in hdr:
        print(line)
        logger.info(line)
    print("="*60)
    
    train_loader, val_loader, test_loader = create_dataloaders(args, index=args.index)
    train_dataset_raw = train_loader.dataset.original_dataset
    scaler = InverseNormalizer(train_dataset_raw.data_min, train_dataset_raw.data_max)
    if True:
        print("Continual learning mode: train Index 1-5 (two stages)")
        logger.info("Continual learning mode: train Index 1-5 (two stages)")
        prev_best_st_model = None
        prev_best_multimodal_model = None
        all_metrics = []
        real_train_secs_list = []
        real_infer_secs_list = []
        stage2_epochs_list = []
        gflops_list = []
        gpu_train_peak_list = []
        gpu_infer_peak_list = []
        text_feature_extractor = None
        text_precompute_secs = 0.0
        
        for index in range(2, 3):
            print(f"\n{'='*60}")
            print(f"Start training Index {index}")
            logger.info(f"{'='*60}")
            logger.info(f"Start training Index {index}")
            seg_start = time.time()
            
            train_loader, val_loader, test_loader = create_dataloaders(args, index=index)
            train_dataset_raw = train_loader.dataset.original_dataset
            scaler = InverseNormalizer(train_dataset_raw.data_min, train_dataset_raw.data_max)
            if args.use_text:
                text_feature_extractor, text_precompute_secs = setup_text_precoding(args, index=index)
            else:
                text_feature_extractor = None
                text_precompute_secs = 0.0
            
            print(f"\n--- Index {index} Stage 1: Train spatiotemporal model ---")
            logger.info(f"--- Index {index} Stage 1: Train spatiotemporal model ---")
            current_st_model_path, stage1_train_secs = train_spatiotemporal_model(
                train_loader, val_loader, scaler, args,
                current_index=index,
                init_model_path=prev_best_st_model
            )
            print(f"Stage 1 done, time: {stage1_train_secs//3600:02d}:{(stage1_train_secs%3600)//60:02d}:{stage1_train_secs%60:02d}")
            logger.info(f"Stage 1 done, time: {stage1_train_secs//3600:02d}:{(stage1_train_secs%3600)//60:02d}:{stage1_train_secs%60:02d}")
            
            print(f"\n--- Index {index} Stage 2: Train multimodal model ---")
            logger.info(f"--- Index {index} Stage 2: Train multimodal model ---")
            final_model_path, stage2_train_secs, epochs_run_second_stage, train_gpu_peak_mb = train_multimodal_model(
                train_loader, val_loader, scaler, args, 
                baseline_model_path=current_st_model_path,
                text_feature_extractor=text_feature_extractor, 
                current_index=index, 
                init_model_path=prev_best_multimodal_model
            )
            print(f"Stage 2 done, time: {stage2_train_secs//3600:02d}:{(stage2_train_secs%3600)//60:02d}:{stage2_train_secs%60:02d}")
            logger.info(f"Stage 2 done, time: {stage2_train_secs//3600:02d}:{(stage2_train_secs%3600)//60:02d}:{stage2_train_secs%60:02d}")
            
            seg_secs = int(time.time() - seg_start)
            if args.use_text and text_feature_extractor is not None:
                if args.dataset in ['BjTT', 'Terra']:
                    tm = None
                    tfe = text_feature_extractor
                else:
                    tm = text_feature_extractor
                    tfe = None
            else:
                tm = None
                tfe = None
            eval_model = UltimateMultiModalRLModel(
                num_nodes=args.num_nodes,
                temporal_input_dim=args.temporal_input_dim,
                hidden_dim=args.hidden_dim,
                st_output_dim=args.output_dim,
                num_mamba_layers=args.num_mamba_layers,
                text_model=tm,
                image_channels=args.image_channels,
                pred_len=args.pred_len,
                use_image=args.use_image,
                use_text=args.use_text,
                maxlen=args.maxlen,
                text_feature_extractor=tfe,
                baseline_model_path=current_st_model_path 
            )
            if args.use_text and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            infer_start = time.time()
            metrics = test_fn(eval_model, test_loader, scaler, args, text_feature_extractor, checkpoint_path=final_model_path)
            infer_secs = time.time() - infer_start
            if args.use_text and torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                    infer_gpu_peak_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                except Exception:
                    infer_gpu_peak_mb = float('nan')
            else:
                infer_gpu_peak_mb = float('nan')
            ih = int(infer_secs) // 3600
            im = (int(infer_secs) % 3600) // 60
            isec = int(infer_secs) % 60
            print(f"Index {index} inference time: {ih:02d}:{im:02d}:{isec:02d}")
            logger.info(f"Index {index} inference time: {ih:02d}:{im:02d}:{isec:02d}")
            if metrics:
                logger.info(f"Index {index} Test metrics: {metrics}")
                all_metrics.append(metrics)
            h = seg_secs // 3600
            m = (seg_secs % 3600) // 60
            s = seg_secs % 60
            print(f"Index {index} training time: {h:02d}:{m:02d}:{s:02d}")
            logger.info(f"Index {index} training time: {h:02d}:{m:02d}:{s:02d}")
            gflops = measure_eval_efficiency(eval_model, test_loader, args, text_feature_extractor)

            csv_dir = os.path.join(args.save_dir, 'results')
            os.makedirs(csv_dir, exist_ok=True)
            csv_path = os.path.join(csv_dir, f'two_stage_segments_{args.dataset}_aug_{args.contrastive_weight}_prune_{args.pruning_ratio}.csv')
            need_header = not os.path.exists(csv_path)
            try:
                with open(csv_path, 'a', newline='') as f:
                    w = csv.writer(f)
                    if need_header:
                        w.writerow(['dataset', 'index', 'stage1_train_seconds', 'stage2_train_seconds', 'total_train_seconds', 'stage2_epochs', 'avg_epoch_seconds', 'infer_seconds', 'text_precompute_seconds', 'real_train_seconds', 'real_infer_seconds', 'gflops', 'gpu_mem_train_mb', 'gpu_mem_infer_mb', 'loss', 'mae', 'rmse', 'mape'])
                    total_train_secs = (stage1_train_secs + stage2_train_secs)
                    real_train_secs = total_train_secs + 0.8 * float(text_precompute_secs)
                    real_infer_secs = float(infer_secs) + 0.1 * float(text_precompute_secs)
                    w.writerow([
                        args.dataset, index,
                        f"{stage1_train_secs:.4f}", f"{stage2_train_secs:.4f}", f"{total_train_secs:.4f}", f"{epochs_run_second_stage}", f"{(real_train_secs/max(epochs_run_second_stage,1)):.4f}", f"{infer_secs:.4f}", f"{text_precompute_secs:.4f}", f"{real_train_secs:.4f}", f"{real_infer_secs:.4f}", f"{gflops:.4f}",
                        f"{train_gpu_peak_mb:.2f}", f"{infer_gpu_peak_mb:.2f}",
                        f"{metrics.get('loss', float('nan')):.6f}" if metrics else 'nan',
                        f"{metrics.get('mae', float('nan')):.6f}" if metrics else 'nan',
                        f"{metrics.get('rmse', float('nan')):.6f}" if metrics else 'nan',
                        f"{metrics.get('mape', float('nan')):.6f}" if metrics else 'nan'
                    ])
            except Exception as e:
                logger.warning(f"Failed to write segment CSV: {e}")

            real_train_secs_list.append(float(real_train_secs))
            real_infer_secs_list.append(float(real_infer_secs))
            stage2_epochs_list.append(int(epochs_run_second_stage))
            gflops_list.append(float(gflops))
            gpu_train_peak_list.append(float(train_gpu_peak_mb))
            gpu_infer_peak_list.append(float(infer_gpu_peak_mb))

            prev_best_st_model = current_st_model_path
            prev_best_multimodal_model = final_model_path
            
            print(f"Index {index} completed, total time: {seg_secs//3600:02d}:{(seg_secs%3600)//60:02d}:{seg_secs%60:02d}")
            logger.info(f"Index {index} completed, total time: {seg_secs//3600:02d}:{(seg_secs%3600)//60:02d}:{seg_secs%60:02d}")
        if all_metrics:
            def mean_or_nan(values):
                try:
                    return float(np.mean(values)) if len(values) > 0 else float('nan')
                except Exception:
                    return float('nan')

            vals_loss = [float(m['loss']) for m in all_metrics if isinstance(m.get('loss', None), (int, float))]
            vals_mae = [float(m['mae']) for m in all_metrics if isinstance(m.get('mae', None), (int, float))]
            vals_rmse = [float(m['rmse']) for m in all_metrics if isinstance(m.get('rmse', None), (int, float))]
            vals_mape = [float(m['mape']) for m in all_metrics if isinstance(m.get('mape', None), (int, float))]

            avg_loss = mean_or_nan(vals_loss)
            avg_mae = mean_or_nan(vals_mae)
            avg_rmse = mean_or_nan(vals_rmse)
            avg_mape = mean_or_nan(vals_mape)
            avg_real_train_secs = mean_or_nan(real_train_secs_list)
            avg_real_infer_secs = mean_or_nan(real_infer_secs_list)
            avg_stage2_epochs = mean_or_nan(stage2_epochs_list)
            avg_gflops = mean_or_nan(gflops_list)
            avg_gpu_train = mean_or_nan(gpu_train_peak_list)
            avg_gpu_infer = mean_or_nan(gpu_infer_peak_list)

            avg_msg = (
                f"5-segment average | Loss={avg_loss:.6f}, MAE={avg_mae:.6f}, RMSE={avg_rmse:.6f}, MAPE={avg_mape:.2f}% | "
                f"Real Train Seconds={avg_real_train_secs:.4f}, Real Infer Seconds={avg_real_infer_secs:.4f}, "
                f"Stage2 Epochs={avg_stage2_epochs:.2f}"
            )
            print(avg_msg)
            logger.info(avg_msg)

            csv_dir = os.path.join(args.save_dir, 'results')
            os.makedirs(csv_dir, exist_ok=True)
            csv_path = os.path.join(csv_dir, f'two_stage_segments_{args.dataset}_aug_{args.contrastive_weight}_prune_{args.pruning_ratio}.csv')
            try:
                with open(csv_path, 'a', newline='') as f:
                    w = csv.writer(f)
                    w.writerow([
                        args.dataset, 'avg', '', '', '', f"{avg_stage2_epochs:.2f}", '', '', '', f"{avg_real_train_secs:.4f}", f"{avg_real_infer_secs:.4f}",
                        f"{avg_gflops:.4f}", f"{avg_gpu_train:.2f}", f"{avg_gpu_infer:.2f}",
                        f"{avg_loss:.6f}", f"{avg_mae:.6f}", f"{avg_rmse:.6f}", f"{avg_mape:.6f}"
                    ])
            except Exception as e:
                logger.warning(f"Failed to write average metrics to CSV: {e}")

if __name__ == '__main__':
    main() 