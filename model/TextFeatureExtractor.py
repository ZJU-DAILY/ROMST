import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import sentencepiece
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
from datasets import load_dataset
import gc
from collections import OrderedDict
import time
from torch.cuda.amp import autocast

class CustomSynFlowPruner:
    def __init__(self):
        pass

    def get_block_importances(self, model, device):
        import gc
        model.eval()
        grad_sums = []
        synapse_terms = []
        num_blocks = len(model.model.layers)

        @torch.no_grad()
        def linearize(model):
            signs = {}
            for name, param in model.named_parameters():
                if param.requires_grad and param is not None:
                    signs[name] = torch.sign(param)
                    param.abs_()
            return signs

        @torch.no_grad()
        def nonlinearize(model, signs):
            for name, param in model.named_parameters():
                if param.requires_grad and param is not None and name in signs:
                    param.mul_(signs[name])

        signs = linearize(model)

        try:
            seq_len = 8 
            batch_size = 1
            input_ids = torch.ones([batch_size, seq_len], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = logits.sum() / logits.numel()
            print(f"[SynFlow] loss: {loss.item():.6f}")

            for n, layer in enumerate(model.model.layers):
                grad_sum = 0.0
                for name, param in layer.named_parameters():
                    if param.requires_grad and param is not None:
                        grad = torch.autograd.grad(loss, param, retain_graph=True, allow_unused=True)[0]
                        if grad is not None:
                            grad_sum += torch.abs(grad * param).sum().item()
                        del grad
                        torch.cuda.empty_cache()
                        gc.collect()
                log_sum = 0.0
                count = 0
                for l in range(n+1):
                    for name, param in model.model.layers[l].named_parameters():
                        if 'weight' in name and param.dim() >= 2:
                            log_abs = torch.log(param.detach().abs().clamp(min=1e-5).float())
                            log_sum += log_abs.sum().item()
                            count += log_abs.numel()
                synapse_term = np.exp(log_sum / count) if count > 0 else 0.0
                grad_sums.append(grad_sum)
                synapse_terms.append(synapse_term)
            block_scores = {}
            for n in range(num_blocks):
                block_score = grad_sums[n] + synapse_terms[n]

                block_scores[n] = block_score
        finally:
            try:
                nonlinearize(model, signs)
            except:
                pass
            torch.cuda.empty_cache()
            gc.collect()
        return block_scores

class CustomMagnitudePruner:
    def __init__(self, lambda_l2=0.05):
        self.lambda_l2 = lambda_l2

    def calculate_importance(self, weight_matrix, name=None, block_index=None):
        l1_norm = torch.sum(torch.abs(weight_matrix.float()))
        l2_norm_sq = torch.sum(weight_matrix.float() ** 2)
        importance = l1_norm + self.lambda_l2 * l2_norm_sq
        return importance.item()

    def prune_block(self, model, block_index):
        if hasattr(model.model, 'layers') and block_index < len(model.model.layers):
            block = model.model.layers[block_index]
            total_importance = 0
            num_weights = 0

            for name, param in block.named_parameters():
                if 'weight' in name or 'bias' in name:
                    total_importance += self.calculate_importance(param.data, name=name, block_index=block_index)
                    num_weights += param.numel()

            return total_importance / num_weights if num_weights > 0 else 0
        return 0


class TextFeatureExtractor(nn.Module):
    def __init__(self, model_name="/home/zzh/lth/RMCL/llama 1b",
                 device="cuda", use_fp16=False, max_length=966):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16
        self._cache = OrderedDict()
        self._cache_size = 1000
        self.max_length = max_length

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if use_fp16 else torch.float32,
                device_map="cuda:0",
                local_files_only=True,
                trust_remote_code=True,
                use_safetensors=True,
                low_cpu_mem_usage=True
            )

            if hasattr(self.model, "tie_weights"):
                self.model.tie_weights()
        except Exception as e:
            import traceback
            print(f"Failed to load model, detailed error: {traceback.format_exc()}")
            raise RuntimeError("Unable to load model, please check model path and file integrity") from e

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=True,
                trust_remote_code=True,
                use_safetensors=True,
                tokenizer_file=os.path.join(self.model_name, "tokenizer.json")
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        except Exception as e:
            print(f"Failed to load tokenizer: {e}")
            raise RuntimeError("Unable to load tokenizer, please check model path and file integrity") from e

        self.pruner = None
        self.pruned_blocks = []
        self.is_pruned = False
        self.is_retrained = False

    def prune_model(self, pruning_ratio=0.2, method="custom_magnitude", **pruner_kwargs):
        layers = self.model.model.layers
        num_blocks = len(layers)
        num_to_prune = int(num_blocks * pruning_ratio)
        params_before_pruning = sum(p.numel() for p in self.model.parameters())
        if method == "custom_magnitude":
            lambda_l2 = pruner_kwargs.get("lambda_l2", 0.05)
            self.pruner = CustomMagnitudePruner(lambda_l2=lambda_l2)
            block_importances = [(i, self.pruner.prune_block(self.model, i)) for i in range(num_blocks)]
        elif method == "custom_synflow":
            self.pruner = CustomSynFlowPruner()
            block_scores = self.pruner.get_block_importances(self.model, self.device)
            block_importances = list(block_scores.items())
        block_importances.sort(key=lambda x: x[1])
        self.pruned_blocks = [idx for idx, _ in block_importances[:num_to_prune]]
        new_layers = [layer for i, layer in enumerate(layers) if i not in self.pruned_blocks]
        self.model.model.layers = nn.ModuleList(new_layers)
        params_after_pruning = sum(p.numel() for p in self.model.parameters())
        self.compression_ratio = (params_before_pruning - params_after_pruning) / params_before_pruning if params_before_pruning > 0 else 0.0
        self.is_pruned = True
        print(f"Successfully pruned {len(self.pruned_blocks)}/{num_blocks} Transformer blocks")
        print(f"Pruned block indices: {self.pruned_blocks}")
        print(f"Number of parameters: {params_before_pruning:,} -> {params_after_pruning:,}")
        print(f"Actual compression ratio: {self.compression_ratio:.3f}")


    def extract_features(self, text, max_length=None):
        if not isinstance(text, str) or not text.strip():
            return torch.zeros(self.model.config.hidden_size, device=self.model.device)
        
        if max_length is None:
            max_length = self.max_length
        
        model_max_length = getattr(self.model.config, 'max_position_embeddings', 2048)
        max_length = min(max_length, model_max_length)
        
        try:
            encoded = self.encode_text([text], max_length=max_length)
            
            if encoded["input_ids"].shape[0] == 0:
                return torch.zeros(self.model.config.hidden_size, device=self.model.device)
            
            input_ids = encoded["input_ids"].to(self.model.device)
            attention_mask = encoded["attention_mask"].to(self.model.device)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.use_fp16):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        use_cache=False
                    )
            
            hidden_states = outputs.hidden_states[-1]  
            mask = attention_mask.unsqueeze(-1).float()
            masked_hidden = hidden_states * mask
            sum_hidden = masked_hidden.sum(dim=1)
            sum_mask = mask.sum(dim=1).clamp(min=1e-9)
            features = (sum_hidden / sum_mask).squeeze(0)
            
            return features
            
        except Exception as e:
            return torch.zeros(self.model.config.hidden_size, device=self.model.device)
    
    def encode_text(self, text_list, max_length=None):
        if max_length is None:
            max_length = self.max_length
        
        model_max_length = getattr(self.model.config, 'max_position_embeddings', 2048)
        max_length = min(max_length, model_max_length)
        
        text_list = [t for t in text_list if isinstance(t, str) and t.strip()]
        
        if not text_list:
            return {
                'input_ids': torch.zeros((0, max_length), dtype=torch.long),
                'attention_mask': torch.zeros((0, max_length), dtype=torch.long)
            }
        
        encoded = self.tokenizer(
            text_list,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        return encoded
        
    def get_trainable_parameters(self):
        trainable_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append((name, param.numel()))
        return trainable_params
        
    def print_parameter_status(self):
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        
        print(f"Text encoder parameter status:")
        print(f"  Trainable: {trainable:,} ({trainable/total:.2%})")
        print(f"  Frozen: {total-trainable:,} ({(total-trainable)/total:.2%})")
        print(f"  Total: {total:,}")
        
        if trainable > 0:
            print("Trainable parameter details:")
            for name, count in self.get_trainable_parameters()[:10]:
                print(f"  {name}: {count:,}")
            if len(self.get_trainable_parameters()) > 10:
                print(f"  ... and {len(self.get_trainable_parameters())-10} more parameter groups")
    
    def get_feature_dim(self):
        hidden_size = self.model.config.hidden_size
        return hidden_size


