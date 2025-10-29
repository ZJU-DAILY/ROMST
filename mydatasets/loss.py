#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
class InverseNormalizer:

    def __init__(self, data_min, data_max):
        self.data_min = data_min
        self.data_max = data_max

    def inverse_normalize_tensor(self, normalized_tensor):
        return normalized_tensor * (self.data_max - self.data_min) + self.data_min

def masked_mape(y_true, y_pred, eps=1e-5):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + eps))) * 100

def get_loss(predictions, targets, inverse_normalizer):
 
    if inverse_normalizer is not None:
        pred_denorm = inverse_normalizer.inverse_normalize_tensor(predictions)
        y_denorm = inverse_normalizer.inverse_normalize_tensor(targets)
    else:
        pred_denorm = predictions
        y_denorm = targets

    pred_np = pred_denorm.detach().cpu().numpy()
    y_np = y_denorm.detach().cpu().numpy()
    
    pred_flat = pred_np.flatten()
    y_flat = y_np.flatten()
    
    mae = mean_absolute_error(y_flat, pred_flat)
    rmse = np.sqrt(mean_squared_error(y_flat, pred_flat))
    mape = masked_mape(y_flat, pred_flat)

    return mae, rmse, mape
