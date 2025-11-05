# ROMST
This is a PyTorch implementation of the paper: Online Multi-Modal Spatio-Temporal Prediction: A Reinforcement Learning and Dynamic Contrastive Framework(ROMST)
![framewwork](./image/framework.pdf)

## Requirements & Configuration

The code is tested with Python 3.10+ and PyTorch (CUDA recommended).

```bash
pip install -r requirements.txt
```

Logging is automatically saved to `checkpoint/log/` with timestamped filenames.

## Data Preparation

The script supports multiple datasets. Prepare the following directories/files.

1) BjTT
- Structure:
  - `mydatasets/BjTT/data/{1,2,3}/*.npy`
  - `mydatasets/BjTT/text/{1,2,3}/*.txt`
- Prior matrix: `mydatasets/prior_matrix/BjTT_matrix_prior.npy`

2) Terra
- Time series: `mydatasets/Terra/time_series/wind_daily.npy`
- Images: `mydatasets/Terra/image/relief_{lat}{N|S}_{lon}{E|W}.png`
- Texts: `mydatasets/Terra/texts/meta_{lat}{N|S}_{lon}{E|W}.txt`
- Prior matrix: `mydatasets/prior_matrix/Terra_matrix_prior.npy`

3) PEMS04
- Time series: `mydatasets/PEMS04/data.npy` 
- Prior matrix: `mydatasets/prior_matrix/PEMS04_matrix_prior.npy`

4) GreenEarthNet
- Temporal: `mydatasets/GreenEarthNet/time_series.npy`
- Image RGB sequence: `mydatasets/GreenEarthNet/image_rgb.npy`
- Prior matrix: `mydatasets/prior_matrix/GreenEarthNet_matrix_prior.npy`

Create folders as needed:
```bash
mkdir -p mydatasets/prior_matrix
mkdir -p mydatasets/BjTT
mkdir -p mydatasets/Terra/{time_series,image,texts}
mkdir -p mydatasets/PEMS04
mkdir -p mydatasets/GreenEarthNet
```
Note: Due to the excessive size of certain datasets, only partial datasets are provided herein. The complete datasets can be found on the official website
- BjTT: https://github.com/ChyaZhang/BjTT
- Terra: https://github.com/CityMind-Lab/NeurIPS24-Terra
- PEMS04: https://github.com/Davidham3/ASTGCN
- GreenEarthNet: https://github.com/vitusbenson/greenearthnet

## Model Training

Run the two-stage pipeline with `train.py`. Key flags are shown below. Adjust paths if your data is elsewhere.

- BjTT
```bash
python train.py \
  --dataset BjTT \
  --data_dir mydatasets/BjTT \
  --prior_matrix_path mydatasets/prior_matrix \
  --seq_len 12 --pred_len 12 \
  --num_nodes 1260 \
  --use_text True --use_image False\
  --pruning_ratio 0.3 --contrastive_weight 0.0001 \
  --batch_size 32 --epochs 100 \
  --device cuda --save_dir ./checkpoint
```

- Terra
```bash
python train.py \
  --dataset Terra \
  --data_dir mydatasets/Terra \
  --time_series_path mydatasets/Terra/time_series/wind_daily.npy \
  --image_dir mydatasets/Terra/image \
  --text_dir mydatasets/Terra/texts \
  --prior_matrix_path mydatasets/prior_matrix \
  --seq_len 12 --pred_len 12 \
  --num_nodes 100 \
  --use_text True --use_image True \
  --pruning_ratio 0.3 --contrastive_weight 0.6 \
  --batch_size 16 --epochs 100 \
  --device cuda --save_dir ./checkpoint
```

- PEMS04
```bash
python train.py \
  --dataset PEMS04 \
  --data_dir mydatasets/PEMS04 \
  --prior_matrix_path mydatasets/prior_matrix \
  --seq_len 12 --pred_len 12 \
  --num_nodes 307 \
  --use_image False --use_text False \
  --contrastive_weight 1 \
  --batch_size 32 --epochs 100 \
  --device cuda --save_dir ./checkpoint
```

- GreenEarthNet
```bash
python train.py \
  --dataset GreenEarthNet \
  --data_dir mydatasets/GreenEarthNet \
  --prior_matrix_path mydatasets/prior_matrix \
  --seq_len 12 --pred_len 12 \
  --num_nodes 1024 \
  --use_image True --use_text False \
  --contrastive_weight 0.001 \
  --batch_size 32 --epochs 100 \
  --device cuda --save_dir ./checkpoint
```

Notes
- The script prunes the text encoder once at startup when `--use_text` is enabled: `--pruning_ratio [0.0–1.0]`.
- Continual learning segments are controlled internally; you can change `index` via `--index` and rerun if needed.


## Outputs

- Logs: `checkpoint/log/training_log_{dataset}_p{pruning_ratio}_c{contrastive_weight}_{YYYYMMDD_HHMMSS}.log`
- Best checkpoints: `checkpoint/{dataset}/best_spatiotemporal_model_*.pth` and `checkpoint/{dataset}/best_multimodal_model_*.pth`
- Segment summary: `checkpoint/results/two_stage_segments_{dataset}_aug_{contrastive_weight}_prune_{pruning_ratio}.csv`
- Optional visualizations (enable with `--enable_reward_viz`):
  - Rewards: `checkpoint/reward/reward_trend_{dataset}_*.csv|.png`
  - Modal weights: `checkpoint/weights/weights_trend_{dataset}_*.csv|.png|.txt`

