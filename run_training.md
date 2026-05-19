# GR1 Pick-Place Pipeline

All commands need `OPENBLAS_NUM_THREADS=1` prefix on this machine.

## Step 1: Generate dataset with Mimic (1000 demos from 5 annotated)

```bash
OPENBLAS_NUM_THREADS=1 ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py --device cpu --headless --num_envs 20 --generation_num_trials 1000 --input_file ./datasets/dataset_annotated_gr1.hdf5 --output_file ./datasets/generated_dataset_gr1.hdf5
```

## Step 2: Train BC policy on generated dataset

```bash
OPENBLAS_NUM_THREADS=1 ./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py --task Isaac-PickPlace-GR1T2-Abs-v0 --algo bc --normalize_training_actions --dataset ./datasets/generated_dataset_gr1.hdf5
```

## Step 3: Play (visualize trained policy)

```bash
OPENBLAS_NUM_THREADS=1 ./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py --task Isaac-PickPlace-GR1T2-Abs-v0 --visualizer kit --device cpu --num_rollouts 50 --horizon 400 --norm_factor_min -1.7409999370574951 --norm_factor_max 1.6014130115509033 --checkpoint logs/robomimic/Isaac-PickPlace-GR1T2-Abs-v0/bc_rnn_low_dim_gr1t2/20260331184955/models/model_epoch_500.pth
```

## Available Checkpoints (current run: 20260331184955, trained on generated dataset)

All at `logs/robomimic/Isaac-PickPlace-GR1T2-Abs-v0/bc_rnn_low_dim_gr1t2/20260331184955/models/`:

- model_epoch_100.pth through model_epoch_600.pth (training still running)
