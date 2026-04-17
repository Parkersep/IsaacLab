# Finetuning GR00T on G1 Locomanipulation

End-to-end walkthrough: from the teleop recording you already captured in `QUEST_TELEOP_SETUP.md` to a finetuned GR00T policy running in Isaac Lab.

**Assumes you have:**
- Working teleop session producing `./datasets/dataset_g1_locomanip.hdf5` (see [`QUEST_TELEOP_SETUP.md`](./QUEST_TELEOP_SETUP.md))
- GR00T **N1.6** already cloned and installed in a Python env (details in Step 6)
- This repo (`IsaacLab3`) with the `env_isaaclab` venv activated

## Pipeline at a glance

```
1. record_demos.py          → dataset_g1_locomanip.hdf5                     (state-only, controllers)
2. annotate_demos.py        → dataset_annotated_g1_locomanip.hdf5           (state-only, Mimic subtask labels)
3. generate_dataset.py      → generated_dataset_g1_locomanip.hdf5           (state-only, ~1000 demos via Mimic)
4. sdg/generate_data.py     → generated_dataset_g1_locomanipulation_sdg.hdf5 (VISUOMOTOR: adds cameras + navigation)
5. sdg/gr00t/convert_dataset.py → datasets_*_lerobot/                        (LeRobot parquet + videos + meta)
6. (Isaac-GR00T) install + copy data_config.py
7. (Isaac-GR00T) gr00t_finetune.py                                          (writes checkpoints)
8. (IsaacLab)    sdg/gr00t/rollout_policy.py                                (visualize trained policy)
```

**Steps 1–3 are state-only.** Cameras are added in Step 4 — that's what makes the dataset visuomotor and usable by GR00T.

---

## Step 1: Record demonstrations

Already covered in [`QUEST_TELEOP_SETUP.md`](./QUEST_TELEOP_SETUP.md) → "Recording demonstrations" under Step 5. Recap:

```bash
./isaaclab.sh -p scripts/tools/record_demos.py \
    --device cpu \
    --xr \
    --visualizer kit \
    --task Isaac-PickPlace-Locomanipulation-G1-Abs-v0 \
    --dataset_file ./datasets/dataset_g1_locomanip.hdf5 \
    --num_demos 5
```

Reminders:
- **Use the Quest Touch controllers**, not optical hand tracking — the G1 locomanip retargeting pipeline is hardcoded for controllers.
- Collect at least 5 good demos; more is better. Quality > quantity.

## Step 2: Annotate demonstrations

Tag subtask boundaries. Already in [`QUEST_TELEOP_SETUP.md`](./QUEST_TELEOP_SETUP.md) → "Annotating recorded demos for Mimic":

```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
    --device cpu \
    --visualizer kit \
    --task Isaac-Locomanipulation-G1-Abs-Mimic-v0 \
    --input_file ./datasets/dataset_g1_locomanip.hdf5 \
    --output_file ./datasets/dataset_annotated_g1_locomanip.hdf5
```

## Step 3: Mimic-generate 1000 demos (state-only)

Use the annotated handful to bootstrap a larger dataset:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --device cpu \
    --visualizer none \
    --num_envs 20 \
    --generation_num_trials 1000 \
    --input_file ./datasets/dataset_annotated_g1_locomanip.hdf5 \
    --output_file ./datasets/generated_dataset_g1_locomanip.hdf5
```

> **Notes on the env-var prefix and flags:**
> - `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` — forces single-threaded BLAS. Without this, Kit's telemetry plugin forks a subprocess, and the atfork handler crashes in `blas_thread_shutdown_` (multi-threaded BLAS + `fork()` is not safe). If you'd rather not retype these, `export` them in your `~/.zshrc` once and drop the prefix.
> - `--visualizer none` — Isaac Lab 3.0 deprecated `--headless`. Use `--visualizer none` (or omit `--visualizer` — headless is the default). Passing the old `--headless` can trigger a different startup crash.
> - **Memory:** `--num_envs 20` on CPU with a humanoid can blow through 30 GB RAM + 8 GB swap. Drop to `--num_envs 5` or `10` if you see swap thrash.
> - If you see `File already exists in database: grpc/health/v1/health.proto`, also prefix with `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (usually non-fatal, just noisy).

**This step is still state-only** — no `--enable_cameras`, no camera images in the HDF5. Don't worry: cameras come in Step 4.

Expected: ~65–82% success over 1000 trials, ~18–40 min on a decent GPU. Bump `--num_envs` on a powerful machine.

**Shortcut:** if you'd rather skip Steps 1–3 and use the pre-recorded annotated dataset, download `dataset_annotated_g1_locomanip.hdf5` from the Isaac Lab "Annotated G1 Dataset" link, place it under `./datasets/`, and jump to Step 3.

## Step 4: Add cameras + navigation (visuomotor SDG step)

This is the step that converts state-only Mimic output into a visuomotor dataset GR00T can consume. It runs the manipulation data through a scene that adds point-to-point navigation and renders cameras.

```bash
./isaaclab.sh -p scripts/imitation_learning/locomanipulation_sdg/generate_data.py \
    --device cpu \
    --kit_args="--enable isaacsim.replicator.mobility_gen" \
    --task="Isaac-G1-SteeringWheel-Locomanipulation" \
    --dataset ./datasets/generated_dataset_g1_locomanip.hdf5 \
    --num_runs 1 \
    --lift_step 60 \
    --navigate_step 130 \
    --output_file ./datasets/generated_dataset_g1_locomanipulation_sdg.hdf5 \
    --enable_cameras \
    --randomize_placement \
    --visualizer kit
```

Key flags:
- `--enable_cameras` — **required** for GR00T; this is what renders the ego-view camera.
- `--randomize_placement` — randomizes obstacle and fixture positions across runs (recommended for generalization).
- `--lift_step 60` — timestep immediately after the robot has grasped the object.
- `--navigate_step 130` — timestep where the robot has lifted the object and is ready to walk.
- `--kit_args="--enable isaacsim.replicator.mobility_gen"` — enables the MobilityGen extension, required by the SDG scene.

**Optional:** use a NuRec photorealistic background by adding:
```bash
--background_usd_path <PATH>/stage.usdz \
--background_occupancy_yaml_file <PATH>/occupancy_map.yaml \
--high_res_video
```

**Verify it's visuomotor** — quick check after the script finishes:
```bash
./isaaclab.sh -p -c "import h5py; f=h5py.File('./datasets/generated_dataset_g1_locomanipulation_sdg.hdf5','r'); d=f['data']; ep=list(d.keys())[0]; print('keys:', list(d[ep].keys())); print('obs:', list(d[ep].get('obs',{}).keys()) if 'obs' in d[ep] else 'n/a')"
```
You should see camera/video-related keys. The `convert_dataset.py` script expects `locomanipulation_sdg_output_data/object_pose` among others.

**Sanity-check the navigation paths — plot one PNG per demo:**

```bash
./isaaclab.sh -p scripts/imitation_learning/locomanipulation_sdg/plot_navigation_trajectory.py \
    --input_file ./datasets/generated_dataset_g1_locomanipulation_sdg.hdf5 \
    --output_dir ./datasets/plots/nav_trajectories
```

Creates `./datasets/plots/nav_trajectories/demo_<n>.png` for every episode in the HDF5, showing the planned path, robot poses along the path, and obstacle positions. Open one with `xdg-open ./datasets/plots/nav_trajectories/demo_0.png` to confirm the paths look reasonable (no zero-length paths, no paths cutting through obstacles). Accepts an optional `--demo_filter <name>` to restrict to a specific demo.

## Step 5: Convert to LeRobot format

GR00T N1.5/N1.6 both expect data in LeRobot (GNx) format. The converter takes a **directory** of HDF5 files (not a single file) so you can batch multiple runs together.

```bash
mkdir -p ./datasets/sdg_input
mv ./datasets/generated_dataset_g1_locomanipulation_sdg.hdf5 ./datasets/sdg_input/

./isaaclab.sh -p scripts/imitation_learning/locomanipulation_sdg/gr00t/convert_dataset.py \
    ./datasets/sdg_input \
    ./datasets/datasets_train_lerobot
```

Note: the args are **positional** (`input_dir` then `output_path`) — no `--` flags. Episodes with very low object displacement are skipped automatically.

After it runs, `./datasets/datasets_train_lerobot/` will contain `meta/`, `data/` (parquet), and `videos/` subdirectories. `convert_dataset.py` also writes `meta/modality.json`, `info.json`, `episodes.jsonl`, and `tasks.jsonl` — so the LeRobot output is complete and ready to upload as-is (no manual `modality.json` copy needed, unlike the SO100 reference flow).

### Step 5b (optional): Push dataset to HuggingFace Hub

If you're training on a remote machine (vast.ai, RunPod, Lambda, etc.), pushing the dataset to HF Hub once is much better than scp-ing it to every rented instance. LeRobot format is natively HuggingFace-compatible, so this is a single command.

**Run: Local (Isaac Lab workstation)**
```bash
# one-time auth (if you haven't)
huggingface-cli login

# upload under the SensoriRobotics org as a private dataset
huggingface-cli upload \
    --repo-type dataset \
    --private \
    SensoriRobotics/g1_locomanipulation_sdg \
    ./datasets/datasets_train_lerobot
```

(Swap `SensoriRobotics` for `SensoriDev` if you'd rather push to your personal namespace. The org namespace is preferred for shared team datasets.)

First upload sends everything; subsequent uploads send only deltas (git-lfs under the hood). Verify:

```bash
huggingface-cli download --repo-type dataset SensoriRobotics/g1_locomanipulation_sdg \
    --include "meta/*" --local-dir /tmp/verify_hf && cat /tmp/verify_hf/meta/info.json
```

**Size expectations** for ~1000 SDG episodes: `videos/` 10–50 GB (dominant), `data/` <1 GB, `meta/` KB. HF Hub handles this comfortably — no need to worry about quota.

**Pull on a cloud training instance:**

```bash
# on the remote instance, after huggingface-cli login
# use --include to grab only the run you want (e.g. run2)
hf download \
    --repo-type dataset \
    SensoriRobotics/g1_locomanipulation_sdg \
    --include "run2/*" \
    --local-dir /workspace/datasets/g1_locomanipulation_sdg
```

Then pass `--dataset-path /workspace/datasets/g1_locomanipulation_sdg` to `gr00t_finetune.py` in Step 7.

See also: `/home/parker/VLA_MODEL/Groot_projects/Isaac-GR00T/Notes/cloud_training_vastai.md` for the full vast.ai deployment workflow that wraps around this.

## Step 6: Wire up GR00T N1.6

You said you already have Isaac-GR00T cloned and installed. The **one critical step** is copying this repo's data config into the GR00T tree:

```bash
# adjust paths to match where your Isaac-GR00T clone lives
cp /home/parker/Nvidia/IsaacLab3/scripts/imitation_learning/locomanipulation_sdg/gr00t/data_config.py \
   /path/to/Isaac-GR00T/gr00t/experiment/data_config.py
```

This overwrites GR00T's stock `data_config.py` with one that defines `G1LocomanipulationSDGDataConfig` and registers it in `DATA_CONFIG_MAP` under the key `"g1_locomanipulation_sdg"`. Both the finetune script and Isaac Lab's rollout script look up this key.

### N1.5 vs N1.6 caveat

The Isaac Lab docs were written for **N1.5** (`git clone -b n1.5-release ...`). Since you're on **N1.6**:

- The data config is **probably still compatible** — it only depends on public `gr00t.data.*` and `gr00t.model.transforms.GR00TTransform` APIs. If any of those names moved or changed signatures in N1.6, you'll get a quick `ImportError` on first run of Step 7 or 8. Fix by updating the imports at the top of your copied `data_config.py`.
- The finetune CLI (`scripts/gr00t_finetune.py`) may have added/renamed/removed args in N1.6. The command in Step 7 is the N1.5 shape — run `python scripts/gr00t_finetune.py --help` in your N1.6 clone first to verify.
- The policy wrapper `Gr00tPolicy` (imported by `policy.py` in Step 8) may have a different constructor signature in N1.6. If the rollout script errors out on policy init, read the traceback and adjust `scripts/imitation_learning/locomanipulation_sdg/gr00t/policy.py` accordingly.

These three files are the only N1.5↔N1.6 surface; everything else in this pipeline is inside Isaac Lab and is version-agnostic.

## Step 7: Finetune GR00T N1.6

From the `Isaac-GR00T` directory, using **your GR00T env** (not `env_isaaclab`):

### Step 7a: Copy the modality config to the remote

The finetune script needs the `data_config.py` from this repo as its modality config. Copy it to the remote training machine first:

```bash
# from your local machine — create the target dir and scp the file
ssh root@<REMOTE>  "mkdir -p /workspace/gr00t/examples/G1-SDG"
scp /home/parker/Nvidia/IsaacLab3/scripts/imitation_learning/locomanipulation_sdg/gr00t/data_config.py \
    root@<REMOTE>:/workspace/gr00t/examples/G1-SDG/g1_sdg_config.py
```

### Step 7b: Launch finetuning

```bash
cd /workspace/gr00t

python gr00t/experiment/launch_finetune.py \
    --base_model_path nvidia/GR00T-N1.6-3B \
    --dataset_path /workspace/datasets/g1_locomanipulation_sdg \
    --embodiment_tag NEW_EMBODIMENT \
    --modality_config_path examples/G1-SDG/g1_sdg_config.py \
    --output_dir /tmp/g1_finetune \
    --num_gpus 1 \
    --max_steps 20000 \
    --save_steps 5000 \
    --save_total_limit 5 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --global_batch_size 64 \
    --dataloader_num_workers 4 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --use_wandb
```

Confirm each flag with `--help` first if you're unsure whether N1.6 renamed any.

Monitor with:
```bash
tensorboard --logdir ./checkpoints/g1_locomanip
```

## Step 8: Roll out the finetuned policy in Isaac Lab

Back to `env_isaaclab` for this one. The rollout script loads GR00T, so your GR00T install must be importable from `env_isaaclab`'s Python (i.e. GR00T must be installed into `env_isaaclab`, or you need to use your GR00T env with Isaac Lab's sim). Per the Isaac Lab docs' original guidance, GR00T gets installed into the same uv env as Isaac Lab:

```bash
# Install gr00t into the IsaacLab venv
cd /path/to/Isaac-GR00T
uv pip install --python /home/parker/Nvidia/IsaacLab3/env_isaaclab/bin/python -e .

# Install the CUDA toolkit (required for flash-attn and deepspeed)
sudo apt install nvidia-cuda-toolkit

# flash-attn is required by the Eagle backbone — it will not run without it
uv pip install --python /home/parker/Nvidia/IsaacLab3/env_isaaclab/bin/python \
    flash-attn --no-build-isolation
```

Copy the processor files into the checkpoint directory (the training script puts them in a sibling `processor/` dir, but `Gr00tPolicy` expects both model weights and processor at the same path):

```bash
cp /path/to/Isaac-GR00T/checkpoints/g1_locomanip/processor/* \
   /path/to/Isaac-GR00T/checkpoints/g1_locomanip/checkpoint-4000/
```

Then run:

```bash
cd /home/parker/Nvidia/IsaacLab3

./isaaclab.sh -p scripts/imitation_learning/locomanipulation_sdg/gr00t/rollout_policy.py \
    --model_path /path/to/Isaac-GR00T/checkpoints/g1_locomanip/checkpoint-4000 \
    --embodiment_tag new_embodiment \
    --dataset /home/parker/Nvidia/IsaacLab3/datasets/sdg_input/generated_dataset_g1_locomanipulation_sdg.hdf5 \
    --demo demo_0 \
    --output_file ./datasets/rollout_output.hdf5 \
    --task Isaac-G1-SteeringWheel-Locomanipulation \
    --device cpu \
    --enable_cameras \
    --visualizer kit \
    --policy_quat_format wxyz
```

> **`--model_path`** must point to a **specific checkpoint subdirectory** (e.g. `checkpoint-4000`), not the parent training output dir. `Gr00tPolicy` uses `AutoModel.from_pretrained()` which expects model files directly in that directory.

> **`--policy_quat_format wxyz` is required for checkpoints trained with this pipeline.** `convert_dataset.py` stores poses with WXYZ quaternions (`scalar_first=True`), but the Isaac Lab env provides XYZW. Without this flag the rollout sends unconverted XYZW quats to a policy trained on WXYZ, and applies the model's WXYZ action outputs directly to the env without converting back — causing the arms to appear inverted or move erratically.

Options worth knowing:
- `--randomize_placement` — if your checkpoint trained on randomized scenes.

## Troubleshooting

| Problem | Likely cause / fix |
|---|---|
| `ImportError` loading `gr00t.*` during Step 7/8 | N1.6 moved/renamed something. Check your copied `data_config.py` imports and `policy.py` against N1.6's current API. |
| `convert_dataset.py` skips all episodes | Check `get_total_object_displacement` — episodes with tiny xy movement are filtered. Inspect your source HDF5's `locomanipulation_sdg_output_data/object_pose`. |
| No `video.ego_view` key in LeRobot output | Step 4 was run without `--enable_cameras`. Re-run Step 4. |
| Rollout can't find `policy` module | Run from IsaacLab root using `./isaaclab.sh -p` — the script's own dir is added to `sys.path` automatically. |
| `CUDA OOM` during finetune | Lower batch size in `gr00t_finetune.py` args, or reduce `action_horizon` in `data_config.py`. |
| Arms appear inverted / mirrored at rollout | Missing `--policy_quat_format wxyz`. `convert_dataset.py` trains on WXYZ quats; the env provides XYZW. Without conversion the policy misinterprets EEF orientations, producing inverted arm motion. |
| Policy produces wild actions at rollout | Verify `--policy_quat_format wxyz` is set (see above). If still erratic, check train/test normalization stats match (LeRobot `meta/` dir). |

## Expected timing (reference: RTX ADA 6000)

| Step | Duration |
|---|---|
| 3. Mimic generate 1000 demos (state-only) | 18–40 min |
| 4. SDG with cameras | several hours (disk-heavy, renders video) |
| 5. Convert to LeRobot | minutes |
| 7. GR00T finetune (10k steps) | depends on GPU + batch size |
| 8. Rollout (50 episodes) | minutes |

Your hardware (RTX 5070 Ti per `QUEST_TELEOP_SETUP.md` prereqs) should land in the same ballpark for generation, and finetuning will be bottlenecked by VRAM + step count.

## References

- Updated develop docs: https://isaac-sim.github.io/IsaacLab/develop/source/overview/imitation-learning/mimic_humanoid_demos.html
- Isaac-GR00T (N1.6 branch): use whichever branch/tag you installed
- LeRobot (GNx) format spec: https://github.com/huggingface/lerobot
