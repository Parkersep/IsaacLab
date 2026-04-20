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

## Setup

One-time setup for the end-to-end pipeline. Steps 1–3 run locally; Step 4 (the heavy SDG render pass) can run on a rented GPU via the datagen Docker image built here. Steps 7–8 have their own env (your Isaac-GR00T clone) — see Step 6.

### 0.1 Local prerequisites

- **Quest headset + controllers** (Step 1 only — teleop is the one step that can't be moved to the cloud).
- **Isaac Lab 3 `env_isaaclab` venv activated.** If you haven't bootstrapped it yet: `./isaaclab.sh --install`.
- **Docker** (only needed if you want to run Step 4 remotely). Confirm with `docker info`.
- **HuggingFace CLI.** `pip install "huggingface_hub[cli]>=0.34,<1.0"` then `hf auth login` (token needs `write` scope for dataset pushes). Datasets live under the `SensoriRobotics` org. The `<1.0` upper bound avoids a `huggingface_hub` 1.x that breaks `transformers <5` during GR00T finetune.
- **vast.ai CLI** (only for remote Step 4). `pipx install vastai && vastai set api-key <KEY>`.

### 0.2 Build and push the datagen Docker image (one-time, optional)

Skip this if you'll run Step 4 locally. Otherwise, bake the develop-branch checkout on top of NVIDIA's pre-built `nvcr.io/nvidia/isaac-lab:3.0.0-beta1` so any rented GPU can pull the exact code that runs on your workstation:

**Run: Local**
```bash
cd /home/parker/Nvidia/IsaacLab3

# build (~30 GB image; base is pulled from NGC, public)
docker build -t par4ker/isaaclab3-datagen:develop \
    -f docker/Dockerfile.datagen .

# push to Docker Hub
docker login
docker push par4ker/isaaclab3-datagen:develop
```

After the first push, confirm the repo is public at `hub.docker.com/r/par4ker/isaaclab3-datagen` → **Settings** → **Visibility: Public**. vast.ai then pulls anonymously.

### 0.3 Quick remote setup (automated)

Once the image is on Docker Hub, the wrapper script creates a vast.ai instance, waits for SSH, and logs into HF for you:

**Run: Local**
```bash
bash scripts/imitation_learning/locomanipulation_sdg/vast_ai_datagen_setup.sh \
    --offer-id <OFFER_ID> \
    --hf-token <HF_TOKEN>

# Optional flags:
#   --disk <GB>     disk size in GB (default: 200 — 2× expected HDF5 per 1000 demos)
#   --image <IMG>   Docker image (default: par4ker/isaaclab3-datagen:develop)
```

Find `<OFFER_ID>` with `vastai search offers 'gpu_name=RTX_4090 num_gpus=1 disk_space>=200 inet_down>=200' -o 'dph+'`. Once the script finishes, SSH in and jump straight to Step 2 (annotate) or Step 4 (SDG) pointing paths at `/workspace/datasets`.

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

### Step 1b (optional): Push recorded demos to HuggingFace Hub

If Steps 2–5 will run on a rented GPU (vast.ai, RunPod, Lambda, etc.), pushing the teleop HDF5 to HF Hub once is much simpler than scp-ing it to every instance. Teleop is the only step that has to stay on your workstation; everything after can pull this file from HF.

**Run: Local (Isaac Lab workstation)**
```bash
# one-time auth (if you haven't)
hf auth login

# upload under the SensoriRobotics org as a private dataset
hf upload \
    --repo-type dataset \
    --private \
    SensoriRobotics/g1_locomanipulation_teleop \
    ./datasets/dataset_g1_locomanip.hdf5 \
    dataset_g1_locomanip.hdf5
```

The third positional arg (`dataset_g1_locomanip.hdf5`) is the path **inside the repo** — pin it so appending more recordings later (e.g. `dataset_g1_locomanip_run2.hdf5`) doesn't overwrite prior uploads.

(Swap `SensoriRobotics` for `SensoriDev` if you'd rather push to your personal namespace. The org namespace is preferred for shared team datasets.)

**Pull on a cloud instance** (before Step 2):

```bash
# on the remote instance, after `hf auth login`
hf download \
    --repo-type dataset \
    SensoriRobotics/g1_locomanipulation_teleop \
    --include "mimic_dataset_g1_locomanip.hdf5" \
    --local-dir /workspace/datasets/teleop
```

Then point `--input_file` at `/workspace/datasets/teleop/dataset_g1_locomanip.hdf5` in Step 2.

**Size expectations:** a raw teleop HDF5 from ~5–10 demos is typically 10–200 MB (state-only, no camera frames yet), so the upload is fast — no LFS concerns at this stage.

## Step 2: Annotate demonstrations

Tag subtask boundaries. Already in [`QUEST_TELEOP_SETUP.md`](./QUEST_TELEOP_SETUP.md) → "Annotating recorded demos for Mimic":

**Run: Local**
```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
    --device cpu \
    --visualizer kit \
    --task Isaac-Locomanipulation-G1-Abs-Mimic-v0 \
    --input_file ./datasets/dataset_g1_locomanip.hdf5 \
    --output_file ./datasets/dataset_annotated_g1_locomanip.hdf5
```

**Run: Remote (vast.ai datagen instance)**

Assumes you pulled the raw teleop HDF5 from HF into `/workspace/datasets/teleop/` (Step 1b). Output stays on the mounted volume so it survives instance teardown:

```bash
cd /workspace/IsaacLab

./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
    --device cpu \
    --visualizer none \
    --task Isaac-Locomanipulation-G1-Abs-Mimic-v0 \
    --input_file /workspace/datasets/teleop/dataset_g1_locomanip.hdf5 \
    --output_file /workspace/datasets/dataset_annotated_g1_locomanip.hdf5
```

Note: if annotation requires GUI interaction in your workflow, do this step locally before uploading, and pull the annotated HDF5 on the remote (`SensoriRobotics/g1_locomanipulation_teleop` already hosts `mimic_dataset_g1_locomanip.hdf5` which skips Steps 2–3 entirely).

## Step 3: Mimic-generate 1000 demos (state-only)

Use the annotated handful to bootstrap a larger dataset:

**Run: Local**
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

**Run: Remote (vast.ai datagen instance)**

The env-var prefix is baked into the datagen image (`ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ...`), so you can drop it. Bump `--num_envs` if the instance has plenty of RAM — an RTX 4090 host typically handles 40+ without swap:

```bash
cd /workspace/IsaacLab

./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --device cpu \
    --visualizer none \
    --num_envs 40 \
    --generation_num_trials 1000 \
    --input_file /workspace/datasets/dataset_annotated_g1_locomanip.hdf5 \
    --output_file /workspace/datasets/generated_dataset_g1_locomanip.hdf5
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

**Run: Local**
```bash
./isaaclab.sh -p scripts/imitation_learning/locomanipulation_sdg/generate_data.py \
    --device cpu \
    --kit_args="--enable isaacsim.replicator.mobility_gen" \
    --task="Isaac-G1-SteeringWheel-Locomanipulation" \
    --dataset ./datasets/generated_dataset_g1_locomanip.hdf5 \
    --num_runs 1 \
    --lift_step 60 \
    --navigate_step 119 \
    --output_file ./datasets/generated_dataset_g1_locomanipulation_sdg.hdf5 \
    --enable_cameras \
    --randomize_placement \
    --visualizer kit 
```

**Run: Remote (vast.ai datagen instance)**

Assumes you already pulled the Mimic-generated input from HF into `/workspace/datasets/teleop/` (see Step 1b pull). Use `/workspace/datasets/` for both input and output so the results land on the mounted volume instead of ephemeral container storage:

```bash
cd /workspace/isaaclab

./isaaclab.sh -p scripts/imitation_learning/locomanipulation_sdg/generate_data.py \
    --device cpu \
    --kit_args="--enable isaacsim.replicator.mobility_gen" \
    --task="Isaac-G1-SteeringWheel-Locomanipulation" \
    --dataset /workspace/datasets/teleop/mimic_dataset_g1_locomanip.hdf5 \
    --num_runs 500 \
    --lift_step 60 \
    --navigate_step 119 \
    --output_file /workspace/datasets/vla_dataset_g1_locomanipulation_sdg.hdf5 \
    --enable_cameras \
    --randomize_placement \
    --visualizer none
```

Note `--visualizer none` (lowercase) on the headless remote — `kit` requires a display.

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

**Run: Local**
```bash
mkdir -p ./datasets/sdg_input
mv ./datasets/generated_dataset_g1_locomanipulation_sdg.hdf5 ./datasets/sdg_input/

./isaaclab.sh -p scripts/imitation_learning/locomanipulation_sdg/gr00t/convert_dataset.py \
    ./datasets/sdg_input \
    ./datasets/datasets_train_lerobot
```

**Run: Remote (vast.ai datagen instance)**
```bash
cd /workspace/IsaacLab

mkdir -p /workspace/datasets/sdg_input
mv /workspace/datasets/vla_dataset_g1_locomanipulation_sdg.hdf5 /workspace/datasets/sdg_input/

./isaaclab.sh -p scripts/imitation_learning/locomanipulation_sdg/gr00t/convert_dataset.py \
    /workspace/datasets/sdg_input \
    /workspace/datasets/datasets_train_lerobot
```

Note: the args are **positional** (`input_dir` then `output_path`) — no `--` flags. Episodes with very low object displacement are skipped automatically.

After it runs, `./datasets/datasets_train_lerobot/` will contain `meta/`, `data/` (parquet), and `videos/` subdirectories. `convert_dataset.py` also writes `meta/modality.json`, `info.json`, `episodes.jsonl`, and `tasks.jsonl` — so the LeRobot output is complete and ready to upload as-is (no manual `modality.json` copy needed, unlike the SO100 reference flow).

### Step 5b (optional): Push dataset to HuggingFace Hub

If you're training on a remote machine (vast.ai, RunPod, Lambda, etc.), pushing the dataset to HF Hub once is much better than scp-ing it to every rented instance. LeRobot format is natively HuggingFace-compatible, so this is a single command.

**Run: Local (Isaac Lab workstation)**
```bash
# one-time auth (if you haven't)
hf auth login

# upload under the SensoriRobotics org as a private dataset
hf upload \
    --repo-type dataset \
    --private \
    SensoriRobotics/g1_locomanipulation_sdg \
    ./datasets/datasets_train_lerobot
```

**Run: Remote (vast.ai datagen instance)**

If you ran Step 5 on the remote, push from there — avoids scp-ing 10–50 GB back home:

```bash
# vast.ai datagen instance is already logged in (from Step 0.3 --hf-token); pin
# a run subfolder so subsequent uploads don't overwrite prior runs.
hf upload \
    --repo-type dataset \
    --private \
    SensoriRobotics/g1_locomanipulation_sdg \
    /workspace/datasets/datasets_train_lerobot \
    run3
```

(Swap `SensoriRobotics` for `SensoriDev` if you'd rather push to your personal namespace. The org namespace is preferred for shared team datasets.)

First upload sends everything; subsequent uploads send only deltas (git-lfs under the hood). Verify:

```bash
hf download --repo-type dataset SensoriRobotics/g1_locomanipulation_sdg \
    --include "meta/*" --local-dir /tmp/verify_hf && cat /tmp/verify_hf/meta/info.json
```

**Size expectations** for ~1000 SDG episodes: `videos/` 10–50 GB (dominant), `data/` <1 GB, `meta/` KB. HF Hub handles this comfortably — no need to worry about quota.

**Pull on a cloud training instance:**

```bash
# on the remote instance, after `hf auth login`
# full dataset:
hf download \
    --repo-type dataset \
    SensoriRobotics/g1_locomanipulation_sdg \
    --local-dir /workspace/datasets/g1_locomanipulation_sdg

# only a specific run subfolder (e.g. run2) — use "run2/**" to recurse into
# data/, videos/, and meta/. A single-star glob "run2/*" only matches files
# directly in run2/ and silently skips the subdirectories you actually need.
hf download \
    --repo-type dataset \
    SensoriRobotics/g1_locomanipulation_sdg \
    --include "run2/**" \
    --local-dir /workspace/datasets/g1_locomanipulation_sdg

# multiple runs in one pull — pass several patterns to --include separated by
# spaces. Finetune can then train on the union by pointing --dataset-path at
# the parent dir.
hf download \
    --repo-type dataset \
    SensoriRobotics/g1_locomanipulation_sdg \
    --include "run2/**" "run3/**" \
    --local-dir /workspace/datasets/g1_locomanipulation_sdg
```

Files land under `/workspace/datasets/g1_locomanipulation_sdg/run2/...` since HF preserves repo paths. Point `--dataset-path` at `/workspace/datasets/g1_locomanipulation_sdg/run2` in Step 7 (or use the parent dir for the full-repo / multi-run pull).

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

The N1.6 `launch_finetune.py` takes a **single** `--dataset_path` pointing at one LeRobot dataset root (the dir with `meta/info.json`). Pick the variant below that matches your data layout.

**Single run** — point `--dataset_path` at one LeRobot root:

```bash
cd /workspace/gr00t

python gr00t/experiment/launch_finetune.py \
    --base_model_path nvidia/GR00T-N1.6-3B \
    --dataset_path /workspace/datasets/g1_locomanipulation_sdg/run2 \
    --embodiment_tag NEW_EMBODIMENT \
    --modality_config_path examples/G1-SDG/g1_sdg_config.py \
    --output_dir /tmp/g1_finetune \
    --num_gpus 1 \
    --max_steps 40000 \
    --save_steps 8000 \
    --save_total_limit 5 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --global_batch_size 96 \
    --dataloader_num_workers 4 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --use_wandb
```

**Multiple runs** — use the `launch_finetune_multirun.py` wrapper in this repo, which adds `--extra_dataset_paths` on top of the stock N1.6 CLI. All paths become one `SingleDatasetConfig` and are concatenated proportionally by episode count.

Copy the wrapper to the remote (from your local machine):

```bash
scp /home/parker/Nvidia/IsaacLab3/scripts/imitation_learning/locomanipulation_sdg/gr00t/launch_finetune_multirun.py \
    root@<REMOTE>:/workspace/gr00t/launch_finetune_multirun.py
```

Then on the remote, from `/workspace/gr00t`:

```bash
python launch_finetune_multirun.py \
    --base_model_path nvidia/GR00T-N1.6-3B \
    --dataset_path /workspace/datasets/g1_locomanipulation_sdg/run2 \
    --extra_dataset_paths /workspace/datasets/g1_locomanipulation_sdg/run3 \
    --embodiment_tag NEW_EMBODIMENT \
    --modality_config_path examples/G1-SDG/g1_sdg_config.py \
    --output_dir /tmp/g1_finetune \
    --num_gpus 1 \
    --max_steps 40000 \
    --save_steps 8000 \
    --save_total_limit 5 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --global_batch_size 96 \
    --dataloader_num_workers 4 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --use_wandb
```

`--extra_dataset_paths` accepts any number of paths (e.g. `... run3 run4 run5`), so you can keep adding runs without touching the wrapper.

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
    --randomize_placement \
    --policy_quat_format wxyz
```

> **`--randomize_placement` is required for checkpoints trained with fixture randomization.** Without it, `setup_navigation_scene` skips `place_randomly`, so the drop-off bench stays at its config-default pose (`[-2, -3.55, -0.3]`, −45° yaw) across all resets — which is out-of-distribution for a policy trained on random bench placements and usually produces wrong-direction navigation.

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

## Remote data generation on vast.ai (Docker)

If you want to run Steps 2–5 on a rented GPU (to produce 1k+ demos without tying up your workstation), the datagen Docker image built in **Setup 0.2** carries your develop-branch source into any vast.ai instance. Teleop (Step 1) still has to happen locally because it needs the Quest hardware — everything after is headless-capable.

### Option A: Automated (script)

Use the wrapper from Setup 0.3 — one command creates the instance, waits for SSH, and logs into HF:

```bash
bash scripts/imitation_learning/locomanipulation_sdg/vast_ai_datagen_setup.sh \
    --offer-id <OFFER_ID> \
    --hf-token <HF_TOKEN> \
    --disk 200
```

### Option B: Manual

1. Launch a vast.ai instance with image `par4ker/isaaclab3-datagen:develop`, a GPU (RTX 4090 or better for Step 4), and a volume mounted at `/workspace/datasets` (size ~2× the expected HDF5 per the timing table above — e.g. 200 GB for 1000 demos).
2. SSH in and set up auth + data:
   ```bash
   hf auth login  # paste a token with repo write
   hf download --repo-type dataset \
       SensoriRobotics/g1_locomanipulation_teleop \
       --local-dir /workspace/datasets/teleop
   ```
3. Run Steps 2 → 5 as documented above, pointing paths at `/workspace/datasets`.
4. Push the generated LeRobot output back to HF:
   ```bash
   hf upload --repo-type dataset --private \
       SensoriRobotics/g1_locomanipulation_sdg \
       /workspace/datasets/datasets_train_lerobot_run3 run3
   ```

Image size is ~30 GB (Isaac Sim base is ~20 GB on its own); first pull on vast.ai takes 5–10 min on a fast instance.

## References

- Updated develop docs: https://isaac-sim.github.io/IsaacLab/develop/source/overview/imitation-learning/mimic_humanoid_demos.html
- Isaac-GR00T (N1.6 branch): use whichever branch/tag you installed
- LeRobot (GNx) format spec: https://github.com/huggingface/lerobot
