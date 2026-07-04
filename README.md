<div align="center">

# GRAIL: Generating Humanoid Loco-Manipulation from 3D Assets and Video Priors

<p>
  <a href="https://research.nvidia.com/labs/dair/grail/">
    <img src="https://img.shields.io/badge/Project-Page-blue?style=flat-square" alt="Project Page"/>
  </a>
  <a href="https://arxiv.org/pdf/2606.05160">
    <img src="https://img.shields.io/badge/Paper-PDF-red?style=flat-square" alt="Paper"/>
  </a>
  <a href="https://nvlabs.github.io/GRAIL/">
    <img src="https://img.shields.io/badge/Docs-online-success?style=flat-square" alt="Docs"/>
  </a>
  <a href="https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL">
    <img src="https://img.shields.io/badge/Dataset-HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=000" alt="Dataset"/>
  </a>
</p>

<img src="assets/videos/teaser.gif" alt="GRAIL teaser" width="100%"/>

</div>

**GRAIL** is a fully digital data-generation pipeline for humanoid loco-manipulation. It composes 3D assets, simulator-ready scenes, robot-proportioned characters, and video foundation model priors to synthesize metric 4D human-object interaction (HOI) trajectories, then retargets them to a Unitree G1 and trains task-general policies for pick-up, whole-body manipulation, sitting, and terrain traversal. Using only GRAIL-generated data, the resulting egocentric visual policies transfer to real-world object pick-up and stair-climbing.

## Motion Gallery

| Tabletop Pickup | Ground Pickup |
|:---:|:---:|
| <img src="assets/videos/pickup_table.gif" width="420"/> | <img src="assets/videos/pickup_ground.gif" width="420"/> |

| Tabletop Manipulation | Ground Manipulation |
|:---:|:---:|
| <img src="assets/videos/manip_tabletop.gif" width="420"/> | <img src="assets/videos/manip_large.gif" width="420"/> |

| Sitting | Curb |
|:---:|:---:|
| <img src="assets/videos/sitting.gif" width="420"/> | <img src="assets/videos/terrain_curbs.gif" width="420"/> |

| Slope | Stairs |
|:---:|:---:|
| <img src="assets/videos/terrain_slopes.gif" width="420"/> | <img src="assets/videos/terrain_stairs.gif" width="420"/> |

## Sim-to-Real Deployment

**Rendered Egocentric Views**

<img src="assets/videos/deployment_egocentric_views.gif" width="100%"/>

| Pick-up | Stair-Climbing |
|:---:|:---:|
| <img src="assets/videos/deployment_pickup.gif" width="420"/> | <img src="assets/videos/deployment_stairs.gif" width="420"/> |

## Quick Start

**Pull the docker image** and install local extras inside the bind-mounted checkout:

```bash
git clone https://github.com/NVlabs/GRAIL.git
cd GRAIL
git submodule update --init --recursive

docker pull docker.io/nvgrail/grail:latest

docker run --gpus all -it --shm-size=16g \
    -v "$PWD":/workspace/grail \
    docker.io/nvgrail/grail:latest

# inside the container
cd /workspace/grail
bash scripts/setup/install_env_docker.sh   # validates native extensions, downloads Blender
bash scripts/setup/download_checkpoints.sh # GEM-SMPL / GEM-SOMA / FoundationPose weights
bash scripts/setup/download_comasset.sh --category cordless_drill # quick-start object
source /root/miniconda3/etc/profile.d/conda.sh
conda activate grail
[ -f .env ] && source .env                  # OPENAI_API_KEY, KLING_*, HF_TOKEN
```

The setup script rebuilds GPU/Python-specific native extensions when needed
(`nvdiffrast`, FoundationPose `mycpp`) and installs Blender into the mounted
checkout.

Run any stage end-to-end. Pipeline stages are package entrypoints; invoke them
with `python -m grail.pipelines.*` rather than project-root wrapper scripts.

```bash
# 3D asset generation (procedural terrain or AI-generated objects)
python -m grail.pipelines.gen_terrain --type stairs --num 50 --output_dir data/syn_stairs
conda run -n hunyuan python -m grail.pipelines.gen_3d_assets \
    -i configs/gen_3d/example_objects.yaml -o data/gen_example

# 2D HOI generation (Blender + Kling video)
python -m grail.pipelines.gen_2dhoi --dataset ComAsset --category cordless_drill \
    --character kid --results_dir results --video_model_api kling-ai

# 4D HOI reconstruction
python -m grail.pipelines.recon_4dhoi --dataset ComAsset --category cordless_drill --results_dir results
```

Full install, dataset, and config notes: see [Documentation](#documentation) below.

## Reproducing Asset Prep And Selected-Asset Runs

This section records the operational steps used for the small selected-asset
run in this workspace. It is intended as a copyable recipe for another server.
Do not commit API keys; put provider credentials in `.env` or `.quickstart_env`
and `source` that file before running stages that call OpenAI-compatible or
Kling APIs.

### Large-data storage

If a shared TOS/GPFS path is available, keep large datasets, model caches, and
generated assets there, then symlink them back to the paths expected by GRAIL:

```bash
mkdir -p /physis/bx-workspace/GRAIL_storage/{root_cache,GRAIL/data,tmp_robocasa_probe}

rsync -aH --delete /root/.cache/huggingface/ \
  /physis/bx-workspace/GRAIL_storage/root_cache/huggingface/
mv /root/.cache/huggingface /root/.cache/huggingface.local
ln -s /physis/bx-workspace/GRAIL_storage/root_cache/huggingface /root/.cache/huggingface

rsync -aH --delete /root/.cache/hy3dgen/ \
  /physis/bx-workspace/GRAIL_storage/root_cache/hy3dgen/
mv /root/.cache/hy3dgen /root/.cache/hy3dgen.local
ln -s /physis/bx-workspace/GRAIL_storage/root_cache/hy3dgen /root/.cache/hy3dgen
```

Use the same pattern for cold GRAIL data such as `data/ComAsset`,
`data/Scene`, `data/RoboCasa`, `data/Terrain`, and `data/gen_stairs`. Keep
hot training paths such as `data/motion_lib/` local while training is running.

### Procedural terrain assets

Generate 10 curbs, 10 slopes, and 10 procedural stairs:

```bash
cd /root/GRAIL
conda run -n sonic python -m grail.pipelines.gen_terrain \
  --type all --num 10 --output_dir data/Terrain
```

Expected layout:

```text
data/Terrain/
  curb_000..curb_009/{model.obj,model.mtl,texture.jpg}
  slope_000..slope_009/{model.obj,model.mtl,texture.jpg}
  stairs_000..stairs_009/{model.obj,model.mtl,texture.jpg}
```

For the selected run, use `configs/objects/selected_terrain_4.yaml`:
`curb_000`, `curb_001`, `slope_000`, and `slope_001`.

### Hunyuan generated stair assets

Create a small input list, then generate textured meshes:

```bash
cd /root/GRAIL
conda run -n hunyuan python -m pip install --no-cache-dir open3d==0.18.0

set -a
source .quickstart_env   # OPENAI-compatible endpoint/key, no secrets in git
set +a

conda run -n hunyuan python -u -m grail.pipelines.gen_3d_assets \
  -i configs/gen_3d/gen_stairs_10.yaml \
  -o data/gen_stairs
```

Each generated stair folder should contain `model.obj`, `model.mtl`, and
`model.jpg`. `open3d` is required by the Hunyuan texture/remesh stage; without
it the pipeline may stop after `mesh.glb` / `white_mesh_remesh.obj`.

For the selected run, use `configs/objects/selected_gen_stairs_4.yaml`, which
selects the first four completed Hunyuan stair assets.

### RoboCasa object assets

The small selected run uses official RoboCasa objaverse assets. One practical
way to stage them is:

```bash
cd /root
git clone https://github.com/ARISE-Initiative/robocasa.git tmp_robocasa_probe/robocasa
# Download and extract the official RoboCasa objaverse asset archive into:
# /root/tmp_robocasa_probe/robocasa/robocasa/models/assets/objects/objaverse

cd /root/GRAIL
mkdir -p data/RoboCasa
```

Then symlink each official object instance directory into `data/RoboCasa/`.
Keep any existing locomanip assets under a separate pointer if needed:

```bash
ln -sfn \
  /root/GRAIL/imports/SONIC/decoupled_wbc/dexmg/gr00trobocasa/robocasa/models/assets/objects/omniverse/locomanip \
  data/RoboCasa_locomanip

# Example for one objaverse instance:
ln -sfn \
  /root/tmp_robocasa_probe/robocasa/robocasa/models/assets/objects/objaverse/apple_0 \
  data/RoboCasa/apple_0
```

Validate that the configured assets resolve:

```bash
python - <<'PY'
import os, yaml
with open("configs/objects/robocasa.yaml") as f:
    data = yaml.safe_load(f)
names = list(data["objects"].keys())
ok = sum(os.path.exists(os.path.join("data/RoboCasa", n)) for n in names)
print(f"RoboCasa resolvable: {ok}/{len(names)}")
PY
```

For the selected run, use `configs/objects/selected_robocasa_pickup_table_4.yaml`:
`apple_0`, `banana_1`, `bowl_0`, and `can_0`.

### Selected 12-asset G1 trajectory run

Generate 2D HOI videos:

```bash
cd /root/GRAIL
set -a; source .quickstart_env; set +a

conda run -n sonic python -u -m grail.pipelines.gen_2dhoi \
  --config configs/gen_2dhoi/selected_terrain_4.yaml --skip_done

conda run -n sonic python -u -m grail.pipelines.gen_2dhoi \
  --config configs/gen_2dhoi/selected_gen_stairs_4.yaml --skip_done

conda run -n sonic python -u -m grail.pipelines.gen_2dhoi \
  --config configs/gen_2dhoi/selected_robocasa_pickup_table_4.yaml --skip_done
```

Run 4D reconstruction in the `grail` environment. If the host only has
`sonic`/`hunyuan`, use the provided Docker image and mount both the repo and
shared storage so symlinked datasets remain visible:

```bash
docker run --rm --gpus all --net=host --ipc=host --entrypoint="" \
  -v /root/GRAIL:/workspace/grail \
  -v /physis:/physis \
  -w /workspace/grail docker.io/nvgrail/grail:latest \
  bash -lc 'source /root/miniconda3/etc/profile.d/conda.sh &&
    conda activate grail &&
    set -a && source .quickstart_env && set +a &&
    export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 &&
    python -u -m grail.pipelines.recon_4dhoi \
      --config configs/recon_4dhoi/loco_smplx.yaml \
      --dataset Terrain \
      --results_dir results_collection/selected_terrain_4 \
      --skip_done'
```

Repeat with `--dataset gen_stairs --results_dir results_collection/selected_gen_stairs_4`
for Hunyuan stairs. For tabletop RoboCasa pickup, use
`--config configs/recon_4dhoi/pickup_smplx.yaml --dataset RoboCasa --results_dir results_collection/selected_robocasa_pickup_table_4`.

Retarget to Unitree G1:

```bash
conda run -n sonic python -m grail.pipelines.retarget \
  --data_dir results_collection/selected_terrain_4/generation/4dhoi_recon_smplx_valid \
  --output_folder selected_terrain_4_g1 \
  --no_process --no_bps --zero_out_wrist

conda run -n sonic python -m grail.pipelines.retarget \
  --data_dir results_collection/selected_gen_stairs_4/generation/4dhoi_recon_smplx_valid \
  --output_folder selected_gen_stairs_4_g1 \
  --no_process --no_bps --zero_out_wrist

conda run -n sonic python -m grail.pipelines.retarget \
  --data_dir results_collection/selected_robocasa_pickup_table_4/generation/4dhoi_recon_smplx_valid \
  --output_folder selected_robocasa_pickup_table_4_g1
```

Train with the official SONIC configs. Terrain uses the scene/terrain tracking
config; pickup/manipulation data uses a HOI config:

```bash
cd /root/GRAIL/imports/SONIC
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 WANDB_MODE=online

accelerate launch --num_processes=8 train_agent_trl.py \
  +exp=manager/universal_token/scene/terrain_tracking \
  num_envs=4096 headless=True \
  ++resume=True \
  ++checkpoint=models/terrain_stairs/last.pt \
  experiment_dir=/root/GRAIL/imports/SONIC/logs_rl/GRAB_Tracking/selected_terrain_multitask \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/root/GRAIL/data/motion_lib/selected_terrain_4_g1/robot \
  ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=/root/GRAIL/data/motion_lib/selected_terrain_4_g1/objects \
  ++manager_env.config.terrain_motion_dir=/root/GRAIL/data/motion_lib/selected_terrain_4_g1
```

For multi-object pickup:

```bash
accelerate launch --num_processes=8 train_agent_trl.py \
  +exp=manager/universal_token/hoi/pnp_table \
  num_envs=2048 headless=True \
  ++resume=True \
  ++checkpoint=models/pnp_table/last.pt \
  experiment_dir=/root/GRAIL/imports/SONIC/logs_rl/GRAB_Tracking/selected_robocasa_pickup_table_4 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/root/GRAIL/data/motion_lib/selected_robocasa_pickup_table_4_g1_ha/robot \
  ++manager_env.commands.motion.motion_lib_cfg.object_motion_file=/root/GRAIL/data/motion_lib/selected_robocasa_pickup_table_4_g1_ha/objects \
  ++manager_env.config.object_usd_path=/root/GRAIL/data/motion_lib/selected_robocasa_pickup_table_4_g1_ha/object_usd \
  ++manager_env.commands.motion.motion_lib_cfg.bps_dir=/root/GRAIL/data/motion_lib/selected_robocasa_pickup_table_4_g1/bps
```

## Documentation

Full documentation can be found at [docs](https://nvlabs.github.io/GRAIL/)
(rendered HTML) and markdown sources are linked below.

### Getting Started
- [Installation](docs/source/installation.md)
- [Quick Start](docs/source/quick_start.md)

### Pipeline
- [3D Asset Generation](docs/source/gen_3d_assets.md)
- [2D HOI Generation](docs/source/gen_2dhoi.md)
- [4D HOI Reconstruction](docs/source/recon_4dhoi.md)
- [Retargeting](docs/source/retargeting.md)
- [Task General Tracking](docs/source/tracking.md)
- [Data Export](docs/source/data_export.md)
- [Data Visualization](docs/source/visualization.md)
- [Web Visualizer](docs/source/web_visualizer.md)

## TODOs

- [ ] Provide quick-start demo script
- [ ] Release GRAIL manipulation dataset
- [ ] Release task-general tracking policy checkpoints

## Citation

If you find GRAIL useful in your research, please cite:

```bibtex
@misc{grail2026,
  title         = {GRAIL: Generating Humanoid Loco-Manipulation from 3D Assets and Video Priors},
  author        = {Tianyi Xie and Haotian Zhang and Jinhyung Park and Zi Wang and Bowen Wen and Jiefeng Li and Xueting Li and Qingwei Ben and Haoyang Weng and Yufei Ye and David Minor and Tingwu Wang and Chenfanfu Jiang and Sanja Fidler and Jan Kautz and Linxi Fan and Yuke Zhu and Zhengyi Luo and Umar Iqbal and Ye Yuan},
  year          = {2026},
  eprint        = {2606.05160},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  doi           = {10.48550/arXiv.2606.05160},
  url           = {https://arxiv.org/abs/2606.05160},
}
```

## License

This project is released under the NVIDIA License; see [LICENSE](LICENSE) for details. The Work and any derivative works may be used only non-commercially, except by NVIDIA Corporation and its affiliates. Third-party components are subject to their own licenses.
