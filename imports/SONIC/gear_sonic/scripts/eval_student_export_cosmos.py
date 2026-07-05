#!/usr/bin/env python3
"""Evaluate an ego-vision student policy and package Cosmos-Transfer inputs."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys


DEFAULT_PROMPT = (
    "A realistic first-person humanoid robot manipulation video in an indoor room. "
    "The camera is mounted on the robot torso, looking toward the robot hands and "
    "the manipulated object. Preserve the robot motion and object geometry while "
    "randomizing the background, lighting, and materials into a realistic scene."
)


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def ffprobe_video(path: Path) -> dict[str, str]:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,duration,r_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    data = json.loads(out)
    return data["streams"][0]


def transcode_for_cosmos(
    src: Path,
    dst: Path,
    fps: int,
    resolution: int,
    max_frames: int,
    exact_frames: bool,
    scale_flags: str = "lanczos",
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    video_info = ffprobe_video(src)
    filter_fps = fps
    if exact_frames and max_frames > 0:
        duration = float(video_info.get("duration") or 0)
        if duration <= 0:
            raise RuntimeError(f"Cannot infer duration for exact-frame export: {src}")
        filter_fps = max_frames / duration
    vf_parts = [
        f"fps={filter_fps}",
        f"scale=-2:{resolution}:flags={scale_flags}",
        "format=yuv420p",
    ]
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-an",
    ]
    if max_frames > 0:
        cmd += ["-frames:v", str(max_frames)]
    cmd += [
        "-vf",
        ",".join(vf_parts),
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(dst),
    ]
    run(cmd)


def write_edge_video(src: Path, dst: Path) -> None:
    import cv2

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 16
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(dst),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 80, 160)
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            writer.write(edges_bgr)
    finally:
        cap.release()
        writer.release()

    # Re-encode for broad compatibility with Cosmos/imageio.
    tmp = dst.with_suffix(".tmp.mp4")
    dst.rename(tmp)
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(tmp),
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(dst),
        ]
    )
    tmp.unlink(missing_ok=True)


def write_synthetic_guided_mask_video(src: Path, dst: Path, mode: str) -> None:
    """Write a synthetic guided mask for debugging only.

    Real Cosmos/IsaacSim pipelines should pass a rendered or segmented foreground
    mask via --guided-mask-video instead of relying on these geometric fallbacks.
    """

    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 16
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(dst),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    try:
        while True:
            ok, _frame = cap.read()
            if not ok:
                break
            mask = np.zeros((height, width), dtype=np.uint8)
            if mode == "full":
                mask[:, :] = 255
            elif mode == "lower_center":
                center = (width // 2, int(height * 0.68))
                axes = (int(width * 0.46), int(height * 0.34))
                cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
                mask[int(height * 0.55) :, :] = np.maximum(mask[int(height * 0.55) :, :], 255)
            else:
                raise ValueError(f"Unknown mask mode: {mode}")
            writer.write(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
    finally:
        cap.release()
        writer.release()

    tmp = dst.with_suffix(".tmp.mp4")
    dst.rename(tmp)
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(tmp),
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(dst),
        ]
    )
    tmp.unlink(missing_ok=True)


def make_cosmos_package(
    *,
    ego_video: Path,
    third_person_video: Path | None,
    out_dir: Path,
    cosmos_root: Path,
    prompt: str,
    fps: int,
    resolution: int,
    max_frames: int,
    exact_frames: bool,
    mask_mode: str,
    guided_mask_video: Path | None,
    guided_generation_step_threshold: int,
    guided_generation_foreground_labels: list[int] | None,
    edge_control_video: Path | None,
    depth_control_video: Path | None,
    seg_control_video: Path | None,
    vis_control_video: Path | None,
    edge_control_weight: float,
    depth_control_weight: float,
    seg_control_weight: float,
    vis_control_weight: float,
    cosmos_num_steps: int,
    cosmos_model: str,
    cosmos_nproc: int,
    cosmos_master_port: int,
    name: str,
) -> None:
    input_video = out_dir / "input" / "ego_rgb.mp4"
    edge_video = out_dir / "control" / "edge.mp4"
    depth_video = out_dir / "control" / "depth.mp4"
    seg_video = out_dir / "control" / "seg.mp4"
    vis_video = out_dir / "control" / "vis.mp4"
    mask_video = out_dir / "masks" / "guided_generation_mask.mp4"
    prompt_path = out_dir / "prompt.txt"
    spec_path = out_dir / "spec_cosmos.json"
    run_path = out_dir / "run_cosmos.sh"
    manifest_path = out_dir / "manifest.json"

    transcode_for_cosmos(
        ego_video,
        input_video,
        fps=fps,
        resolution=resolution,
        max_frames=max_frames,
        exact_frames=exact_frames,
    )

    if edge_control_video is not None:
        transcode_for_cosmos(
            edge_control_video,
            edge_video,
            fps=fps,
            resolution=resolution,
            max_frames=max_frames,
            exact_frames=exact_frames,
        )
    else:
        write_edge_video(input_video, edge_video)

    controls: dict[str, dict[str, str | float]] = {
        "edge": {
            "control_path": str(edge_video),
            "control_weight": edge_control_weight,
        }
    }

    if depth_control_video is not None:
        transcode_for_cosmos(
            depth_control_video,
            depth_video,
            fps=fps,
            resolution=resolution,
            max_frames=max_frames,
            exact_frames=exact_frames,
        )
        controls["depth"] = {"control_path": str(depth_video), "control_weight": depth_control_weight}

    if seg_control_video is not None:
        transcode_for_cosmos(
            seg_control_video,
            seg_video,
            fps=fps,
            resolution=resolution,
            max_frames=max_frames,
            exact_frames=exact_frames,
            scale_flags="neighbor",
        )
        controls["seg"] = {"control_path": str(seg_video), "control_weight": seg_control_weight}

    if vis_control_video is not None:
        transcode_for_cosmos(
            vis_control_video,
            vis_video,
            fps=fps,
            resolution=resolution,
            max_frames=max_frames,
            exact_frames=exact_frames,
        )
        controls["vis"] = {"control_path": str(vis_video), "control_weight": vis_control_weight}

    guided_mask_path: Path | None = None
    guided_mask_source: str | None = None
    if guided_mask_video is not None:
        guided_mask_path = mask_video
        guided_mask_source = str(guided_mask_video)
        transcode_for_cosmos(
            guided_mask_video,
            guided_mask_path,
            fps=fps,
            resolution=resolution,
            max_frames=max_frames,
            exact_frames=exact_frames,
            scale_flags="neighbor",
        )
    elif mask_mode != "none":
        guided_mask_path = mask_video
        guided_mask_source = f"synthetic:{mask_mode}"
        write_synthetic_guided_mask_video(input_video, guided_mask_path, mode=mask_mode)

    if guided_generation_foreground_labels is not None and guided_mask_path is None:
        raise ValueError("--guided-generation-foreground-labels requires --guided-mask-video or --mask-mode")

    prompt_path.write_text(prompt + "\n", encoding="utf-8")

    spec = {
        "name": name,
        "prompt_path": str(prompt_path),
        "video_path": str(input_video),
        "guidance": 3,
        "num_steps": cosmos_num_steps,
        "num_video_frames_per_chunk": 93,
        "max_frames": max_frames if max_frames > 0 else None,
        "keep_input_resolution": True,
        **controls,
    }
    if guided_mask_path is not None:
        spec["guided_generation_mask"] = str(guided_mask_path)
        spec["guided_generation_step_threshold"] = guided_generation_step_threshold
        if guided_generation_foreground_labels is not None:
            spec["guided_generation_foreground_labels"] = guided_generation_foreground_labels
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")

    run_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"cd {cosmos_root}",
                (
                    f"torchrun --nproc_per_node={cosmos_nproc} --master_port={cosmos_master_port} "
                    f"examples/inference.py -i {spec_path} "
                    f"-o {out_dir / 'cosmos_outputs'} --model={cosmos_model} --disable-guardrails"
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    run_path.chmod(0o755)

    manifest = {
        "ego_video": str(ego_video),
        "third_person_video": str(third_person_video) if third_person_video else None,
        "cosmos_root": str(cosmos_root),
        "cosmos_input_video": str(input_video),
        "cosmos_controls": controls,
        "cosmos_guided_mask": str(guided_mask_path) if guided_mask_path else None,
        "cosmos_guided_mask_source": guided_mask_source,
        "cosmos_spec": str(spec_path),
        "cosmos_run_script": str(run_path),
        "cosmos_model": cosmos_model,
        "cosmos_nproc": cosmos_nproc,
        "cosmos_num_steps": cosmos_num_steps,
        "cosmos_exact_frames": exact_frames,
        "input_video_info": ffprobe_video(input_video),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def latest_student_checkpoint() -> Path:
    root = Path("/root/GRAIL/imports/SONIC/logs_rl/GRAB_Tracking/manager/universal_token/distill")
    candidates = sorted(root.glob("robocasa_ego_distill*/last.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No student last.pt found under {root}")
    return candidates[-1]


def run_student_eval(args: argparse.Namespace, out_dir: Path) -> tuple[Path, Path]:
    ckpt_src = Path(args.checkpoint) if args.checkpoint else latest_student_checkpoint()
    snapshot_dir = out_dir / "snapshot"
    third_dir = out_dir / "third_person"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    third_dir.mkdir(parents=True, exist_ok=True)
    ckpt = snapshot_dir / "student_last.pt"
    shutil.copy2(ckpt_src, ckpt)
    config_src = ckpt_src.parent / "config.yaml"
    if config_src.exists():
        shutil.copy2(config_src, snapshot_dir / "config.yaml")

    ego_video = out_dir / "ego_rgb_raw.mp4"
    cmd = [
        sys.executable,
        "-u",
        "gear_sonic/eval_agent_trl.py",
        f"+checkpoint={ckpt}",
        "+headless=True",
        "++use_wandb=false",
        "++num_envs=1",
        "++run_eval_loop=True",
        "++run_once=True",
        f"++max_render_steps={args.eval_steps}",
        f"++capture_camera_rgb_path={ego_video}",
        "++capture_camera_rgb_source=raw",
        f"++capture_camera_rgb_max_frames={args.eval_steps}",
        "++manager_env.config.enable_cameras=True",
        "++manager_env.config.render_results=True",
        f"++manager_env.config.save_rendering_dir={third_dir}",
        "++manager_env.config.max_render_envs=1",
        f"++manager_env.config.render_frame_skip={args.third_person_frame_skip}",
        "++manager_env.config.env_spacing=10.0",
        "++manager_env.recorders.render_envs._target_=gear_sonic.envs.manager_env.mdp.recorders.RenderEnvsRecorderCfg",
        f"++manager_env.recorders.render_envs.video_save_path={third_dir}",
        "++manager_env.recorders.render_envs.video_quality=5",
        "++manager_env.commands.motion.motion_lib_cfg.multi_thread=False",
    ]
    env = os.environ.copy()
    env["LOGURU_LEVEL"] = args.loguru_level
    sonic_root = Path(args.sonic_root)
    run(cmd, cwd=sonic_root, env=env)

    third_person = third_dir / "000000.mp4"
    return ego_video, third_person


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help="Student checkpoint. Defaults to latest robocasa ego-distill last.pt.")
    parser.add_argument("--output-dir", required=True, help="Directory for eval videos and Cosmos input package.")
    parser.add_argument("--sonic-root", default="/root/GRAIL/imports/SONIC")
    parser.add_argument("--cosmos-root", default="/physis/cosmos-transfer2.5")
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--third-person-frame-skip", type=int, default=2)
    parser.add_argument("--skip-eval", action="store_true", help="Only package an existing --ego-video.")
    parser.add_argument("--ego-video", default=None, help="Existing ego video used with --skip-eval.")
    parser.add_argument("--third-person-video", default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--cosmos-fps", type=int, default=16)
    parser.add_argument("--cosmos-resolution", type=int, default=720)
    parser.add_argument("--cosmos-max-frames", type=int, default=93)
    parser.add_argument(
        "--cosmos-exact-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resample the whole ego video to exactly --cosmos-max-frames frames.",
    )
    parser.add_argument(
        "--mask-mode",
        choices=["none", "full", "lower_center"],
        default="none",
        help=(
            "Synthetic guided mask mode. Defaults to none. Use --guided-mask-video "
            "for real IsaacSim/SAM/segmentation masks; lower_center is a debug fallback only."
        ),
    )
    parser.add_argument(
        "--guided-mask-video",
        default=None,
        help="Precomputed binary/label guided-generation mask video from IsaacSim, SAM2, or render IDs.",
    )
    parser.add_argument(
        "--guided-generation-step-threshold",
        type=int,
        default=25,
        help="Cosmos guided generation denoising step threshold, used only when a guided mask is supplied.",
    )
    parser.add_argument(
        "--guided-generation-foreground-labels",
        type=int,
        nargs="*",
        default=None,
        help="Optional foreground label ids for a label-valued guided mask.",
    )
    parser.add_argument("--edge-control-video", default=None, help="Optional precomputed edge control video.")
    parser.add_argument("--depth-control-video", default=None, help="Optional precomputed depth control video.")
    parser.add_argument("--seg-control-video", default=None, help="Optional precomputed segmentation control video.")
    parser.add_argument("--vis-control-video", default=None, help="Optional precomputed RGB/blur/vis control video.")
    parser.add_argument("--edge-control-weight", type=float, default=1.0)
    parser.add_argument("--depth-control-weight", type=float, default=1.0)
    parser.add_argument("--seg-control-weight", type=float, default=1.0)
    parser.add_argument("--vis-control-weight", type=float, default=1.0)
    parser.add_argument(
        "--cosmos-num-steps",
        type=int,
        default=35,
        help="Cosmos sampling steps. Use 35 for the general edge/depth/seg/vis models; 4 is only for distilled checkpoints.",
    )
    parser.add_argument("--cosmos-model", default="edge", choices=["edge", "depth", "seg", "vis"])
    parser.add_argument("--cosmos-nproc", type=int, default=4)
    parser.add_argument("--cosmos-master-port", type=int, default=12344)
    parser.add_argument("--name", default="grail_student_ego_edge_guided")
    parser.add_argument("--loguru-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_eval:
        if args.ego_video is None:
            raise ValueError("--ego-video is required with --skip-eval")
        ego_video = Path(args.ego_video)
        third_person = Path(args.third_person_video) if args.third_person_video else None
    else:
        ego_video, third_person = run_student_eval(args, out_dir)

    make_cosmos_package(
        ego_video=ego_video,
        third_person_video=third_person,
        out_dir=out_dir / "cosmos_transfer_inputs",
        cosmos_root=Path(args.cosmos_root),
        prompt=args.prompt,
        fps=args.cosmos_fps,
        resolution=args.cosmos_resolution,
        max_frames=args.cosmos_max_frames,
        exact_frames=args.cosmos_exact_frames,
        mask_mode=args.mask_mode,
        guided_mask_video=Path(args.guided_mask_video) if args.guided_mask_video else None,
        guided_generation_step_threshold=args.guided_generation_step_threshold,
        guided_generation_foreground_labels=args.guided_generation_foreground_labels,
        edge_control_video=Path(args.edge_control_video) if args.edge_control_video else None,
        depth_control_video=Path(args.depth_control_video) if args.depth_control_video else None,
        seg_control_video=Path(args.seg_control_video) if args.seg_control_video else None,
        vis_control_video=Path(args.vis_control_video) if args.vis_control_video else None,
        edge_control_weight=args.edge_control_weight,
        depth_control_weight=args.depth_control_weight,
        seg_control_weight=args.seg_control_weight,
        vis_control_weight=args.vis_control_weight,
        cosmos_num_steps=args.cosmos_num_steps,
        cosmos_model=args.cosmos_model,
        cosmos_nproc=args.cosmos_nproc,
        cosmos_master_port=args.cosmos_master_port,
        name=args.name,
    )
    print(f"Wrote eval/Cosmos package under {out_dir}")


if __name__ == "__main__":
    main()
