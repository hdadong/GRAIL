#!/usr/bin/env python3
"""Run teacher-policy eval videos under several visual domain-randomization profiles."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


DEFAULT_TEACHER = (
    "/root/GRAIL/imports/SONIC/logs_rl/GRAB_Tracking/"
    "mixed_robocasa_pickup_table_2_g1_ha_pnp_table_2gpu-20260701_153746/last.pt"
)

STUDENT_EGO_CAMERA_OVERRIDES = [
    "++manager_env.config.cameras.camera_resolution=[108,192]",
    "++manager_env.config.cameras.camera_attached_link=torso_link",
    "++manager_env.config.cameras.camera_pos_offset=[0.0576235,0.05253,0.41987]",
    "++manager_env.config.cameras.camera_rot_offset=[0.9024433930625543,0.0,0.43080837368602093,0.0]",
    "++manager_env.config.cameras.camera_yaw_only=True",
    "++manager_env.config.cameras.camera_focal_length=1.88",
    "++manager_env.config.cameras.camera_focus_distance=0.5",
    "++manager_env.config.cameras.camera_horizontal_aperture=2.6035",
    "++manager_env.config.cameras.camera_vertical_aperture=1.4621",
    "++manager_env.config.cameras.camera_clipping_range=[0.1,20.0]",
]

DEFAULT_PROFILES = ["warm_wood", "cool_tile", "marble_bright", "dark_lab", "green_studio"]


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None, log_path: Path | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    if log_path is None:
        subprocess.run(cmd, cwd=cwd, env=env, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        subprocess.run(cmd, cwd=cwd, env=env, stdout=log_file, stderr=subprocess.STDOUT, check=True)


def transcode(src: Path, dst: Path, width: int, height: int, frames: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=25,format=yuv420p"
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-an",
            "-vf",
            vf,
            "-frames:v",
            str(frames),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(dst),
        ]
    )


def hstack(videos: list[Path], dst: Path) -> None:
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for video in videos:
        cmd.extend(["-i", str(video)])
    cmd.extend(["-filter_complex", f"hstack=inputs={len(videos)}[v]", "-map", "[v]", "-an", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(dst)])
    run(cmd)


def ffprobe(path: Path) -> dict[str, object]:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    return json.loads(out)["streams"][0]


def run_teacher_eval(args: argparse.Namespace, profile: str, profile_dir: Path) -> tuple[Path, Path]:
    ego_video = profile_dir / "ego_rgb_raw.mp4"
    third_dir = profile_dir / "third_person"
    third_video = third_dir / "000000.mp4"
    if args.skip_existing and ego_video.exists() and third_video.exists():
        return ego_video, third_video

    third_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-u",
        "gear_sonic/eval_agent_trl.py",
        f"+checkpoint={args.checkpoint}",
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
        *STUDENT_EGO_CAMERA_OVERRIDES,
        "++manager_env.config.render_results=True",
        f"++manager_env.config.save_rendering_dir={third_dir}",
        "++manager_env.config.max_render_envs=1",
        f"++manager_env.config.render_frame_skip={args.third_person_frame_skip}",
        "++manager_env.config.env_spacing=10.0",
        "++manager_env.config.fix_camera_after_first_frame=True",
        "++manager_env.recorders.render_envs._target_=gear_sonic.envs.manager_env.mdp.recorders.RenderEnvsRecorderCfg",
        f"++manager_env.recorders.render_envs.video_save_path={third_dir}",
        "++manager_env.recorders.render_envs.video_quality=5",
        "++manager_env.commands.motion.motion_lib_cfg.multi_thread=False",
        f"++visual_domain_randomization.profile={profile}",
    ]
    env = os.environ.copy()
    env["LOGURU_LEVEL"] = args.loguru_level
    run(cmd, cwd=Path(args.sonic_root), env=env, log_path=profile_dir / "teacher_eval.log")
    return ego_video, third_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_TEACHER)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sonic-root", default="/root/GRAIL/imports/SONIC")
    parser.add_argument("--profiles", nargs="+", default=DEFAULT_PROFILES)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--third-person-frame-skip", type=int, default=2)
    parser.add_argument("--preview-width", type=int, default=1232)
    parser.add_argument("--preview-height", type=int, default=704)
    parser.add_argument("--preview-frames", type=int, default=93)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--loguru-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    previews_dir = out_dir / "previews"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    ego_previews = []
    third_previews = []
    for profile in args.profiles:
        profile_dir = out_dir / profile
        profile_dir.mkdir(parents=True, exist_ok=True)
        ego_video, third_video = run_teacher_eval(args, profile, profile_dir)
        ego_preview = previews_dir / f"ego_{profile}_{args.preview_width}x{args.preview_height}_{args.preview_frames}f.mp4"
        third_preview = previews_dir / f"third_{profile}_{args.preview_width}x{args.preview_height}_{args.preview_frames}f.mp4"
        transcode(ego_video, ego_preview, args.preview_width, args.preview_height, args.preview_frames)
        transcode(third_video, third_preview, args.preview_width, args.preview_height, args.preview_frames)
        ego_previews.append(ego_preview)
        third_previews.append(third_preview)
        records.append(
            {
                "profile": profile,
                "ego_raw": str(ego_video),
                "third_raw": str(third_video),
                "ego_preview": str(ego_preview),
                "third_preview": str(third_preview),
                "ego_preview_info": ffprobe(ego_preview),
                "third_preview_info": ffprobe(third_preview),
            }
        )

    if len(ego_previews) > 1:
        hstack(ego_previews, previews_dir / "ego_visual_dr_5styles_hstack.mp4")
        hstack(third_previews, previews_dir / "third_visual_dr_5styles_hstack.mp4")

    manifest = {
        "checkpoint": args.checkpoint,
        "profiles": args.profiles,
        "records": records,
        "hstack_ego": str(previews_dir / "ego_visual_dr_5styles_hstack.mp4"),
        "hstack_third": str(previews_dir / "third_visual_dr_5styles_hstack.mp4"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
