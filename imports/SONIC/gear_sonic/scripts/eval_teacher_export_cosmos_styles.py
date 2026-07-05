#!/usr/bin/env python3
"""Evaluate a teacher actor and prepare Cosmos Transfer style variants.

The script records both the raw ego camera and a third-person render. The
third-person camera is fixed after the first frame so it does not follow the
robot root during the episode.
"""

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

EGO_STYLES = {
    "ego_chinese_home": (
        "A sharp realistic first-person humanoid robot camera video in a warm Chinese family dining room. "
        "The robot picks up an apple from a table. Preserve the robot hands, apple position, table geometry, "
        "camera motion, and timing. Change the room to light wood furniture, cream walls, soft daylight, "
        "simple cabinets, and subtle Chinese home decor. Avoid blur, warped hands, deformed objects, text, or subtitles."
    ),
    "ego_modern_kitchen": (
        "A sharp realistic first-person humanoid robot camera video in a modern apartment kitchen and dining area. "
        "The robot picks up an apple from a table. Preserve the robot hands, apple position, table geometry, "
        "camera motion, and timing. Use white cabinets, stone countertop, pendant lights, polished floor, and natural daylight. "
        "Avoid blur, warped hands, deformed objects, text, or subtitles."
    ),
    "ego_wood_study": (
        "A sharp realistic first-person humanoid robot camera video in a cozy wood-paneled study and dining corner. "
        "The robot picks up an apple from a table. Preserve the robot hands, apple position, table geometry, "
        "camera motion, and timing. Use bookshelves, warm lamps, wooden chairs, muted curtains, and realistic indoor lighting. "
        "Avoid blur, warped hands, deformed objects, text, or subtitles."
    ),
    "ego_lab_cleanroom": (
        "A sharp realistic first-person humanoid robot camera video in a clean robotics lab dining-test area. "
        "The robot picks up an apple from a table. Preserve the robot hands, apple position, table geometry, "
        "camera motion, and timing. Use matte white walls, lab benches, soft overhead panels, and tidy calibration equipment. "
        "Avoid blur, warped hands, deformed objects, text, or subtitles."
    ),
    "ego_sunlit_cafe": (
        "A sharp realistic first-person humanoid robot camera video in a sunlit small cafe interior. "
        "The robot picks up an apple from a table. Preserve the robot hands, apple position, table geometry, "
        "camera motion, and timing. Use warm window light, cafe chairs, plants, ceramic cups, and a clean wooden table. "
        "Avoid blur, warped hands, deformed objects, text, or subtitles."
    ),
}

THIRD_STYLES = {
    "third_chinese_home": (
        "A sharp realistic third-person video of a humanoid robot standing at a dining table and picking up an apple "
        "inside a warm Chinese family dining room. Preserve the robot body, table shape, apple position, pickup motion, "
        "fixed camera angle, timing, and scene geometry. Use light wood furniture, cream walls, soft daylight, simple cabinets, "
        "and subtle Chinese home decor. Avoid blur, warped robot limbs, deformed table edges, duplicate apples, text, or subtitles."
    ),
    "third_modern_kitchen": (
        "A sharp realistic third-person video of a humanoid robot standing at a dining table and picking up an apple "
        "in a modern apartment kitchen and dining area. Preserve the robot body, table shape, apple position, pickup motion, "
        "fixed camera angle, timing, and scene geometry. Use white cabinets, stone countertop, pendant lights, polished floor, "
        "and natural indoor daylight. Avoid blur, warped robot limbs, deformed table edges, duplicate apples, text, or subtitles."
    ),
    "third_wood_study": (
        "A sharp realistic third-person video of a humanoid robot standing at a dining table and picking up an apple "
        "in a cozy wood-paneled study and dining corner. Preserve the robot body, table shape, apple position, pickup motion, "
        "fixed camera angle, timing, and scene geometry. Use bookshelves, warm lamps, wooden chairs, muted curtains, and realistic lighting. "
        "Avoid blur, warped robot limbs, deformed table edges, duplicate apples, text, or subtitles."
    ),
    "third_lab_cleanroom": (
        "A sharp realistic third-person video of a humanoid robot standing at a table and picking up an apple "
        "in a clean robotics lab test area. Preserve the robot body, table shape, apple position, pickup motion, "
        "fixed camera angle, timing, and scene geometry. Use matte white walls, lab benches, soft overhead panels, and tidy calibration equipment. "
        "Avoid blur, warped robot limbs, deformed table edges, duplicate apples, text, or subtitles."
    ),
    "third_sunlit_cafe": (
        "A sharp realistic third-person video of a humanoid robot standing at a cafe table and picking up an apple "
        "in a sunlit small cafe interior. Preserve the robot body, table shape, apple position, pickup motion, "
        "fixed camera angle, timing, and scene geometry. Use warm window light, cafe chairs, plants, ceramic cups, and a clean wooden table. "
        "Avoid blur, warped robot limbs, deformed table edges, duplicate apples, text, or subtitles."
    ),
}


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    log_path: Path | None = None,
) -> None:
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


def ffprobe(path: Path) -> dict[str, str]:
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


def run_teacher_eval(args: argparse.Namespace, out_dir: Path) -> tuple[Path, Path]:
    ego_video = out_dir / "ego_rgb_raw.mp4"
    third_dir = out_dir / "third_person"
    camera_outputs_dir = out_dir / "ego_camera_outputs"
    camera_raw_dir = out_dir / "ego_camera_raw"
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
        "++manager_env.recorders.render_envs._target_=gear_sonic.envs.manager_env.mdp.recorders.RenderEnvsRecorderCfg",
        f"++manager_env.recorders.render_envs.video_save_path={third_dir}",
        "++manager_env.recorders.render_envs.video_quality=5",
        "++manager_env.commands.motion.motion_lib_cfg.multi_thread=False",
    ]
    if args.fix_third_person_camera:
        cmd.append("++manager_env.config.fix_camera_after_first_frame=True")
    if args.capture_isaac_camera_outputs:
        cmd.extend(
            [
                f"++capture_camera_output_dir={camera_outputs_dir}",
                f"++capture_camera_output_raw_dir={camera_raw_dir}",
                f"++capture_camera_output_max_frames={args.capture_output_frames}",
                f"++capture_camera_output_raw_max_frames={args.capture_raw_frames}",
                "++capture_camera_output_types=[semantic_segmentation,instance_segmentation_fast,instance_id_segmentation_fast,distance_to_image_plane]",
                "++manager_env.config.cameras.camera_data_types=[rgb,semantic_segmentation,instance_segmentation_fast,instance_id_segmentation_fast,distance_to_image_plane]",
            ]
        )
    env = os.environ.copy()
    env["LOGURU_LEVEL"] = args.loguru_level
    run(cmd, cwd=Path(args.sonic_root), env=env, log_path=out_dir / "teacher_eval.log")
    return ego_video, third_dir / "000000.mp4"


def write_cosmos_specs(args: argparse.Namespace, out_dir: Path, ego_video: Path, third_video: Path) -> Path:
    cosmos_dir = out_dir / "cosmos_transfer_inputs"
    input_dir = cosmos_dir / "input"
    prompt_dir = cosmos_dir / "prompts"
    input_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    ego_input = input_dir / "teacher_ego_1232x704_93f.mp4"
    third_input = input_dir / "teacher_third_fixed_1232x704_93f.mp4"
    transcode(ego_video, ego_input, args.cosmos_width, args.cosmos_height, args.cosmos_frames)
    transcode(third_video, third_input, args.cosmos_width, args.cosmos_height, args.cosmos_frames)

    samples: list[dict[str, object]] = []
    for style_name, prompt in EGO_STYLES.items():
        prompt_path = prompt_dir / f"{style_name}.txt"
        prompt_path.write_text(prompt + "\n", encoding="utf-8")
        samples.append(
            {
                "name": f"teacher_{style_name}_seg_no_depth",
                "prompt_path": str(prompt_path),
                "video_path": str(ego_input),
                "guidance": args.guidance,
                "num_steps": args.cosmos_steps,
                "num_video_frames_per_chunk": args.cosmos_frames,
                "max_frames": args.cosmos_frames,
                "keep_input_resolution": True,
                "seg": {"control_prompt": "robot hand . gripper . apple . dining table", "control_weight": 1.0},
            }
        )
    for style_name, prompt in THIRD_STYLES.items():
        prompt_path = prompt_dir / f"{style_name}.txt"
        prompt_path.write_text(prompt + "\n", encoding="utf-8")
        samples.append(
            {
                "name": f"teacher_{style_name}_seg_no_depth",
                "prompt_path": str(prompt_path),
                "video_path": str(third_input),
                "guidance": args.guidance,
                "num_steps": args.cosmos_steps,
                "num_video_frames_per_chunk": args.cosmos_frames,
                "max_frames": args.cosmos_frames,
                "keep_input_resolution": True,
                "seg": {
                    "control_prompt": "humanoid robot . dining table . apple . robot hand . gripper",
                    "control_weight": 1.0,
                },
            }
        )

    spec_path = cosmos_dir / "spec_styles.jsonl"
    spec_path.write_text("\n".join(json.dumps(s) for s in samples) + "\n", encoding="utf-8")
    manifest = {
        "teacher_checkpoint": args.checkpoint,
        "fix_camera_after_first_frame": True,
        "ego_video": str(ego_video),
        "third_person_video": str(third_video),
        "ego_cosmos_input": str(ego_input),
        "third_cosmos_input": str(third_input),
        "spec": str(spec_path),
        "depth_used": False,
        "sample_count": len(samples),
        "ego_input_info": ffprobe(ego_input),
        "third_input_info": ffprobe(third_input),
    }
    (cosmos_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    run_script = cosmos_dir / "run_cosmos_styles.sh"
    run_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"cd {args.cosmos_root}",
                f"torchrun --nproc_per_node={args.cosmos_nproc} --master_port={args.cosmos_master_port} examples/inference.py \\",
                f"  -i {spec_path} \\",
                f"  -o {cosmos_dir / 'cosmos_outputs'} \\",
                "  --model=edge --disable-guardrails",
                "",
            ]
        ),
        encoding="utf-8",
    )
    run_script.chmod(0o755)
    return spec_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_TEACHER)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sonic-root", default="/root/GRAIL/imports/SONIC")
    parser.add_argument("--cosmos-root", default="/physis/cosmos-transfer2.5")
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--third-person-frame-skip", type=int, default=2)
    parser.add_argument("--fix-third-person-camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--capture-isaac-camera-outputs", action="store_true")
    parser.add_argument("--capture-output-frames", type=int, default=93)
    parser.add_argument("--capture-raw-frames", type=int, default=8)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--ego-video", default=None)
    parser.add_argument("--third-video", default=None)
    parser.add_argument("--cosmos-width", type=int, default=1232)
    parser.add_argument("--cosmos-height", type=int, default=704)
    parser.add_argument("--cosmos-frames", type=int, default=93)
    parser.add_argument("--cosmos-steps", type=int, default=35)
    parser.add_argument("--cosmos-nproc", type=int, default=4)
    parser.add_argument("--cosmos-master-port", type=int, default=12349)
    parser.add_argument("--guidance", type=int, default=3)
    parser.add_argument("--loguru-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_eval:
        if not args.ego_video or not args.third_video:
            raise ValueError("--skip-eval requires --ego-video and --third-video")
        ego_video, third_video = Path(args.ego_video), Path(args.third_video)
    else:
        ego_video, third_video = run_teacher_eval(args, out_dir)
    spec_path = write_cosmos_specs(args, out_dir, ego_video, third_video)
    print(f"Teacher ego video: {ego_video}")
    print(f"Teacher third-person video: {third_video}")
    print(f"Cosmos styles spec: {spec_path}")


if __name__ == "__main__":
    main()
