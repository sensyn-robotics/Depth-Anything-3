#!/usr/bin/env python3
"""
Convert equirectangular 360 video to 3D Gaussian Splatting model using Depth Anything 3.

This script:
1. Extracts frames from equirectangular video
2. Converts each frame to cubemap faces (front, back, left, right, up; optionally down)
3. Processes cubemap faces with DA3-Streaming for proper chunk alignment and loop closure

Usage:
    python scripts/equirect_to_3dgs.py \
        --input /path/to/360_video.mp4 \
        --output-dir ./output/3dgs \
        --fps 2.0 \
        --cube-size 1024
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


# Cubemap face configurations
FACE_CONFIGS = {
    "front":  {"yaw": 0,   "pitch": 0,   "index": 0},
    "back":   {"yaw": 180, "pitch": 0,   "index": 1},
    "left":   {"yaw": -90, "pitch": 0,   "index": 2},
    "right":  {"yaw": 90,  "pitch": 0,   "index": 3},
    "up":     {"yaw": 0,   "pitch": 90,  "index": 4},
    "down":   {"yaw": 0,   "pitch": -90, "index": 5},
}


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    fps: float = 1.0,
    max_frames: int = None,
) -> list:
    """Extract frames from video at specified FPS."""
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",
        "-i", video_path,
        "-vf", f"fps={fps}",
    ]

    if max_frames:
        cmd.extend(["-frames:v", str(max_frames)])

    output_pattern = os.path.join(output_dir, "frame_%06d.png")
    cmd.append(output_pattern)

    print(f"Extracting frames at {fps} FPS...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"FFmpeg error: {result.stderr}")
        raise RuntimeError("Frame extraction failed")

    frames = sorted(glob.glob(os.path.join(output_dir, "frame_*.png")))
    print(f"Extracted {len(frames)} frames")
    return frames


def equirect_to_cubemap_ffmpeg(
    input_path: str,
    output_path: str,
    face: str,
    cube_size: int = 1024,
) -> bool:
    """Convert equirectangular image to a single cubemap face using ffmpeg."""
    config = FACE_CONFIGS[face]
    yaw = config["yaw"]
    pitch = config["pitch"]

    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",
        "-i", input_path,
        "-vf", f"v360=e:flat:yaw={yaw}:pitch={pitch}:h_fov=90:v_fov=90:w={cube_size}:h={cube_size}",
        "-frames:v", "1",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 and result.stderr:
        print(f"  FFmpeg error for {face}: {result.stderr.strip()}")
    return result.returncode == 0


def convert_frames_to_cubemap(
    frame_paths: list,
    output_dir: str,
    faces: list,
    cube_size: int = 1024,
) -> int:
    """Convert equirectangular frames to cubemap faces.

    Returns total number of images created.
    """
    os.makedirs(output_dir, exist_ok=True)

    total_images = 0
    for frame_path in tqdm(frame_paths, desc="Converting to cubemap"):
        frame_name = os.path.basename(frame_path)
        frame_num = int(re.search(r'(\d+)', frame_name).group(1))

        for face in faces:
            face_idx = FACE_CONFIGS[face]["index"]
            output_filename = f"{frame_num:06d}_{face_idx}_{face}.png"
            output_path = os.path.join(output_dir, output_filename)

            if equirect_to_cubemap_ffmpeg(frame_path, output_path, face, cube_size):
                total_images += 1

    return total_images


def check_da3_streaming_weights(da3_streaming_dir: str) -> bool:
    """Check if DA3-streaming weights are downloaded."""
    weights_dir = os.path.join(da3_streaming_dir, "weights")
    required_files = ["model.safetensors", "config.json", "dino_salad.ckpt"]

    for f in required_files:
        if not os.path.exists(os.path.join(weights_dir, f)):
            return False
    return True


def download_da3_streaming_weights(da3_streaming_dir: str):
    """Download DA3-streaming weights if not present."""
    script_path = os.path.join(da3_streaming_dir, "scripts", "download_weights.sh")

    if os.path.exists(script_path):
        print("Downloading DA3-streaming weights...")
        result = subprocess.run(
            ["bash", script_path],
            cwd=da3_streaming_dir,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Warning: Weight download may have failed: {result.stderr}")
    else:
        print(f"Warning: Download script not found at {script_path}")
        print("Please download weights manually following da3_streaming/README.md")


def create_streaming_config(
    config_path: str,
    chunk_size: int = 60,
    overlap: int = 30,
    loop_enable: bool = True,
):
    """Create a config file for DA3-streaming."""
    config = {
        "Weights": {
            "DA3": "./weights/model.safetensors",
            "DA3_CONFIG": "./weights/config.json",
            "SALAD": "./weights/dino_salad.ckpt",
        },
        "Model": {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "loop_chunk_size": 20,
            "loop_enable": loop_enable,
            "useDBoW": False,
            "delete_temp_files": True,
            "align_lib": "triton",
            "align_method": "sim3",
            "scale_compute_method": "auto",
            "align_type": "dense",
            "ref_view_strategy": "saddle_balanced",
            "ref_view_strategy_loop": "saddle_balanced",
            "depth_threshold": 15.0,
            "save_depth_conf_result": False,
            "save_debug_info": False,
            "Sparse_Align": {
                "keypoint_select": "orb",
                "keypoint_num": 5000,
            },
            "IRLS": {
                "delta": 0.1,
                "max_iters": 5,
                "tol": 1e-9,
            },
            "Pointcloud_Save": {
                "sample_ratio": 0.015,
                "conf_threshold_coef": 0.75,
            },
        },
        "Loop": {
            "SALAD": {
                "image_size": [336, 336],
                "batch_size": 32,
                "similarity_threshold": 0.85,
                "top_k": 5,
                "use_nms": True,
                "nms_threshold": 25,
            },
            "SIM3_Optimizer": {
                "lang_version": "cpp",
                "max_iterations": 30,
                "lambda_init": 1e-6,
            },
        },
    }

    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return config_path


def run_da3_streaming(
    image_dir: str,
    output_dir: str,
    da3_streaming_dir: str,
    config_path: str = None,
    chunk_size: int = 60,
    overlap: int = 30,
):
    """Run DA3-streaming on the cubemap images."""

    # Check weights
    if not check_da3_streaming_weights(da3_streaming_dir):
        download_da3_streaming_weights(da3_streaming_dir)

        if not check_da3_streaming_weights(da3_streaming_dir):
            raise RuntimeError(
                "DA3-streaming weights not found. Please download them manually:\n"
                f"  cd {da3_streaming_dir} && bash scripts/download_weights.sh"
            )

    # Create config if not provided
    if config_path is None:
        config_path = os.path.join(output_dir, "streaming_config.yaml")
        create_streaming_config(config_path, chunk_size, overlap)

    # Run DA3-streaming
    script_path = os.path.join(da3_streaming_dir, "da3_streaming.py")

    cmd = [
        sys.executable, script_path,
        "--image_dir", image_dir,
        "--output_dir", output_dir,
        "--config", config_path,
    ]

    print(f"\nRunning DA3-streaming...")
    print(f"  Image dir: {image_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Config: {config_path}")

    # Run with real-time output
    process = subprocess.Popen(
        cmd,
        cwd=da3_streaming_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        print(line, end='')

    process.wait()

    if process.returncode != 0:
        raise RuntimeError(f"DA3-streaming failed with return code {process.returncode}")

    return output_dir


def process_equirect_to_3dgs(
    input_path: str,
    output_dir: str,
    faces: list = None,
    fps: float = 1.0,
    max_frames: int = None,
    cube_size: int = 1024,
    chunk_size: int = 60,
    overlap: int = 30,
    loop_enable: bool = True,
    keep_temp: bool = False,
):
    """Process equirectangular video to 3DGS model using DA3-streaming."""

    if faces is None:
        # Default: exclude bottom (usually captures camera operator)
        faces = ["front", "back", "left", "right", "up"]

    # Find DA3-streaming directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    da3_streaming_dir = os.path.join(project_root, "da3_streaming")

    if not os.path.exists(da3_streaming_dir):
        raise RuntimeError(f"DA3-streaming directory not found: {da3_streaming_dir}")

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = os.path.join(output_dir, "_temp")
    frames_dir = os.path.join(temp_dir, "frames")
    cubemap_dir = os.path.join(temp_dir, "cubemap")

    # Step 1: Extract frames from video
    print(f"\n{'='*60}")
    print("Step 1: Extracting frames from video")
    print(f"{'='*60}")
    frame_paths = extract_frames_from_video(
        input_path, frames_dir, fps=fps, max_frames=max_frames
    )

    if len(frame_paths) == 0:
        print("No frames extracted!")
        return

    # Step 2: Convert to cubemap
    print(f"\n{'='*60}")
    print("Step 2: Converting to cubemap faces")
    print(f"{'='*60}")
    print(f"Faces: {faces}")
    print(f"Cube size: {cube_size}x{cube_size}")

    total_images = convert_frames_to_cubemap(
        frame_paths, cubemap_dir, faces, cube_size
    )
    print(f"Total cubemap images: {total_images}")

    # Step 3: Run DA3-streaming
    print(f"\n{'='*60}")
    print("Step 3: Running DA3-streaming (with Sim3 alignment & loop closure)")
    print(f"{'='*60}")

    streaming_output = os.path.join(output_dir, "streaming_output")

    # Create config
    config_path = os.path.join(output_dir, "streaming_config.yaml")
    create_streaming_config(
        config_path,
        chunk_size=chunk_size,
        overlap=overlap,
        loop_enable=loop_enable,
    )

    run_da3_streaming(
        image_dir=cubemap_dir,
        output_dir=streaming_output,
        da3_streaming_dir=da3_streaming_dir,
        config_path=config_path,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    # Step 4: Copy final outputs to main output dir
    print(f"\n{'='*60}")
    print("Step 4: Organizing outputs")
    print(f"{'='*60}")

    # Copy main outputs
    pcd_src = os.path.join(streaming_output, "pcd", "combined_pcd.ply")
    poses_src = os.path.join(streaming_output, "camera_poses.txt")
    intrinsics_src = os.path.join(streaming_output, "intrinsic.txt")

    if os.path.exists(pcd_src):
        shutil.copy(pcd_src, os.path.join(output_dir, "combined_pcd.ply"))
        print(f"  Point cloud: {output_dir}/combined_pcd.ply")

    if os.path.exists(poses_src):
        shutil.copy(poses_src, os.path.join(output_dir, "camera_poses.txt"))
        print(f"  Camera poses: {output_dir}/camera_poses.txt")

    if os.path.exists(intrinsics_src):
        shutil.copy(intrinsics_src, os.path.join(output_dir, "intrinsic.txt"))
        print(f"  Intrinsics: {output_dir}/intrinsic.txt")

    # Clean up temp files
    if not keep_temp:
        print("\nCleaning up temporary files...")
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Save metadata
    metadata = {
        "input_video": input_path,
        "faces": faces,
        "fps": fps,
        "cube_size": cube_size,
        "num_frames": len(frame_paths),
        "num_images": total_images,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "loop_enable": loop_enable,
        "method": "da3_streaming",
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print("Results saved to:")
    print(f"  - Point cloud: {output_dir}/combined_pcd.ply")
    print(f"  - Camera poses: {output_dir}/camera_poses.txt")
    print(f"  - Intrinsics: {output_dir}/intrinsic.txt")
    print(f"  - Metadata: {output_dir}/metadata.json")
    print(f"  - Streaming output: {streaming_output}/")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert equirectangular 360 video to aligned 3D point cloud using DA3-streaming",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python scripts/equirect_to_3dgs.py -i video.mp4 -o ./output

    # Higher quality with more frames
    python scripts/equirect_to_3dgs.py -i video.mp4 -o ./output --fps 2.0 --cube-size 1024

    # Quick test
    python scripts/equirect_to_3dgs.py -i video.mp4 -o ./output --fps 0.5 --max-frames 30

    # Disable loop closure for faster processing
    python scripts/equirect_to_3dgs.py -i video.mp4 -o ./output --no-loop
        """
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input equirectangular video file (MP4)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./output/equirect_3dgs",
        help="Output directory"
    )
    parser.add_argument(
        "--faces",
        default="front,back,left,right,up",
        help="Cubemap faces to process (default: excludes 'down')"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frame extraction rate (default: 1.0)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to extract (for testing)"
    )
    parser.add_argument(
        "--cube-size",
        type=int,
        default=1024,
        help="Cubemap face size in pixels (default: 1024)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=60,
        help="DA3-streaming chunk size (default: 60)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=30,
        help="Overlap between chunks (default: 30, should be ~half of chunk-size)"
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="Disable loop closure detection (faster but less accurate)"
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary files (extracted frames, cubemap images)"
    )

    args = parser.parse_args()

    faces = [f.strip() for f in args.faces.split(",")]

    process_equirect_to_3dgs(
        input_path=args.input,
        output_dir=args.output_dir,
        faces=faces,
        fps=args.fps,
        max_frames=args.max_frames,
        cube_size=args.cube_size,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        loop_enable=not args.no_loop,
        keep_temp=args.keep_temp,
    )


if __name__ == "__main__":
    main()
