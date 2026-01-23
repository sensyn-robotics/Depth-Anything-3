#!/usr/bin/env python3
"""
Convert equirectangular 360 video to aligned 3D point cloud using Depth Anything 3.

This script:
1. Extracts frames from equirectangular video
2. Converts each frame to cubemap faces (front, back, left, right, up; optionally down)
3. Processes cubemap faces with DA3-Streaming for depth estimation, Sim3 alignment and loop closure

Outputs:
- combined_pcd.ply: Aligned merged point cloud
- camera_poses.txt: Camera poses (4x4 C2W matrices)
- intrinsic.txt: Camera intrinsics (fx, fy, cx, cy)

Usage:
    python scripts/equirect_to_pointcloud.py \\
        --input /path/to/360_video.mp4 \\
        --output-dir ./output/pointcloud \\
        --fps 2.0 \\
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
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def equirect_to_cubemap_all_faces(
    input_path: str,
    output_dir: str,
    frame_num: int,
    faces: list,
    cube_size: int = 1024,
) -> list:
    """Convert equirectangular image to all cubemap faces in a single ffmpeg call.

    Uses filter_complex to decode input once and output all faces simultaneously.
    This is ~5x faster than calling ffmpeg separately for each face.
    """
    # Build filter_complex for all faces at once
    filter_parts = []
    output_maps = []
    output_paths = []

    for i, face in enumerate(faces):
        config = FACE_CONFIGS[face]
        yaw = config["yaw"]
        pitch = config["pitch"]
        face_idx = config["index"]

        output_filename = f"{frame_num:06d}_{face_idx}_{face}.png"
        output_path = os.path.join(output_dir, output_filename)
        output_paths.append(output_path)

        # Create filter for this face
        filter_parts.append(
            f"[0:v]v360=e:flat:yaw={yaw}:pitch={pitch}:h_fov=90:v_fov=90:w={cube_size}:h={cube_size}[face{i}]"
        )
        output_maps.extend(["-map", f"[face{i}]", output_path])

    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-i", input_path,
        "-filter_complex", filter_complex,
        "-y",
    ] + output_maps

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 and result.stderr:
        print(f"  FFmpeg error: {result.stderr.strip()}")
        return []

    return output_paths


def process_single_frame(args) -> tuple:
    """Process a single frame to cubemap (for parallel execution)."""
    frame_path, output_dir, faces, cube_size, frame_num = args

    try:
        output_paths = equirect_to_cubemap_all_faces(
            frame_path, output_dir, frame_num, faces, cube_size
        )
        return (frame_num, output_paths)
    except Exception as e:
        print(f"  Warning: Failed to process frame {frame_num}: {e}")
        return (frame_num, [])


def convert_frames_to_cubemap(
    frame_paths: list,
    output_dir: str,
    faces: list,
    cube_size: int = 1024,
    max_workers: int = None,
) -> int:
    """Convert equirectangular frames to cubemap faces with parallel processing.

    Optimizations:
    1. Single-decode multi-output: Each frame decoded once, all faces output simultaneously
    2. Parallel processing: Multiple frames processed concurrently

    Returns total number of images created.
    """
    import multiprocessing

    os.makedirs(output_dir, exist_ok=True)

    # Auto-detect worker count (cap at 8 to avoid overwhelming system)
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 8)

    # Prepare task arguments
    task_args = []
    for frame_path in frame_paths:
        frame_name = os.path.basename(frame_path)
        frame_num = int(re.search(r'(\d+)', frame_name).group(1))
        task_args.append((frame_path, output_dir, faces, cube_size, frame_num))

    total_images = 0

    # Process frames in parallel with progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_frame, args): args[4] for args in task_args}

        with tqdm(total=len(frame_paths), desc=f"Converting to cubemap ({max_workers} workers)") as pbar:
            for future in as_completed(futures):
                frame_num, output_paths = future.result()
                total_images += len(output_paths)
                pbar.update(1)

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
    salad_batch_size: int = 32,
    process_res: int = 504,
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
            "process_res": process_res,
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
                "tol": "1e-9",
            },
            "Pointcloud_Save": {
                "sample_ratio": 0.015,
                "conf_threshold_coef": 0.75,
            },
        },
        "Loop": {
            "SALAD": {
                "image_size": [336, 336],
                "batch_size": salad_batch_size,
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

    # Convert all paths to absolute to avoid cwd issues
    da3_streaming_dir = os.path.abspath(da3_streaming_dir)
    image_dir = os.path.abspath(image_dir)
    output_dir = os.path.abspath(output_dir)

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
    config_path = os.path.abspath(config_path)

    # Run DA3-streaming (use just filename since we set cwd)
    script_path = "da3_streaming.py"

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

    # Set PYTHONPATH to include da3_streaming directory for module imports
    env = os.environ.copy()
    env["PYTHONPATH"] = da3_streaming_dir + os.pathsep + env.get("PYTHONPATH", "")

    # Run with real-time output
    process = subprocess.Popen(
        cmd,
        cwd=da3_streaming_dir,
        env=env,
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
    salad_batch_size: int = 32,
    process_res: int = 504,
):
    """Process equirectangular video to 3DGS model using DA3-streaming.

    Supports resuming from intermediate state:
    - If frames already extracted, skips extraction
    - If some cubemap faces already converted, only processes remaining frames
    """

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

    # Step 1: Extract frames from video (with resume support)
    print(f"\n{'='*60}")
    print("Step 1: Extracting frames from video")
    print(f"{'='*60}")

    existing_frames = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    if existing_frames:
        print(f"Found {len(existing_frames)} existing frames, skipping extraction")
        frame_paths = existing_frames
    else:
        frame_paths = extract_frames_from_video(
            input_path, frames_dir, fps=fps, max_frames=max_frames
        )

    if len(frame_paths) == 0:
        print("No frames extracted!")
        return

    # Step 2: Convert to cubemap (with resume support)
    print(f"\n{'='*60}")
    print("Step 2: Converting to cubemap faces")
    print(f"{'='*60}")
    print(f"Faces: {faces}")
    print(f"Cube size: {cube_size}x{cube_size}")

    # Check which frames are already converted
    os.makedirs(cubemap_dir, exist_ok=True)
    existing_cubemap = os.listdir(cubemap_dir)
    processed_frames = set()
    for f in existing_cubemap:
        match = re.match(r'(\d+)_', f)
        if match:
            processed_frames.add(int(match.group(1)))

    # Filter to only unprocessed frames
    remaining_frames = []
    for frame_path in frame_paths:
        frame_num = int(re.search(r'(\d+)', os.path.basename(frame_path)).group(1))
        if frame_num not in processed_frames:
            remaining_frames.append(frame_path)

    if remaining_frames:
        print(f"Already processed: {len(processed_frames)} frames")
        print(f"Remaining to process: {len(remaining_frames)} frames")
        new_images = convert_frames_to_cubemap(
            remaining_frames, cubemap_dir, faces, cube_size
        )
        total_images = len(processed_frames) * len(faces) + new_images
    else:
        print(f"All {len(frame_paths)} frames already converted, skipping")
        total_images = len(existing_cubemap)

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
        salad_batch_size=salad_batch_size,
        process_res=process_res,
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
        "salad_batch_size": salad_batch_size,
        "process_res": process_res,
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

    # Low VRAM GPU (~8-12GB): use --low-memory preset
    python scripts/equirect_to_3dgs.py -i video.mp4 -o ./output --low-memory --keep-temp

    # Fine-tune memory usage: reduce chunk-size and SALAD batch
    python scripts/equirect_to_3dgs.py -i video.mp4 -o ./output --chunk-size 30 --salad-batch-size 8

    # OOM troubleshooting order (try each if previous OOMs):
    #   1. --chunk-size 30 --overlap 15
    #   2. --no-loop (disables SALAD memory usage)
    #   3. --salad-batch-size 8 (if loop needed)
    #   4. --cube-size 768 (last resort, reduces quality)
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
    parser.add_argument(
        "--salad-batch-size",
        type=int,
        default=32,
        help="SALAD loop closure batch size (default: 32, reduce to 8-16 for low VRAM)"
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Enable low-memory mode for 8-12GB VRAM GPUs: sets chunk-size=10, overlap=5, process-res=336, disables loop closure"
    )
    parser.add_argument(
        "--process-res",
        type=int,
        default=504,
        help="DA3 internal processing resolution (default: 504, reduce to 336 or 378 for low VRAM GPUs)"
    )

    args = parser.parse_args()

    # Apply low-memory defaults if enabled, but allow explicit overrides
    chunk_size = args.chunk_size
    overlap = args.overlap
    salad_batch_size = args.salad_batch_size
    loop_enable = not args.no_loop
    process_res = args.process_res

    if args.low_memory:
        print("Low-memory mode enabled: using conservative settings for 8-12GB VRAM GPUs")
        # Only apply low-memory values if user didn't explicitly override
        if args.chunk_size == 60:  # default
            chunk_size = 10
        if args.overlap == 30:  # default
            overlap = 5
        if args.salad_batch_size == 32:  # default
            salad_batch_size = 8
        if not args.no_loop:  # only disable if user didn't explicitly set --no-loop
            loop_enable = False
        if args.process_res == 504:  # default
            process_res = 336  # Tested on 11.6GB GPU

    faces = [f.strip() for f in args.faces.split(",")]

    process_equirect_to_3dgs(
        input_path=args.input,
        output_dir=args.output_dir,
        faces=faces,
        fps=args.fps,
        max_frames=args.max_frames,
        cube_size=args.cube_size,
        chunk_size=chunk_size,
        overlap=overlap,
        loop_enable=loop_enable,
        keep_temp=args.keep_temp,
        salad_batch_size=salad_batch_size,
        process_res=process_res,
    )


if __name__ == "__main__":
    main()
