#!/usr/bin/env python3
"""
Video/Images to CityGaussian processing pipeline.

This script processes videos or images through DA3-streaming for depth
estimation and camera poses, then trains a CityGaussian model for
large-scale 3D Gaussian Splatting reconstruction.

CityGaussian is optimized for large-scale scenes and provides:
- Block-wise training for memory efficiency
- Better handling of large outdoor scenes
- PyTorch Lightning-based training

Usage:
    # From video
    python scripts/run_citygaussian.py --video data/video.mp4

    # From images
    python scripts/run_citygaussian.py --images data/frames/

    # Multiple videos -> ONE combined model
    python scripts/run_citygaussian.py --video data/*.mp4

    # Low memory mode for 8-12GB VRAM GPUs
    python scripts/run_citygaussian.py --video data/video.mp4 --low-memory

    # Quick test
    python scripts/run_citygaussian.py --images data/frames/ --max-steps 3000

Output:
    output/{name}/
        ├── frames/           # Extracted frames (if from video)
        ├── pointcloud/       # DA3 output
        │   ├── combined_pcd.ply
        │   ├── camera_poses.txt
        │   └── intrinsic.txt
        ├── scene/            # COLMAP format for CityGaussian
        │   ├── images/
        │   └── sparse/0/
        └── gaussian_splatting/  # CityGaussian output
"""

import argparse
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np


def find_citygaussian_path() -> Optional[Path]:
    """Find CityGaussian installation path."""
    script_dir = Path(__file__).parent.parent  # DA3 root

    possible_paths = [
        script_dir.parent / "CityGaussian",
        Path.home() / "proj/sensyn/CityGaussian",
        Path("/home/mas/proj/sensyn/CityGaussian"),
    ]

    for path in possible_paths:
        if (path / "main.py").exists():
            return path

    return None


def clear_gpu_memory():
    """Clear GPU memory."""
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def extract_frames(
    video_paths: list,
    output_dir: Path,
    fps: float,
    force: bool = False,
) -> Path:
    """Extract frames from video(s).

    Args:
        video_paths: List of video file paths
        output_dir: Output directory
        fps: Frame extraction rate
        force: Force re-extraction

    Returns:
        Path to frames directory
    """
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    existing_frames = list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png"))
    if existing_frames and not force:
        print(f"Found {len(existing_frames)} existing frames, skipping extraction")
        return frames_dir

    # Clear existing frames if force
    if force:
        for f in frames_dir.glob("*"):
            f.unlink()

    frame_idx = 0
    for i, video_path in enumerate(video_paths):
        video_path = Path(video_path)
        print(f"  Extracting from {video_path.name}...")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            cmd = [
                "ffmpeg", "-i", str(video_path),
                "-vf", f"fps={fps}",
                "-q:v", "2",
                str(temp_path / "frame_%06d.jpg"),
                "-y",
            ]
            subprocess.run(cmd, check=True, capture_output=True)

            temp_frames = sorted(temp_path.glob("frame_*.jpg"))
            for src in temp_frames:
                dst = frames_dir / f"frame_{frame_idx:06d}.jpg"
                shutil.copy2(src, dst)
                frame_idx += 1

    print(f"  Extracted {frame_idx} frames total")
    return frames_dir


def run_da3_streaming(
    images_dir: Path,
    output_dir: Path,
    chunk_size: int = 20,
    overlap: int = 10,
    process_res: int = 504,
    low_memory: bool = False,
    force: bool = False,
) -> dict:
    """Run DA3-streaming for depth and camera poses.

    Args:
        images_dir: Directory containing input images
        output_dir: Output directory
        chunk_size: DA3 chunk size
        overlap: Chunk overlap
        process_res: Processing resolution
        low_memory: Enable low-memory mode
        force: Force reprocessing

    Returns:
        Dictionary with output paths
    """
    pointcloud_dir = output_dir / "pointcloud"

    # Check if already processed
    combined_pcd = pointcloud_dir / "combined_pcd.ply"
    if combined_pcd.exists() and not force:
        print(f"Found existing pointcloud at {combined_pcd}, skipping DA3")
        return {
            "pointcloud_path": str(combined_pcd),
            "poses_path": str(pointcloud_dir / "camera_poses.txt"),
            "intrinsics_path": str(pointcloud_dir / "intrinsic.txt"),
        }

    # Import and run
    from frames_to_pointcloud import process_frames_to_pointcloud

    # Adjust settings for low memory
    if low_memory:
        chunk_size = min(chunk_size, 10)
        overlap = min(overlap, 5)
        process_res = min(process_res, 336)
        salad_batch = 4
    else:
        salad_batch = 32

    print(f"  Settings: chunk_size={chunk_size}, overlap={overlap}, process_res={process_res}")

    result = process_frames_to_pointcloud(
        input_dir=str(images_dir),
        output_dir=str(pointcloud_dir),
        chunk_size=chunk_size,
        overlap=overlap,
        loop_enable=True,
        salad_batch_size=salad_batch,
        process_res=process_res,
        force=force,
    )

    return result


def convert_da3_to_colmap(
    pointcloud_path: str,
    poses_path: str,
    intrinsics_path: str,
    images_dir: Path,
    output_dir: Path,
    max_points: int = 500000,
) -> Path:
    """Convert DA3 output to COLMAP format.

    Args:
        pointcloud_path: Path to combined_pcd.ply
        poses_path: Path to camera_poses.txt
        intrinsics_path: Path to intrinsic.txt
        images_dir: Directory containing input images
        output_dir: Output directory for COLMAP files

    Returns:
        Path to scene directory
    """
    from plyfile import PlyData

    scene_dir = output_dir / "scene"
    sparse_dir = scene_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Create images symlink
    images_link = scene_dir / "images"
    if images_link.exists() or images_link.is_symlink():
        if images_link.is_symlink():
            images_link.unlink()
        else:
            shutil.rmtree(images_link)
    images_link.symlink_to(images_dir.resolve())

    # Load DA3 outputs
    print("  Loading DA3 outputs...")

    # Load poses (C2W matrices)
    poses = []
    with open(poses_path, "r") as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            if len(values) == 16:
                pose = np.array(values).reshape(4, 4)
                poses.append(pose)
    poses = np.array(poses)
    print(f"    Loaded {len(poses)} poses")

    # Load intrinsics
    intrinsics = []
    with open(intrinsics_path, "r") as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            if len(values) == 4:
                intrinsics.append(values)
    intrinsics = np.array(intrinsics)
    print(f"    Loaded {len(intrinsics)} intrinsics")

    # Load point cloud
    plydata = PlyData.read(pointcloud_path)
    vertex = plydata["vertex"]
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1)

    if "red" in vertex.data.dtype.names:
        rgb = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=-1)
    elif "r" in vertex.data.dtype.names:
        rgb = np.stack([vertex["r"], vertex["g"], vertex["b"]], axis=-1)
    else:
        rgb = np.ones_like(xyz, dtype=np.uint8) * 128

    print(f"    Loaded {len(xyz)} points")

    # Subsample if too many points
    if len(xyz) > max_points:
        indices = np.random.choice(len(xyz), max_points, replace=False)
        xyz = xyz[indices]
        rgb = rgb[indices]
        print(f"    Subsampled to {max_points} points")

    # Get image files
    extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    image_files = []
    for ext in extensions:
        image_files.extend(images_dir.glob(f"*{ext}"))
    image_files = sorted(image_files, key=lambda x: x.name)
    print(f"    Found {len(image_files)} images")

    # Match counts
    min_len = min(len(image_files), len(poses), len(intrinsics))
    if len(image_files) != len(poses):
        print(f"    Warning: count mismatch, using {min_len}")
        image_files = image_files[:min_len]
        poses = poses[:min_len]
        intrinsics = intrinsics[:min_len]

    # Get image size
    from PIL import Image
    with Image.open(image_files[0]) as img:
        width, height = img.size
    print(f"    Image size: {width}x{height}")

    # Write COLMAP binary files
    print("  Writing COLMAP files...")

    # cameras.bin
    with open(sparse_dir / "cameras.bin", "wb") as f:
        f.write(struct.pack("Q", len(intrinsics)))
        for i, intr in enumerate(intrinsics):
            fx, fy, cx, cy = intr
            f.write(struct.pack("I", i + 1))  # camera_id
            f.write(struct.pack("i", 1))  # PINHOLE model
            f.write(struct.pack("Q", width))
            f.write(struct.pack("Q", height))
            for p in [fx, fy, cx, cy]:
                f.write(struct.pack("d", p))

    # images.bin
    def rotation_matrix_to_quaternion(R):
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return np.array([w, x, y, z])

    with open(sparse_dir / "images.bin", "wb") as f:
        f.write(struct.pack("Q", len(poses)))
        for i, (pose, img_file) in enumerate(zip(poses, image_files)):
            # Convert C2W to W2C
            w2c = np.linalg.inv(pose)
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            qvec = rotation_matrix_to_quaternion(R)

            f.write(struct.pack("I", i + 1))  # image_id
            for q in qvec:
                f.write(struct.pack("d", q))
            for tv in t:
                f.write(struct.pack("d", tv))
            f.write(struct.pack("I", i + 1))  # camera_id
            name_bytes = img_file.name.encode("utf-8")
            f.write(name_bytes + b"\x00")
            f.write(struct.pack("Q", 0))  # num_points2D

    # points3D.bin
    with open(sparse_dir / "points3D.bin", "wb") as f:
        f.write(struct.pack("Q", len(xyz)))
        for i in range(len(xyz)):
            f.write(struct.pack("Q", i + 1))  # point3D_id
            for x in xyz[i].astype(np.float64):
                f.write(struct.pack("d", x))
            for c in rgb[i].astype(np.uint8):
                f.write(struct.pack("B", c))
            f.write(struct.pack("d", 0.0))  # error
            f.write(struct.pack("Q", 0))  # track length

    print(f"    Written: {sparse_dir}")
    return scene_dir


def run_citygaussian(
    scene_dir: Path,
    output_dir: Path,
    max_steps: int = 30000,
    down_sample: int = 1,
) -> dict:
    """Run CityGaussian training.

    Args:
        scene_dir: COLMAP scene directory
        output_dir: Output directory
        max_steps: Maximum training steps
        down_sample: Image downsample factor

    Returns:
        Dictionary with output paths
    """
    citygaussian_path = find_citygaussian_path()
    if citygaussian_path is None:
        raise FileNotFoundError(
            "CityGaussian not found. Please install it at:\n"
            "  - ../CityGaussian (relative to DA3)\n"
            "  - ~/proj/sensyn/CityGaussian"
        )

    print(f"  CityGaussian: {citygaussian_path}")
    print(f"  Scene: {scene_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Max steps: {max_steps}")

    # Build command
    cmd = [
        sys.executable, str(citygaussian_path / "main.py"), "fit",
        "--data.path", str(scene_dir),
        "--data.parser", "Colmap",
        "--max_steps", str(max_steps),
        "--output", str(output_dir),
    ]

    if down_sample > 1:
        cmd.extend([
            "--data.parser.init_args.down_sample_factor", str(down_sample),
            "--data.parser.init_args.down_sample_rounding_mode", "ceil",
        ])

    # Use CityGaussian's venv if available
    cg_venv = citygaussian_path / ".venv" / "bin" / "python"
    if cg_venv.exists():
        cmd[0] = str(cg_venv)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(citygaussian_path)

    result = subprocess.run(cmd, cwd=str(citygaussian_path), env=env)
    if result.returncode != 0:
        raise RuntimeError("CityGaussian training failed")

    # Find output PLY
    scene_name = scene_dir.parent.name
    gs_output = output_dir / scene_name
    ply_files = list(gs_output.rglob("point_cloud.ply"))

    return {
        "output_dir": str(gs_output),
        "final_ply": str(ply_files[0]) if ply_files else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Video/Images to CityGaussian pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # From video
    python scripts/run_citygaussian.py --video data/video.mp4

    # From images
    python scripts/run_citygaussian.py --images data/frames/

    # Multiple videos -> ONE combined model
    python scripts/run_citygaussian.py --video data/*.mp4

    # Low memory mode for 8-12GB VRAM GPUs
    python scripts/run_citygaussian.py --video data/video.mp4 --low-memory

    # Quick test with fewer steps
    python scripts/run_citygaussian.py --images data/frames/ --max-steps 3000

    # Skip DA3 (use existing pointcloud)
    python scripts/run_citygaussian.py --images data/frames/ -o ./output --skip-da3
        """,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--video", "-v",
        nargs="+",
        help="Input video file(s)",
    )
    input_group.add_argument(
        "--images", "-i",
        type=Path,
        help="Input image directory",
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("./output"),
        help="Output directory (default: ./output)",
    )

    # Video options
    parser.add_argument(
        "--fps",
        type=float,
        default=0.5,
        help="Frame extraction FPS for video (default: 0.5)",
    )

    # DA3 options
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=20,
        help="DA3 chunk size (default: 20 for 12GB VRAM)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=10,
        help="DA3 chunk overlap (default: 10)",
    )
    parser.add_argument(
        "--process-res",
        type=int,
        default=504,
        help="DA3 processing resolution (default: 504)",
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Enable low-memory mode for 8-12GB VRAM GPUs",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=500000,
        help="Maximum points for COLMAP (default: 500000)",
    )

    # CityGaussian options
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30000,
        help="CityGaussian training steps (default: 30000)",
    )
    parser.add_argument(
        "--down-sample",
        type=int,
        default=1,
        help="Image downsample factor (default: 1)",
    )

    # Skip options
    parser.add_argument(
        "--skip-da3",
        action="store_true",
        help="Skip DA3, use existing pointcloud",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip CityGaussian training (only generate COLMAP)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if outputs exist",
    )

    args = parser.parse_args()

    # Determine output name and directory
    if args.video:
        video_paths = [Path(v) for v in args.video]
        for v in video_paths:
            if not v.exists():
                print(f"Error: Video not found: {v}")
                sys.exit(1)
        if len(video_paths) == 1:
            name = video_paths[0].stem
        else:
            name = "combined"
    else:
        if not args.images.exists():
            print(f"Error: Image directory not found: {args.images}")
            sys.exit(1)
        name = args.images.name

    output_dir = args.output / name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("DA3 → CityGaussian Pipeline")
    print("=" * 60)
    print(f"Output: {output_dir}")
    if args.low_memory:
        print("Mode: Low-memory (8-12GB VRAM)")
    print("=" * 60)

    # Stage 1: Get images
    if args.video:
        print("\n[Stage 1/4] Extracting frames from video(s)")
        images_dir = extract_frames(
            [str(v) for v in video_paths],
            output_dir,
            args.fps,
            args.force,
        )
    else:
        print("\n[Stage 1/4] Using existing images")
        images_dir = args.images

    # Stage 2: DA3 streaming
    if not args.skip_da3:
        print("\n[Stage 2/4] Running DA3-streaming (depth + poses)")
        clear_gpu_memory()

        da3_result = run_da3_streaming(
            images_dir,
            output_dir,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            process_res=args.process_res,
            low_memory=args.low_memory,
            force=args.force,
        )
    else:
        print("\n[Stage 2/4] Skipping DA3 (using existing pointcloud)")
        pointcloud_dir = output_dir / "pointcloud"
        da3_result = {
            "pointcloud_path": str(pointcloud_dir / "combined_pcd.ply"),
            "poses_path": str(pointcloud_dir / "camera_poses.txt"),
            "intrinsics_path": str(pointcloud_dir / "intrinsic.txt"),
        }

    clear_gpu_memory()

    # Stage 3: Convert to COLMAP
    print("\n[Stage 3/4] Converting to COLMAP format")
    scene_dir = convert_da3_to_colmap(
        da3_result["pointcloud_path"],
        da3_result["poses_path"],
        da3_result["intrinsics_path"],
        images_dir,
        output_dir,
        max_points=args.max_points,
    )

    # Stage 4: CityGaussian training
    if not args.skip_training:
        print("\n[Stage 4/4] Training CityGaussian")
        clear_gpu_memory()

        gs_output = output_dir / "gaussian_splatting"
        gs_result = run_citygaussian(
            scene_dir,
            gs_output,
            max_steps=args.max_steps,
            down_sample=args.down_sample,
        )
    else:
        print("\n[Stage 4/4] Skipping CityGaussian training")
        gs_result = {}

    print("\n" + "=" * 60)
    print("Pipeline completed!")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"  - Pointcloud: {output_dir / 'pointcloud'}")
    print(f"  - COLMAP scene: {scene_dir}")
    if gs_result.get("final_ply"):
        print(f"  - Final model: {gs_result['final_ply']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
