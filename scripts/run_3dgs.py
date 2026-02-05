#!/usr/bin/env python3
"""
One-liner video to 3DGS processing.

This script processes regular videos through the full pipeline:
1. Extract frames from video(s)
2. DA3-streaming for depth estimation and camera poses
3. 3DGS training with gsplat

Usage:
    # Single video
    python scripts/run_3dgs.py data/video.mp4

    # Multiple videos -> ONE combined model
    python scripts/run_3dgs.py data/*.mp4

    # With options
    python scripts/run_3dgs.py data/*.mp4 --fps 1.0 --low-memory --iterations 30000

Output:
    output/combined/  (or output/{video_name}/ for single video)
        ├── frames/
        ├── pointcloud/
        │   ├── combined_pcd.ply
        │   ├── camera_poses.txt
        │   └── intrinsic.txt
        └── 3dgs/
            └── model/point_cloud/iteration_XXXXX/point_cloud.ply
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from frames_to_pointcloud import process_frames_to_pointcloud
from pointcloud_to_3dgs import process_pointcloud_to_3dgs


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


def extract_frames_with_offset(
    video_path: str,
    output_dir: Path,
    fps: float,
    start_idx: int,
    force: bool,
) -> int:
    """Extract frames from video with global sequential numbering.

    Args:
        video_path: Path to input video file
        output_dir: Output directory for frames
        fps: Frame extraction rate
        start_idx: Starting frame index for numbering
        force: Force reprocessing even if outputs exist

    Returns:
        Number of frames extracted
    """
    # Extract to temp dir first, then rename with global indices
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Use ffmpeg directly for more control
        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-vf",
            f"fps={fps}",
            "-q:v",
            "2",
            str(temp_path / "frame_%06d.jpg"),
            "-y",
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        # Move frames with global numbering
        temp_frames = sorted(temp_path.glob("frame_*.jpg"))
        for i, src in enumerate(temp_frames):
            dst = output_dir / f"frame_{start_idx + i:06d}.jpg"
            if not dst.exists() or force:
                shutil.copy2(src, dst)

        return len(temp_frames)


def process_videos_unified(
    video_paths: list,
    output_dir: Path,
    fps: float,
    low_memory: bool,
    iterations: int,
    resolution: int,
    force: bool,
) -> dict:
    """Process all videos together as ONE unified 3DGS model.

    This extracts frames from ALL videos into a single directory, then runs
    DA3-streaming once on all frames together. This enables loop closure to
    find cross-video matches and produces a unified reconstruction.

    Args:
        video_paths: List of video file paths
        output_dir: Output directory
        fps: Frame extraction rate
        low_memory: Enable low-memory mode for 8-12GB GPUs
        iterations: 3DGS training iterations
        resolution: Training resolution (-1 for original)
        force: Force reprocessing even if outputs exist

    Returns:
        Dictionary with final output paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Extract frames from ALL videos into combined directory
    print("\n" + "=" * 50)
    print("Stage 1: Extracting frames from all videos")
    print("=" * 50)

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Check if frames already exist
    existing_frames = sorted(list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpg")))
    if existing_frames and not force:
        print(f"Found {len(existing_frames)} existing frames in {frames_dir}")
        print("Use --force to re-extract")
    else:
        # Clear existing frames if force
        if force:
            for f in frames_dir.glob("*"):
                f.unlink()

        frame_idx = 0
        for i, video_path in enumerate(video_paths):
            video_path = Path(video_path)
            print(f"\n  Video {i + 1}/{len(video_paths)}: {video_path.name}")

            n_extracted = extract_frames_with_offset(
                str(video_path), frames_dir, fps, frame_idx, force
            )
            print(
                f"    Extracted {n_extracted} frames (indices {frame_idx}-{frame_idx + n_extracted - 1})"
            )
            frame_idx += n_extracted

        print(f"\n  Total frames extracted: {frame_idx}")

    # Stage 2: DA3-streaming (depth + poses) on ALL frames together
    print("\n" + "=" * 50)
    print("Stage 2: DA3-streaming (depth + poses) - unified")
    print("=" * 50)

    pointcloud_dir = output_dir / "pointcloud"

    # Memory-optimized settings with loop closure ENABLED
    # Key insight: loop closure prevents drift accumulation across chunks
    if low_memory:
        chunk_size = 20  # Larger chunks = fewer alignments = less drift
        overlap = 10  # More overlap = better alignment
        salad_batch = 4  # Smaller batch for SALAD to fit in VRAM
        loop_en = True  # CRITICAL: Enable loop closure even in low-memory mode
        proc_res = 336
    else:
        chunk_size = 60
        overlap = 30
        salad_batch = 32
        loop_en = True
        proc_res = 504

    print(f"  Settings: chunk_size={chunk_size}, overlap={overlap}, loop_enable={loop_en}")
    print(f"            salad_batch={salad_batch}, process_res={proc_res}")

    result = process_frames_to_pointcloud(
        input_dir=str(frames_dir),
        output_dir=str(pointcloud_dir),
        chunk_size=chunk_size,
        overlap=overlap,
        loop_enable=loop_en,
        salad_batch_size=salad_batch,
        process_res=proc_res,
        force=force,
    )

    clear_gpu_memory()  # Free VRAM before 3DGS

    # Stage 3: 3DGS training
    print("\n" + "=" * 50)
    print("Stage 3: 3DGS training")
    print("=" * 50)

    gs_dir = output_dir / "3dgs"
    gs_result = process_pointcloud_to_3dgs(
        pointcloud_path=result["pointcloud_path"],
        poses_path=result["poses_path"],
        intrinsics_path=result["intrinsics_path"],
        images_dir=str(frames_dir),
        output_dir=str(gs_dir),
        iterations=iterations,
        resolution=resolution,
        backend="gsplat",
        strategy="mcmc",
        force=force,
    )

    return gs_result


def process_videos(
    video_paths: list,
    base_output: Path,
    fps: float,
    low_memory: bool,
    iterations: int,
    resolution: int,
    force: bool,
) -> dict:
    """Process multiple videos into a single 3DGS model.

    Uses unified processing: extracts all frames first, then runs DA3-streaming
    once on all frames together. This enables loop closure to find cross-video
    matches and produces better results than processing videos separately.

    Args:
        video_paths: List of video file paths
        base_output: Base output directory
        fps: Frame extraction rate
        low_memory: Enable low-memory mode
        iterations: 3DGS training iterations
        resolution: Training resolution (-1 for original)
        force: Force reprocessing

    Returns:
        Dictionary with final output paths
    """
    # Determine output directory name
    if len(video_paths) == 1:
        output_dir = base_output / Path(video_paths[0]).stem
    else:
        output_dir = base_output / "combined"

    # Use unified processing for all cases (single or multiple videos)
    # This ensures loop closure works properly across all frames
    return process_videos_unified(
        video_paths,
        output_dir,
        fps,
        low_memory,
        iterations,
        resolution,
        force,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Video to 3DGS one-liner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single video
    python scripts/run_3dgs.py data/video.mp4

    # Multiple videos -> ONE combined model
    python scripts/run_3dgs.py data/*.mp4

    # Low memory mode for 8-12GB VRAM GPUs
    python scripts/run_3dgs.py data/*.mp4 --low-memory

    # Quick test with fewer iterations
    python scripts/run_3dgs.py data/video.mp4 --iterations 7000

    # Force reprocessing
    python scripts/run_3dgs.py data/video.mp4 --force
        """,
    )
    parser.add_argument("videos", nargs="+", help="Input video file(s)")
    parser.add_argument(
        "-o", "--output", default="./output", help="Base output directory (default: ./output)"
    )
    parser.add_argument(
        "--fps", type=float, default=0.5, help="Frame extraction FPS (default: 0.5)"
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Low VRAM mode (8-12GB GPUs): optimized chunk size, smaller SALAD batch, lower resolution",
    )
    parser.add_argument(
        "--iterations", type=int, default=30000, help="3DGS training iterations (default: 30000)"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=-1,
        help="3DGS training resolution (-1 for original, e.g. 512 or 1024 for lower memory)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force reprocessing even if outputs exist"
    )

    args = parser.parse_args()

    # Validate video paths
    valid_videos = []
    for video in args.videos:
        video_path = Path(video)
        if not video_path.exists():
            print(f"Warning: {video} not found, skipping")
            continue
        valid_videos.append(str(video_path))

    if not valid_videos:
        print("Error: No valid video files found")
        sys.exit(1)

    base_output = Path(args.output)

    # Use lower resolution in low-memory mode if not specified
    resolution = args.resolution
    if args.low_memory and resolution == -1:
        resolution = 1024

    print(f"\n{'=' * 60}")
    print(f"Processing {len(valid_videos)} video(s) into ONE 3DGS model")
    print(f"{'=' * 60}")
    for v in valid_videos:
        print(f"  - {v}")
    print(f"Output: {base_output}")
    print(f"Iterations: {args.iterations}")
    print(f"Resolution: {resolution if resolution > 0 else 'original'}")

    try:
        result = process_videos(
            valid_videos,
            base_output,
            args.fps,
            args.low_memory,
            args.iterations,
            resolution,
            args.force,
        )

        if result.get("final_ply"):
            print(f"\n{'=' * 60}")
            print("[OK] Processing complete!")
            print(f"{'=' * 60}")
            print(f"Final model: {result['final_ply']}")
        else:
            print(f"\n[WARN] No final PLY produced")
            sys.exit(1)

    except Exception as e:
        print(f"\n[ERROR] Processing failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
