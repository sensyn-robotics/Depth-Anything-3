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
import sys
from pathlib import Path

import numpy as np

from extract_frames import extract_frames
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


def process_video_stages_1_2(
    video_path: str,
    output_dir: Path,
    fps: float,
    low_memory: bool,
    force: bool,
) -> dict:
    """Process single video through stages 1-2 (frames + DA3-streaming).

    Args:
        video_path: Path to input video file
        output_dir: Output directory for this video
        fps: Frame extraction rate
        low_memory: Enable low-memory mode for 8-12GB GPUs
        force: Force reprocessing even if outputs exist

    Returns:
        Dictionary with paths to outputs
    """
    # Stage 1: Extract frames
    print("\n" + "=" * 50)
    print("Stage 1: Extracting frames")
    print("=" * 50)
    frames_dir = output_dir / "frames"
    extract_frames(video_path, str(frames_dir), fps=fps, force=force)

    # Stage 2: DA3-streaming (depth + poses)
    print("\n" + "=" * 50)
    print("Stage 2: DA3-streaming (depth + poses)")
    print("=" * 50)
    pointcloud_dir = output_dir / "pointcloud"

    # Apply low-memory settings
    if low_memory:
        chunk_size = 10
        overlap = 5
        salad_batch = 8
        loop_en = False
        proc_res = 336
    else:
        chunk_size = 60
        overlap = 30
        salad_batch = 32
        loop_en = True
        proc_res = 504

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

    return {
        "frames_dir": str(frames_dir),
        "pointcloud_path": result["pointcloud_path"],
        "poses_path": result["poses_path"],
        "intrinsics_path": result["intrinsics_path"],
    }


def merge_video_outputs(video_results: list, merged_dir: Path, force: bool) -> dict:
    """Merge outputs from multiple videos.

    Args:
        video_results: List of dicts with paths from each video
        merged_dir: Output directory for merged data
        force: Force reprocessing

    Returns:
        Dictionary with merged paths
    """
    merged_dir.mkdir(parents=True, exist_ok=True)

    merged_frames_dir = merged_dir / "frames"
    merged_pointcloud_dir = merged_dir / "pointcloud"

    # Check if already merged
    merged_pcd = merged_pointcloud_dir / "combined_pcd.ply"
    merged_poses = merged_pointcloud_dir / "camera_poses.txt"
    merged_intrinsics = merged_pointcloud_dir / "intrinsic.txt"

    if merged_pcd.exists() and merged_poses.exists() and not force:
        print(f"\nFound existing merged outputs in {merged_pointcloud_dir}")
        print("Use --force to re-merge")
        # Count frames
        n_frames = len(list(merged_frames_dir.glob("*.png"))) + len(
            list(merged_frames_dir.glob("*.jpg"))
        )
        return {
            "frames_dir": str(merged_frames_dir),
            "pointcloud_path": str(merged_pcd),
            "poses_path": str(merged_poses),
            "intrinsics_path": str(merged_intrinsics),
            "n_frames": n_frames,
        }

    print("\n" + "=" * 50)
    print("Merging outputs from all videos")
    print("=" * 50)

    # Merge frames (copy with sequential numbering)
    merged_frames_dir.mkdir(parents=True, exist_ok=True)
    merged_pointcloud_dir.mkdir(parents=True, exist_ok=True)

    # Clear existing merged frames if force
    if force:
        for f in merged_frames_dir.glob("*"):
            f.unlink()

    frame_idx = 0
    all_poses = []
    all_intrinsics = []
    all_points = []
    all_colors = []

    for i, result in enumerate(video_results):
        print(f"\nProcessing video {i + 1}/{len(video_results)}...")

        # Copy frames with sequential numbering
        frames_dir = Path(result["frames_dir"])
        frame_files = sorted(
            list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpg"))
        )
        print(f"  Copying {len(frame_files)} frames...")

        for src_frame in frame_files:
            ext = src_frame.suffix
            dst_frame = merged_frames_dir / f"frame_{frame_idx:06d}{ext}"
            if not dst_frame.exists() or force:
                shutil.copy2(src_frame, dst_frame)
            frame_idx += 1

        # Load and append poses
        poses_data = np.loadtxt(result["poses_path"])
        if poses_data.ndim == 1:
            poses_data = poses_data.reshape(1, -1)
        all_poses.append(poses_data)
        print(f"  Loaded {len(poses_data)} poses")

        # Load and append intrinsics
        intrinsics_data = np.loadtxt(result["intrinsics_path"])
        if intrinsics_data.ndim == 1:
            intrinsics_data = intrinsics_data.reshape(1, -1)
        all_intrinsics.append(intrinsics_data)

        # Load and append point cloud
        from plyfile import PlyData

        plydata = PlyData.read(result["pointcloud_path"])
        vertices = plydata["vertex"]
        points = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
        if "red" in vertices.data.dtype.names:
            colors = np.vstack(
                [vertices["red"], vertices["green"], vertices["blue"]]
            ).T
        else:
            colors = np.full((len(points), 3), 128, dtype=np.uint8)
        all_points.append(points)
        all_colors.append(colors)
        print(f"  Loaded {len(points)} points")

    # Concatenate all data
    print("\nMerging data...")
    merged_poses_data = np.vstack(all_poses)
    merged_intrinsics_data = np.vstack(all_intrinsics)
    merged_points = np.vstack(all_points)
    merged_colors = np.vstack(all_colors)

    print(f"  Total frames: {frame_idx}")
    print(f"  Total poses: {len(merged_poses_data)}")
    print(f"  Total points: {len(merged_points)}")

    # Save merged poses
    np.savetxt(merged_poses, merged_poses_data)

    # Save merged intrinsics
    np.savetxt(merged_intrinsics, merged_intrinsics_data)

    # Save merged point cloud
    from plyfile import PlyData, PlyElement

    vertex_data = np.zeros(
        len(merged_points),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertex_data["x"] = merged_points[:, 0]
    vertex_data["y"] = merged_points[:, 1]
    vertex_data["z"] = merged_points[:, 2]
    vertex_data["red"] = merged_colors[:, 0]
    vertex_data["green"] = merged_colors[:, 1]
    vertex_data["blue"] = merged_colors[:, 2]

    el = PlyElement.describe(vertex_data, "vertex")
    PlyData([el]).write(str(merged_pcd))

    print(f"\nMerged outputs saved to {merged_pointcloud_dir}")

    return {
        "frames_dir": str(merged_frames_dir),
        "pointcloud_path": str(merged_pcd),
        "poses_path": str(merged_poses),
        "intrinsics_path": str(merged_intrinsics),
        "n_frames": frame_idx,
    }


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
    # Stage 1-2: Process each video through DA3-streaming
    video_results = []
    for i, video_path in enumerate(video_paths):
        video_path = Path(video_path)
        print(f"\n{'=' * 60}")
        print(f"Video {i + 1}/{len(video_paths)}: {video_path.name}")
        print(f"{'=' * 60}")

        clear_gpu_memory()

        video_output_dir = base_output / video_path.stem
        result = process_video_stages_1_2(
            str(video_path),
            video_output_dir,
            fps,
            low_memory,
            force,
        )
        video_results.append(result)

        clear_gpu_memory()

    # Determine output directory for merged/single result
    if len(video_paths) == 1:
        # Single video - use video name
        final_output_dir = base_output / Path(video_paths[0]).stem
        merged_data = video_results[0]
    else:
        # Multiple videos - merge into combined directory
        final_output_dir = base_output / "combined"
        merged_data = merge_video_outputs(video_results, final_output_dir, force)

    # Stage 3: 3DGS training
    print("\n" + "=" * 60)
    print("Stage 3: 3DGS training (combined)")
    print("=" * 60)

    clear_gpu_memory()

    gs_dir = final_output_dir / "3dgs"
    gs_result = process_pointcloud_to_3dgs(
        pointcloud_path=merged_data["pointcloud_path"],
        poses_path=merged_data["poses_path"],
        intrinsics_path=merged_data["intrinsics_path"],
        images_dir=merged_data["frames_dir"],
        output_dir=str(gs_dir),
        iterations=iterations,
        resolution=resolution,
        backend="gsplat",
        strategy="mcmc",
        force=force,
    )

    return gs_result


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
        "--fps", type=float, default=1.0, help="Frame extraction FPS (default: 1.0)"
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Low VRAM mode (8-12GB GPUs): smaller chunks, no loop closure, lower resolution",
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
