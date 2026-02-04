#!/usr/bin/env python3
"""
One-liner video to 3DGS processing.

This script processes regular videos through the full pipeline:
1. Extract frames from video
2. DA3-streaming for depth estimation and camera poses
3. 3DGS training with gsplat

Usage:
    # Single video
    python scripts/run_3dgs.py data/video.mp4

    # Multiple videos
    python scripts/run_3dgs.py data/*.mp4

    # With options
    python scripts/run_3dgs.py data/*.mp4 --fps 1.0 --low-memory --iterations 30000

Output:
    output/{video_name}/
        ├── frames/
        ├── pointcloud/
        │   ├── combined_pcd.ply
        │   ├── camera_poses.txt
        │   └── intrinsic.txt
        └── 3dgs/
            └── model/point_cloud/iteration_XXXXX/point_cloud.ply
"""

import argparse
import sys
from pathlib import Path

from extract_frames import extract_frames
from frames_to_pointcloud import process_frames_to_pointcloud
from pointcloud_to_3dgs import process_pointcloud_to_3dgs


def process_video(
    video_path: str,
    output_dir: Path,
    fps: float,
    low_memory: bool,
    iterations: int,
    resolution: int,
    force: bool,
) -> dict:
    """Process single video through full pipeline.

    Args:
        video_path: Path to input video file
        output_dir: Output directory for this video
        fps: Frame extraction rate
        low_memory: Enable low-memory mode for 8-12GB GPUs
        iterations: 3DGS training iterations
        resolution: Image resolution for 3DGS training (-1 for original)
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

    # Stage 3: 3DGS training with gsplat
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


def main():
    parser = argparse.ArgumentParser(
        description="Video to 3DGS one-liner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single video
    python scripts/run_3dgs.py data/video.mp4

    # Multiple videos (processes sequentially)
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

    base_output = Path(args.output)

    results = []
    for video in args.videos:
        video_path = Path(video)
        if not video_path.exists():
            print(f"Warning: {video} not found, skipping")
            continue

        # Clear GPU memory before each video
        try:
            import torch
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

        output_dir = base_output / video_path.stem
        print(f"\n{'=' * 60}")
        print(f"Processing: {video_path.name}")
        print(f"Output: {output_dir}")
        print(f"{'=' * 60}")

        try:
            # Use lower resolution in low-memory mode if not specified
            resolution = args.resolution
            if args.low_memory and resolution == -1:
                resolution = 1024  # Default to 1024 in low-memory mode

            result = process_video(
                str(video_path),
                output_dir,
                args.fps,
                args.low_memory,
                args.iterations,
                resolution,
                args.force,
            )

            if result.get("final_ply"):
                print(f"\n[OK] Complete: {result['final_ply']}")
                results.append((video_path.name, result["final_ply"]))
            else:
                print(f"\n[WARN] No final PLY produced for {video_path.name}")
                results.append((video_path.name, None))

        except Exception as e:
            print(f"\n[ERROR] Failed to process {video_path.name}: {e}")
            results.append((video_path.name, None))
        finally:
            # Clear GPU memory after each video
            try:
                import torch
                import gc

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception:
                pass

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    for name, ply_path in results:
        status = "[OK]" if ply_path else "[FAILED]"
        print(f"  {status} {name}")
        if ply_path:
            print(f"        -> {ply_path}")

    # Exit with error if any failed
    if any(ply is None for _, ply in results):
        sys.exit(1)

    print(f"\nAll {len(results)} video(s) processed successfully!")


if __name__ == "__main__":
    main()
