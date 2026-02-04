#!/usr/bin/env python3
"""
Complete pipeline from equirectangular 360 video to 3D Gaussian Splatting.

This script orchestrates the full pipeline:
1. Extract frames from video
2. Convert frames to cubemap faces
3. Process through DA3-streaming for depth and poses
4. Train 3D Gaussian Splatting (optional)

Usage:
    python scripts/equirect_to_3dgs.py -i video.mp4 -o ./output

Output:
    - frames/: Extracted video frames
    - cubemap/: Cubemap face images
    - pointcloud/: DA3-streaming output (point cloud + poses)
    - 3dgs/: Trained 3DGS model (if --train-3dgs enabled)
"""

import argparse
import json
import os
import shutil
from pathlib import Path

# Import from modular scripts
from extract_frames import extract_frames
from frames_to_cubemap import convert_frames_to_cubemap, DEFAULT_FACES
from cubemap_to_pointcloud import process_cubemap_to_pointcloud
from pointcloud_to_3dgs import process_pointcloud_to_3dgs


def process_equirect_to_3dgs(
    input_path: str,
    output_dir: str = None,
    fps: float = 1.0,
    max_frames: int = None,
    faces: list[str] = None,
    cube_size: int = 1024,
    chunk_size: int = 60,
    overlap: int = 30,
    loop_enable: bool = True,
    salad_batch_size: int = 32,
    process_res: int = 504,
    train_3dgs: bool = False,
    train_iterations: int = 30000,
    keep_temp: bool = True,
    force: bool = False,
    gs_backend: str = "gsplat",
    gs_strategy: str = "mcmc",
) -> dict:
    """
    Complete pipeline from equirectangular video to 3DGS.

    Args:
        input_path: Path to input video file
        output_dir: If None, uses ./output/{video_name}/
        fps: Frame extraction rate
        max_frames: Maximum frames to extract
        faces: Cubemap faces to generate
        cube_size: Cubemap face size in pixels
        chunk_size: DA3-streaming chunk size
        overlap: Overlap between chunks
        loop_enable: Enable loop closure detection
        salad_batch_size: SALAD batch size
        process_res: DA3 processing resolution
        train_3dgs: Whether to train 3DGS model
        train_iterations: Number of 3DGS training iterations
        keep_temp: Keep intermediate files
        force: If False, each stage checks for existing outputs

    Returns:
        Dictionary with paths to all outputs

    Resume logic:
        - Stage 1: Skip if frames/ has frames
        - Stage 2: Skip if cubemap/ has all faces for all frames
        - Stage 3: Skip if combined_pcd.ply exists
        - Stage 4: Skip if 3dgs/point_cloud/iteration_X exists
        - Each stage function handles its own resume
    """
    if faces is None:
        faces = DEFAULT_FACES.copy()

    # Default output directory
    if output_dir is None:
        video_name = Path(input_path).stem
        output_dir = os.path.join("./output", video_name)

    os.makedirs(output_dir, exist_ok=True)

    # Define output directories
    frames_dir = os.path.join(output_dir, "frames")
    cubemap_dir = os.path.join(output_dir, "cubemap")
    pointcloud_dir = os.path.join(output_dir, "pointcloud")
    gs_dir = os.path.join(output_dir, "3dgs")

    results = {
        "input_path": input_path,
        "output_dir": output_dir,
    }

    # Stage 1: Extract frames
    print(f"\n{'='*60}")
    print("Stage 1: Extracting frames from video")
    print(f"{'='*60}")

    frame_paths = extract_frames(
        video_path=input_path,
        output_dir=frames_dir,
        fps=fps,
        max_frames=max_frames,
        force=force,
    )

    if len(frame_paths) == 0:
        raise RuntimeError("No frames extracted!")

    results["frames_dir"] = frames_dir
    results["num_frames"] = len(frame_paths)

    # Stage 2: Convert to cubemap
    print(f"\n{'='*60}")
    print("Stage 2: Converting to cubemap faces")
    print(f"{'='*60}")
    print(f"Faces: {faces}")
    print(f"Cube size: {cube_size}x{cube_size}")

    num_cubemap = convert_frames_to_cubemap(
        input_dir=frames_dir,
        output_dir=cubemap_dir,
        faces=faces,
        cube_size=cube_size,
        force=force,
    )

    results["cubemap_dir"] = cubemap_dir
    results["num_cubemap_images"] = num_cubemap

    # Stage 3: Run DA3-streaming
    print(f"\n{'='*60}")
    print("Stage 3: Running DA3-streaming (depth estimation + alignment)")
    print(f"{'='*60}")

    pointcloud_result = process_cubemap_to_pointcloud(
        input_dir=cubemap_dir,
        output_dir=pointcloud_dir,
        chunk_size=chunk_size,
        overlap=overlap,
        loop_enable=loop_enable,
        salad_batch_size=salad_batch_size,
        process_res=process_res,
        force=force,
    )

    results["pointcloud_path"] = pointcloud_result["pointcloud_path"]
    results["poses_path"] = pointcloud_result["poses_path"]
    results["intrinsics_path"] = pointcloud_result["intrinsics_path"]

    # Stage 4: Train 3DGS (optional)
    if train_3dgs:
        print(f"\n{'='*60}")
        print("Stage 4: Training 3D Gaussian Splatting")
        print(f"{'='*60}")

        gs_result = process_pointcloud_to_3dgs(
            pointcloud_path=pointcloud_result["pointcloud_path"],
            poses_path=pointcloud_result["poses_path"],
            intrinsics_path=pointcloud_result["intrinsics_path"],
            images_dir=cubemap_dir,
            output_dir=gs_dir,
            iterations=train_iterations,
            force=force,
            backend=gs_backend,
            strategy=gs_strategy,
        )

        results["colmap_dir"] = gs_result["colmap_dir"]
        results["model_dir"] = gs_result["model_dir"]
        results["final_ply"] = gs_result["final_ply"]
    else:
        print(f"\n{'='*60}")
        print("Stage 4: Skipped (use --train-3dgs to enable)")
        print(f"{'='*60}")

    # Clean up temp files if not keeping
    if not keep_temp:
        print("\nCleaning up intermediate files...")
        # Keep pointcloud outputs but remove streaming internals
        streaming_output = os.path.join(pointcloud_dir, "streaming_output")
        if os.path.exists(streaming_output):
            # Keep important outputs, remove temp data
            temp_dirs = ["depth", "conf", "chunk_*"]
            for pattern in temp_dirs:
                import glob
                for path in glob.glob(os.path.join(streaming_output, pattern)):
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)

    # Save metadata
    metadata = {
        "input_video": input_path,
        "faces": faces,
        "fps": fps,
        "max_frames": max_frames,
        "cube_size": cube_size,
        "num_frames": len(frame_paths),
        "num_cubemap_images": num_cubemap,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "loop_enable": loop_enable,
        "salad_batch_size": salad_batch_size,
        "process_res": process_res,
        "train_3dgs": train_3dgs,
        "train_iterations": train_iterations if train_3dgs else None,
        "gs_backend": gs_backend if train_3dgs else None,
        "gs_strategy": gs_strategy if train_3dgs else None,
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("Pipeline complete!")
    print(f"{'='*60}")
    print(f"\nOutputs:")
    print(f"  Frames:      {frames_dir}/")
    print(f"  Cubemap:     {cubemap_dir}/")
    print(f"  Point cloud: {results['pointcloud_path']}")
    print(f"  Poses:       {results['poses_path']}")
    print(f"  Intrinsics:  {results['intrinsics_path']}")
    if train_3dgs and results.get("final_ply"):
        print(f"  3DGS model:  {results['final_ply']}")
    print(f"  Metadata:    {metadata_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Complete pipeline from equirectangular 360 video to 3DGS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (point cloud only)
    python scripts/equirect_to_3dgs.py -i video.mp4 -o ./output

    # With 3DGS training
    python scripts/equirect_to_3dgs.py -i video.mp4 -o ./output --train-3dgs

    # Low VRAM GPUs (8-12GB)
    python scripts/equirect_to_3dgs.py -i video.mp4 -o ./output --low-memory

    # Quick test
    python scripts/equirect_to_3dgs.py -i video.mp4 -o ./output \\
        --fps 0.5 --max-frames 30 --low-memory

    # Higher quality
    python scripts/equirect_to_3dgs.py -i video.mp4 -o ./output \\
        --fps 2.0 --cube-size 1024

    # Full pipeline with custom settings
    python scripts/equirect_to_3dgs.py -i video.mp4 -o ./output \\
        --fps 1.0 --cube-size 1024 \\
        --chunk-size 60 --overlap 30 \\
        --train-3dgs --train-iterations 30000
        """
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input equirectangular video file (MP4)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory (default: ./output/{video_name}/)"
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
        "--faces",
        default="front,back,left,right,up",
        help="Cubemap faces to process (default: excludes 'down')"
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
        help="Overlap between chunks (default: 30)"
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="Disable loop closure detection"
    )
    parser.add_argument(
        "--salad-batch-size",
        type=int,
        default=32,
        help="SALAD loop closure batch size (default: 32)"
    )
    parser.add_argument(
        "--process-res",
        type=int,
        default=504,
        help="DA3 processing resolution (default: 504)"
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Enable low-memory mode for 8-12GB VRAM GPUs"
    )
    parser.add_argument(
        "--train-3dgs",
        action="store_true",
        help="Train 3D Gaussian Splatting model after point cloud generation"
    )
    parser.add_argument(
        "--train-iterations",
        type=int,
        default=30000,
        help="Number of 3DGS training iterations (default: 30000)"
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep all intermediate files"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing all stages even if outputs exist"
    )
    parser.add_argument(
        "--gs-backend",
        choices=["gsplat", "original"],
        default="gsplat",
        help="3DGS training backend: 'gsplat' (default, no external repo) or 'original' (external gaussian-splatting repo)"
    )
    parser.add_argument(
        "--gs-strategy",
        choices=["mcmc", "default"],
        default="mcmc",
        help="Densification strategy for gsplat backend: 'mcmc' (default) or 'default' (ADC)"
    )

    args = parser.parse_args()

    # Apply low-memory defaults
    chunk_size = args.chunk_size
    overlap = args.overlap
    salad_batch_size = args.salad_batch_size
    loop_enable = not args.no_loop
    process_res = args.process_res

    if args.low_memory:
        print("Low-memory mode enabled: using conservative settings for 8-12GB VRAM GPUs")
        if args.chunk_size == 60:
            chunk_size = 10
        if args.overlap == 30:
            overlap = 5
        if args.salad_batch_size == 32:
            salad_batch_size = 8
        if not args.no_loop:
            loop_enable = False
        if args.process_res == 504:
            process_res = 336

    faces = [f.strip() for f in args.faces.split(",")]

    process_equirect_to_3dgs(
        input_path=args.input,
        output_dir=args.output,
        fps=args.fps,
        max_frames=args.max_frames,
        faces=faces,
        cube_size=args.cube_size,
        chunk_size=chunk_size,
        overlap=overlap,
        loop_enable=loop_enable,
        salad_batch_size=salad_batch_size,
        process_res=process_res,
        train_3dgs=args.train_3dgs,
        train_iterations=args.train_iterations,
        keep_temp=args.keep_temp,
        force=args.force,
        gs_backend=args.gs_backend,
        gs_strategy=args.gs_strategy,
    )


if __name__ == "__main__":
    main()
