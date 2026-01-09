#!/usr/bin/env python3
"""
Convert equirectangular 360 video to 3D Gaussian Splatting model using Depth Anything 3.

This script:
1. Extracts frames from equirectangular video
2. Converts each frame to cubemap faces (front, back, left, right, up; optionally down)
3. Processes cubemap faces with DA3 for depth estimation and 3DGS

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
import torch
from tqdm import tqdm

from depth_anything_3.api import DepthAnything3


# Cubemap face configurations
FACE_CONFIGS = {
    "front":  {"yaw": 0,   "pitch": 0,   "index": 0},
    "back":   {"yaw": 180, "pitch": 0,   "index": 1},
    "left":   {"yaw": -90, "pitch": 0,   "index": 2},
    "right":  {"yaw": 90,  "pitch": 0,   "index": 3},
    "up":     {"yaw": 0,   "pitch": -90, "index": 4},
    "down":   {"yaw": 0,   "pitch": 90,  "index": 5},
}


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    fps: float = 1.0,
    max_frames: int = None,
) -> list:
    """Extract frames from video at specified FPS."""
    os.makedirs(output_dir, exist_ok=True)

    # Build ffmpeg command
    cmd = [
        "ffmpeg", "-y",
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

    # Use v360 filter to extract perspective view
    # e = equirectangular input
    # flat = rectilinear/perspective output
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
) -> dict:
    """Convert equirectangular frames to cubemap faces."""
    os.makedirs(output_dir, exist_ok=True)

    frames_data = {}

    for frame_path in tqdm(frame_paths, desc="Converting to cubemap"):
        frame_name = os.path.basename(frame_path)
        frame_num = int(re.search(r'(\d+)', frame_name).group(1))
        frames_data[frame_num] = {}

        for face in faces:
            face_idx = FACE_CONFIGS[face]["index"]
            output_filename = f"{frame_num:06d}_{face_idx}_{face}.png"
            output_path = os.path.join(output_dir, output_filename)

            if equirect_to_cubemap_ffmpeg(frame_path, output_path, face, cube_size):
                frames_data[frame_num][face] = output_path
            else:
                print(f"Warning: Failed to convert {frame_path} face {face}")

    return frames_data


def merge_ply_files(ply_files: list, output_path: str):
    """Merge multiple PLY files into one."""
    from plyfile import PlyData, PlyElement

    all_vertices = []

    for ply_file in ply_files:
        if not os.path.exists(ply_file):
            continue
        plydata = PlyData.read(ply_file)
        vertices = plydata['vertex']
        all_vertices.append(vertices.data)

    if not all_vertices:
        print("No PLY files to merge!")
        return

    merged_data = np.concatenate(all_vertices)
    merged_vertices = PlyElement.describe(merged_data, 'vertex')
    merged_ply = PlyData([merged_vertices])
    merged_ply.write(output_path)
    print(f"Merged {len(all_vertices)} PLY files -> {output_path}")


def merge_glb_point_clouds(glb_files: list, output_path: str):
    """Merge multiple GLB point cloud files."""
    import trimesh

    all_points = []
    all_colors = []

    for glb_file in glb_files:
        if not os.path.exists(glb_file):
            continue
        try:
            scene = trimesh.load(glb_file)
            if isinstance(scene, trimesh.Scene):
                for geometry in scene.geometry.values():
                    if hasattr(geometry, 'vertices'):
                        all_points.append(geometry.vertices)
                        if hasattr(geometry, 'visual') and hasattr(geometry.visual, 'vertex_colors'):
                            all_colors.append(geometry.visual.vertex_colors[:, :3])
            elif hasattr(scene, 'vertices'):
                all_points.append(scene.vertices)
        except Exception as e:
            print(f"Warning: Could not load {glb_file}: {e}")
            continue

    if not all_points:
        print("No point clouds to merge!")
        return

    merged_points = np.concatenate(all_points)

    if all_colors and len(all_colors) == len(all_points):
        merged_colors = np.concatenate(all_colors)
        cloud = trimesh.PointCloud(merged_points, colors=merged_colors)
    else:
        cloud = trimesh.PointCloud(merged_points)

    cloud.export(output_path)
    print(f"Merged {len(all_points)} point clouds -> {output_path} ({len(merged_points)} points)")


def process_chunk(
    model,
    image_paths: list,
    output_dir: str,
    chunk_idx: int,
    process_res: int,
    model_supports_gs: bool,
):
    """Process a single chunk of images."""
    chunk_dir = os.path.join(output_dir, f"chunk_{chunk_idx:04d}")
    os.makedirs(chunk_dir, exist_ok=True)

    if model_supports_gs:
        export_format = "npz-glb-gs_ply"
        export_kwargs = {}
    else:
        export_format = "npz-glb"
        export_kwargs = {}

    prediction = model.inference(
        image=image_paths,
        infer_gs=model_supports_gs,
        use_ray_pose=True,
        process_res=process_res,
        export_dir=chunk_dir,
        export_format=export_format,
        export_kwargs=export_kwargs,
        conf_thresh_percentile=30.0,
        num_max_points=500000,
    )

    return {
        "chunk_dir": chunk_dir,
        "glb_path": os.path.join(chunk_dir, "scene.glb"),
        "ply_path": os.path.join(chunk_dir, "gs_ply", "0000.ply") if model_supports_gs else None,
        "num_images": len(image_paths),
    }


def process_equirect_to_3dgs(
    input_path: str,
    output_dir: str,
    model_name: str = "depth-anything/DA3-GIANT-1.1",
    faces: list = None,
    fps: float = 1.0,
    max_frames: int = None,
    cube_size: int = 1024,
    process_res: int = 378,
    chunk_size: int = 3,
    overlap: int = 1,
    keep_temp: bool = False,
):
    """Process equirectangular video to 3DGS model."""

    if faces is None:
        # Default: exclude bottom (usually captures camera operator)
        faces = ["front", "back", "left", "right", "up"]

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = os.path.join(output_dir, "_temp")
    frames_dir = os.path.join(temp_dir, "frames")
    cubemap_dir = os.path.join(temp_dir, "cubemap")

    # Step 1: Extract frames from video
    print(f"\n{'='*50}")
    print("Step 1: Extracting frames from video")
    print(f"{'='*50}")
    frame_paths = extract_frames_from_video(
        input_path, frames_dir, fps=fps, max_frames=max_frames
    )

    if len(frame_paths) == 0:
        print("No frames extracted!")
        return

    # Step 2: Convert to cubemap
    print(f"\n{'='*50}")
    print("Step 2: Converting to cubemap faces")
    print(f"{'='*50}")
    print(f"Faces: {faces}")
    print(f"Cube size: {cube_size}x{cube_size}")

    frames_data = convert_frames_to_cubemap(
        frame_paths, cubemap_dir, faces, cube_size
    )

    # Build image list
    image_paths = []
    for frame_num in sorted(frames_data.keys()):
        for face in faces:
            if face in frames_data[frame_num]:
                image_paths.append(frames_data[frame_num][face])

    total_images = len(image_paths)
    print(f"Total cubemap images: {total_images}")

    # Step 3: Load model
    print(f"\n{'='*50}")
    print("Step 3: Loading DA3 model")
    print(f"{'='*50}")
    print(f"Model: {model_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

    model_supports_gs = "GIANT" in model_name.upper()
    if not model_supports_gs:
        print(f"Note: Model does not support Gaussian splatting. Exporting point cloud only.")

    # Step 4: Process in chunks
    print(f"\n{'='*50}")
    print("Step 4: Processing with DA3")
    print(f"{'='*50}")

    # Calculate chunks
    images_per_frame = len(faces)
    images_per_chunk = chunk_size * images_per_frame
    overlap_images = overlap * images_per_frame

    chunks = []
    start = 0
    while start < total_images:
        end = min(start + images_per_chunk, total_images)
        chunks.append((start, end))
        start = end - overlap_images
        if start >= total_images - overlap_images:
            break

    print(f"Processing {len(chunks)} chunks (chunk_size={chunk_size} frames)")

    chunk_results = []
    for i, (start, end) in enumerate(tqdm(chunks, desc="Processing chunks")):
        chunk_images = image_paths[start:end]
        print(f"\nChunk {i+1}/{len(chunks)}: {len(chunk_images)} images")

        try:
            result = process_chunk(
                model=model,
                image_paths=chunk_images,
                output_dir=output_dir,
                chunk_idx=i,
                process_res=process_res,
                model_supports_gs=model_supports_gs,
            )
            chunk_results.append(result)
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(f"  Warning: OOM on chunk {i}, skipping...")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"  Warning: Error on chunk {i}: {e}")
            continue

    print(f"\nSuccessfully processed {len(chunk_results)}/{len(chunks)} chunks")

    # Step 5: Merge results
    print(f"\n{'='*50}")
    print("Step 5: Merging results")
    print(f"{'='*50}")

    glb_files = [r["glb_path"] for r in chunk_results if r["glb_path"]]
    if glb_files:
        merged_glb_path = os.path.join(output_dir, "scene_merged.glb")
        merge_glb_point_clouds(glb_files, merged_glb_path)

    if model_supports_gs:
        ply_files = [r["ply_path"] for r in chunk_results if r["ply_path"]]
        if ply_files:
            merged_ply_dir = os.path.join(output_dir, "gs_ply_merged")
            os.makedirs(merged_ply_dir, exist_ok=True)
            merged_ply_path = os.path.join(merged_ply_dir, "merged.ply")
            merge_ply_files(ply_files, merged_ply_path)

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
        "process_res": process_res,
        "model_name": model_name,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "num_chunks": len(chunks),
        "successful_chunks": len(chunk_results),
        "model_supports_gs": model_supports_gs,
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*50}")
    print("Results saved to:")
    print(f"  - Point cloud: {output_dir}/scene_merged.glb")
    if model_supports_gs:
        print(f"  - Gaussian PLY: {output_dir}/gs_ply_merged/merged.ply")
    print(f"  - Chunks: {output_dir}/chunk_*/")
    print(f"  - Metadata: {output_dir}/metadata.json")
    print(f"{'='*50}")

    return chunk_results


def main():
    parser = argparse.ArgumentParser(
        description="Convert equirectangular 360 video to 3DGS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python scripts/equirect_to_3dgs.py -i video.mp4 -o ./output/3dgs

    # Higher quality with more frames
    python scripts/equirect_to_3dgs.py -i video.mp4 -o ./output/3dgs --fps 2.0 --cube-size 1024

    # Quick test
    python scripts/equirect_to_3dgs.py -i video.mp4 -o ./output/test --fps 0.5 --max-frames 10
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
        help="Output directory for 3DGS model"
    )
    parser.add_argument(
        "--model",
        default="depth-anything/DA3-GIANT-1.1",
        help="DA3 model (DA3-GIANT-1.1 for GS, DA3-LARGE-1.1 for point cloud only)"
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
        "--process-res",
        type=int,
        default=378,
        help="DA3 processing resolution (default: 378)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=3,
        help="Frames per chunk (default: 3)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=1,
        help="Overlapping frames between chunks (default: 1)"
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
        model_name=args.model,
        faces=faces,
        fps=args.fps,
        max_frames=args.max_frames,
        cube_size=args.cube_size,
        process_res=args.process_res,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        keep_temp=args.keep_temp,
    )


if __name__ == "__main__":
    main()
