#!/usr/bin/env python3
"""
Process cubemap images from 360 video into 3D Gaussian Splatting model using Depth Anything 3.
Supports chunked processing for large datasets with limited GPU memory.

Usage:
    python scripts/process_cubemap_to_3dgs.py \
        --input-dir /path/to/cubemap/images \
        --output-dir ./output/3dgs \
        --frame-step 5 \
        --faces front,left,right,back \
        --chunk-size 5
"""

import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from depth_anything_3.api import DepthAnything3


def parse_filename(filename: str) -> tuple:
    """Parse cubemap filename to extract frame number and face name."""
    match = re.match(r'(\d+)_(\d+)_(\w+)\.png', filename)
    if match:
        frame_num = int(match.group(1))
        face_idx = int(match.group(2))
        face_name = match.group(3)
        return frame_num, face_idx, face_name
    return None, None, None


def collect_frames(input_dir: str, faces: list, frame_step: int = 1, max_frames: int = None) -> dict:
    """Collect and organize cubemap frames."""
    all_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))

    frames = {}
    for filepath in all_files:
        filename = os.path.basename(filepath)
        frame_num, face_idx, face_name = parse_filename(filename)

        if frame_num is None:
            continue
        if face_name not in faces:
            continue

        if frame_num not in frames:
            frames[frame_num] = {}
        frames[frame_num][face_name] = filepath

    frame_numbers = sorted(frames.keys())[::frame_step]
    if max_frames:
        frame_numbers = frame_numbers[:max_frames]

    return {fn: frames[fn] for fn in frame_numbers}


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

    # Concatenate all vertices
    merged_data = np.concatenate(all_vertices)

    # Create merged PLY
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
                        if hasattr(geometry, 'colors') and geometry.colors is not None:
                            all_colors.append(geometry.colors[:, :3])
                        elif hasattr(geometry, 'visual') and hasattr(geometry.visual, 'vertex_colors'):
                            all_colors.append(geometry.visual.vertex_colors[:, :3])
            elif hasattr(scene, 'vertices'):
                all_points.append(scene.vertices)
                if hasattr(scene, 'colors') and scene.colors is not None:
                    all_colors.append(scene.colors[:, :3])
        except Exception as e:
            print(f"Warning: Could not load {glb_file}: {e}")
            continue

    if not all_points:
        print("No point clouds to merge!")
        return

    merged_points = np.concatenate(all_points)

    # Create point cloud
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
        num_max_points=500000,  # Limit per chunk
    )

    return {
        "chunk_dir": chunk_dir,
        "glb_path": os.path.join(chunk_dir, "scene.glb"),
        "ply_path": os.path.join(chunk_dir, "gs_ply", "0000.ply") if model_supports_gs else None,
        "num_images": len(image_paths),
        "depth_shape": prediction.depth.shape if prediction.depth is not None else None,
    }


def process_cubemap_to_3dgs(
    input_dir: str,
    output_dir: str,
    model_name: str = "depth-anything/DA3-GIANT-1.1",
    faces: list = None,
    frame_step: int = 1,
    max_frames: int = None,
    process_res: int = 504,
    chunk_size: int = 5,
    overlap: int = 1,
):
    """Process cubemap images into 3DGS model with chunked processing."""

    if faces is None:
        faces = ["front", "back", "left", "right"]

    print(f"Loading model: {model_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

    # Check if model supports GS
    model_supports_gs = "GIANT" in model_name.upper()
    if not model_supports_gs:
        print(f"Note: Model {model_name} does not support Gaussian splatting. Exporting point cloud only.")

    # Collect frames
    print(f"Collecting frames from {input_dir}")
    frames = collect_frames(input_dir, faces, frame_step, max_frames)
    print(f"Found {len(frames)} frames with {len(faces)} faces each")

    if len(frames) == 0:
        print("No frames found!")
        return

    # Build image list
    image_paths = []
    frame_indices = []
    face_names = []

    for frame_num in sorted(frames.keys()):
        for face in faces:
            if face in frames[frame_num]:
                image_paths.append(frames[frame_num][face])
                frame_indices.append(frame_num)
                face_names.append(face)

    total_images = len(image_paths)
    print(f"Total images to process: {total_images}")

    # Get image size
    from PIL import Image
    with Image.open(image_paths[0]) as img:
        img_size = img.size[0]
    print(f"Image size: {img_size}x{img_size}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Calculate chunks
    # Each chunk should have `chunk_size` frames worth of images
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

    print(f"\nProcessing in {len(chunks)} chunks (chunk_size={chunk_size} frames, overlap={overlap})")
    print(f"Images per chunk: ~{images_per_chunk}")

    # Process each chunk
    chunk_results = []
    for i, (start, end) in enumerate(tqdm(chunks, desc="Processing chunks")):
        chunk_images = image_paths[start:end]
        print(f"\nChunk {i+1}/{len(chunks)}: images {start}-{end} ({len(chunk_images)} images)")

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

            # Clear GPU cache between chunks
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(f"  Warning: OOM on chunk {i}, skipping...")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"  Warning: Error on chunk {i}: {e}")
            continue

    print(f"\nSuccessfully processed {len(chunk_results)}/{len(chunks)} chunks")

    # Merge results
    print("\nMerging results...")

    # Merge GLB files
    glb_files = [r["glb_path"] for r in chunk_results if r["glb_path"]]
    if glb_files:
        merged_glb_path = os.path.join(output_dir, "scene_merged.glb")
        merge_glb_point_clouds(glb_files, merged_glb_path)

    # Merge PLY files (for GS)
    if model_supports_gs:
        ply_files = [r["ply_path"] for r in chunk_results if r["ply_path"]]
        if ply_files:
            merged_ply_dir = os.path.join(output_dir, "gs_ply_merged")
            os.makedirs(merged_ply_dir, exist_ok=True)
            merged_ply_path = os.path.join(merged_ply_dir, "merged.ply")
            merge_ply_files(ply_files, merged_ply_path)

    # Generate merged video (if GS)
    if model_supports_gs and len(chunk_results) > 0:
        print("\nGenerating merged GS video...")
        try:
            merged_ply_path = os.path.join(output_dir, "gs_ply_merged", "merged.ply")
            if os.path.exists(merged_ply_path):
                # Re-run inference on a subset to generate video from merged PLY
                # This is a simplified approach - just note the path
                print(f"  Merged PLY available at: {merged_ply_path}")
                print("  To render video, load this PLY in a GS viewer like SuperSplat")
        except Exception as e:
            print(f"  Warning: Could not generate merged video: {e}")

    # Save metadata
    metadata = {
        "input_dir": input_dir,
        "faces": faces,
        "frame_step": frame_step,
        "num_frames": len(frames),
        "num_images": total_images,
        "image_size": img_size,
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
    print(f"Results saved to {output_dir}")
    print(f"  - Merged point cloud: {output_dir}/scene_merged.glb")
    if model_supports_gs:
        print(f"  - Merged Gaussian PLY: {output_dir}/gs_ply_merged/merged.ply")
    print(f"  - Individual chunks: {output_dir}/chunk_*/")
    print(f"  - Metadata: {output_dir}/metadata.json")
    print(f"{'='*50}")

    return chunk_results


def main():
    parser = argparse.ArgumentParser(description="Process cubemap images to 3DGS with chunked processing")
    parser.add_argument(
        "--input-dir", "-i",
        required=True,
        help="Directory containing cubemap images"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./output/cubemap_3dgs",
        help="Output directory for 3DGS model"
    )
    parser.add_argument(
        "--model",
        default="depth-anything/DA3-GIANT-1.1",
        help="DA3 model to use (DA3-GIANT-1.1 for GS, DA3-LARGE-1.1 for point cloud only)"
    )
    parser.add_argument(
        "--faces",
        default="front,left,right,back",
        help="Comma-separated list of faces to process"
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=5,
        help="Process every N-th frame (default: 5)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process"
    )
    parser.add_argument(
        "--process-res",
        type=int,
        default=378,
        help="Processing resolution (default: 378, lower for less memory)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=3,
        help="Number of frames per chunk (default: 3)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=1,
        help="Number of overlapping frames between chunks (default: 1)"
    )

    args = parser.parse_args()

    faces = [f.strip() for f in args.faces.split(",")]

    process_cubemap_to_3dgs(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        faces=faces,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
        process_res=args.process_res,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()
