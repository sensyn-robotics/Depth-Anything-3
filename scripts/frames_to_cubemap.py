#!/usr/bin/env python3
"""
Convert equirectangular frames to cubemap faces.

This script converts equirectangular (360) images to cubemap projections
using ffmpeg's v360 filter.

Usage:
    python scripts/frames_to_cubemap.py -i ./frames -o ./cubemap --cube-size 1024

Output:
    {frame:06d}_{face_idx}_{face_name}.png files
"""

import argparse
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

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

DEFAULT_FACES = ["front", "back", "left", "right", "up"]


def equirect_to_cubemap_all_faces(
    input_path: str,
    output_dir: str,
    frame_num: int,
    faces: list[str],
    cube_size: int = 1024,
) -> list[str]:
    """Convert equirectangular image to all cubemap faces in a single ffmpeg call.

    Uses filter_complex to decode input once and output all faces simultaneously.
    This is ~5x faster than calling ffmpeg separately for each face.

    Args:
        input_path: Path to equirectangular image
        output_dir: Output directory for cubemap faces
        frame_num: Frame number for naming output files
        faces: List of face names to generate
        cube_size: Size of each cubemap face in pixels

    Returns:
        List of output file paths
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


def process_single_frame(args) -> tuple[int, list[str]]:
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


def get_processed_frame_numbers(output_dir: str, faces: list[str]) -> set[int]:
    """Get set of frame numbers that have all faces already processed."""
    if not os.path.exists(output_dir):
        return set()

    existing_files = os.listdir(output_dir)
    frame_face_count = {}

    for f in existing_files:
        match = re.match(r'(\d+)_\d+_(\w+)\.png', f)
        if match:
            frame_num = int(match.group(1))
            face_name = match.group(2)
            if face_name in faces:
                frame_face_count[frame_num] = frame_face_count.get(frame_num, 0) + 1

    # Return frames that have all faces
    num_faces = len(faces)
    return {fn for fn, count in frame_face_count.items() if count >= num_faces}


def convert_frames_to_cubemap(
    input_dir: str,
    output_dir: str = None,
    faces: list[str] = None,
    cube_size: int = 1024,
    max_workers: int = None,
    force: bool = False,
) -> int:
    """
    Convert equirectangular frames to cubemap faces with parallel processing.

    Args:
        input_dir: Directory containing equirectangular frames (frame_*.png)
        output_dir: If None, uses {input_dir}/../cubemap/
        faces: List of faces to generate (default: front, back, left, right, up)
        cube_size: Size of each cubemap face in pixels
        max_workers: Number of parallel workers (default: auto)
        force: If False and cubemap exists for all frames, skip

    Returns:
        Total number of cubemap images (existing + newly created)

    Resume logic:
        - Check which frame numbers already have all face outputs
        - Only process missing frames
        - Return total count including existing
    """
    import glob
    import multiprocessing

    if faces is None:
        faces = DEFAULT_FACES.copy()

    # Default output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(input_dir)), "cubemap")

    os.makedirs(output_dir, exist_ok=True)

    # Get input frames
    frame_paths = sorted(glob.glob(os.path.join(input_dir, "frame_*.png")))
    if not frame_paths:
        raise ValueError(f"No frame_*.png files found in {input_dir}")

    # Auto-detect worker count (cap at 8 to avoid overwhelming system)
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 8)

    # Check which frames are already processed (resume support)
    processed_frames = set() if force else get_processed_frame_numbers(output_dir, faces)

    # Prepare task arguments for unprocessed frames only
    task_args = []
    for frame_path in frame_paths:
        frame_name = os.path.basename(frame_path)
        frame_num = int(re.search(r'(\d+)', frame_name).group(1))
        if frame_num not in processed_frames:
            task_args.append((frame_path, output_dir, faces, cube_size, frame_num))

    # Report resume status
    num_existing = len(processed_frames)
    num_to_process = len(task_args)

    if num_existing > 0 and not force:
        print(f"Found {num_existing} frames already converted (use --force to reconvert)")

    if num_to_process == 0:
        print("All frames already converted, skipping")
        return num_existing * len(faces)

    print(f"Converting {num_to_process} frames to cubemap ({max_workers} workers)")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Faces: {faces}")
    print(f"  Cube size: {cube_size}x{cube_size}")

    new_images = 0

    # Process frames in parallel with progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_frame, args): args[4] for args in task_args}

        with tqdm(total=num_to_process, desc="Converting to cubemap") as pbar:
            for future in as_completed(futures):
                frame_num, output_paths = future.result()
                new_images += len(output_paths)
                pbar.update(1)

    total_images = num_existing * len(faces) + new_images
    print(f"Total cubemap images: {total_images}")
    return total_images


def main():
    parser = argparse.ArgumentParser(
        description="Convert equirectangular frames to cubemap faces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python scripts/frames_to_cubemap.py -i ./frames -o ./cubemap

    # Custom faces and size
    python scripts/frames_to_cubemap.py -i ./frames -o ./cubemap \\
        --faces front,back,left,right --cube-size 512

    # Force reconvert all frames
    python scripts/frames_to_cubemap.py -i ./frames -o ./cubemap --force
        """
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input directory containing equirectangular frames"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory (default: {input_dir}/../cubemap/)"
    )
    parser.add_argument(
        "--faces",
        default="front,back,left,right,up",
        help="Comma-separated list of faces to generate (default: excludes 'down')"
    )
    parser.add_argument(
        "--cube-size",
        type=int,
        default=1024,
        help="Cubemap face size in pixels (default: 1024)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reconvert even if cubemap faces already exist"
    )

    args = parser.parse_args()

    faces = [f.strip() for f in args.faces.split(",")]

    total = convert_frames_to_cubemap(
        input_dir=args.input,
        output_dir=args.output,
        faces=faces,
        cube_size=args.cube_size,
        max_workers=args.workers,
        force=args.force,
    )

    print(f"\nCreated {total} cubemap images")


if __name__ == "__main__":
    main()
