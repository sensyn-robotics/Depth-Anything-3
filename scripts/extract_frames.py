#!/usr/bin/env python3
"""
Extract frames from video at specified FPS.

This script extracts frames from a video file using ffmpeg.

Usage:
    python scripts/extract_frames.py -i video.mp4 -o ./frames --fps 1.0

Output:
    frame_%06d.png files in the output directory
"""

import argparse
import glob
import os
import subprocess


def extract_frames(
    video_path: str,
    output_dir: str = None,
    fps: float = 1.0,
    max_frames: int = None,
    force: bool = False,
) -> list[str]:
    """
    Extract frames from video.

    Args:
        video_path: Path to input video file
        output_dir: If None, uses {video_parent}/frames/
        fps: Frame extraction rate (frames per second)
        max_frames: Maximum number of frames to extract
        force: If False and frames exist, returns existing frames

    Returns:
        List of extracted frame paths

    Resume logic:
        - If output_dir has frame_*.png files and not force, return existing
        - Otherwise extract frames
    """
    # Default output directory
    if output_dir is None:
        video_dir = os.path.dirname(os.path.abspath(video_path))
        output_dir = os.path.join(video_dir, "frames")

    os.makedirs(output_dir, exist_ok=True)

    # Check for existing frames (resume support)
    existing_frames = sorted(glob.glob(os.path.join(output_dir, "frame_*.png")))
    if existing_frames and not force:
        print(f"Found {len(existing_frames)} existing frames in {output_dir}")
        print("Use --force to re-extract frames")
        return existing_frames

    # Build ffmpeg command
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-vf",
        f"fps={fps}",
    ]

    if max_frames:
        cmd.extend(["-frames:v", str(max_frames)])

    output_pattern = os.path.join(output_dir, "frame_%06d.png")
    cmd.append(output_pattern)

    print(f"Extracting frames at {fps} FPS...")
    print(f"  Input: {video_path}")
    print(f"  Output: {output_dir}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"FFmpeg error: {result.stderr}")
        raise RuntimeError("Frame extraction failed")

    frames = sorted(glob.glob(os.path.join(output_dir, "frame_*.png")))
    print(f"Extracted {len(frames)} frames")
    return frames


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python scripts/extract_frames.py -i video.mp4 -o ./frames --fps 1.0

    # Extract at 2 FPS with max 100 frames
    python scripts/extract_frames.py -i video.mp4 --fps 2.0 --max-frames 100

    # Force re-extraction even if frames exist
    python scripts/extract_frames.py -i video.mp4 -o ./frames --force
        """,
    )
    parser.add_argument("--input", "-i", required=True, help="Input video file")
    parser.add_argument(
        "--output", "-o", default=None, help="Output directory (default: {video_dir}/frames/)"
    )
    parser.add_argument(
        "--fps", type=float, default=1.0, help="Frame extraction rate (default: 1.0)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Maximum number of frames to extract"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-extraction even if frames already exist"
    )

    args = parser.parse_args()

    frames = extract_frames(
        video_path=args.input,
        output_dir=args.output,
        fps=args.fps,
        max_frames=args.max_frames,
        force=args.force,
    )

    print(f"\nExtracted {len(frames)} frames to {args.output or 'default location'}")


if __name__ == "__main__":
    main()
