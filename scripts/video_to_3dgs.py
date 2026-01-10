#!/usr/bin/env python3
"""
Process normal video to 3DGS (3D Gaussian Splatting).

Outputs:
  - gs_video/*.mp4  : Novel view synthesis rendered video
  - gs_ply/*.ply    : 3D Gaussian Splatting PLY file

Usage:
  # First, start the backend in a separate terminal (one-time):
  da3 backend --model-dir depth-anything/DA3-GIANT

  # Then run this script:
  python scripts/video_to_3dgs.py /path/to/video.mp4

  # Or with custom output directory:
  python scripts/video_to_3dgs.py /path/to/video.mp4 -o ./my_output

  # Adjust FPS (default: 1.0):
  python scripts/video_to_3dgs.py /path/to/video.mp4 --fps 2.0
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import requests


def wait_for_task(backend_url: str, task_id: str, poll_interval: float = 5.0):
    """Poll backend until task is completed or failed."""
    status_url = f"{backend_url}/task/{task_id}"

    while True:
        try:
            resp = requests.get(status_url, timeout=10)
            if resp.status_code == 404:
                print(f"Task {task_id} not found")
                return False
            resp.raise_for_status()

            data = resp.json()
            status = data.get("status", "unknown")
            message = data.get("message", "")
            progress = data.get("progress")

            if progress is not None:
                print(f"\r[{status}] {progress*100:.1f}% - {message}", end="", flush=True)
            else:
                print(f"\r[{status}] {message}", end="", flush=True)

            if status == "completed":
                print()
                return True
            elif status == "failed":
                print()
                print(f"Task failed: {message}")
                return False

            time.sleep(poll_interval)

        except requests.RequestException as e:
            print(f"\nError checking task status: {e}")
            time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser(
        description="Process video to 3DGS using Depth Anything 3"
    )
    parser.add_argument("video", type=str, help="Input video file path")
    parser.add_argument(
        "-o", "--output", type=str, default=None, help="Output directory (default: ./output/<video_name>)"
    )
    parser.add_argument(
        "--fps", type=float, default=0.5, help="Frame extraction FPS (default: 0.5)"
    )
    parser.add_argument(
        "--process-res", type=int, default=378, help="Processing resolution (default: 378, use 504 for 24GB+ GPU)"
    )
    parser.add_argument(
        "--backend-url", type=str, default="http://localhost:8008", help="Backend URL"
    )
    parser.add_argument(
        "--no-backend", action="store_true", help="Run without backend (slower, loads model each time)"
    )
    parser.add_argument(
        "--model-dir", type=str, default="depth-anything/DA3-GIANT",
        help="Model directory (default: depth-anything/DA3-GIANT)"
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path("./output") / video_path.stem

    # Build command
    cmd = [
        "da3", "video", str(video_path),
        "--export-format", "glb-gs_ply-gs_video",
        "--export-dir", str(output_dir),
        "--fps", str(args.fps),
        "--process-res", str(args.process_res),
        "--auto-cleanup",
    ]

    if args.no_backend:
        cmd.extend(["--model-dir", args.model_dir])
    else:
        cmd.extend(["--use-backend", "--backend-url", args.backend_url])

    print(f"Processing: {video_path}")
    print(f"Output dir: {output_dir}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)

    try:
        # Capture output to extract task ID
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        output = result.stdout + result.stderr
        print(output)

        # If using backend, wait for task completion
        if not args.no_backend:
            match = re.search(r"Task ID: ([a-f0-9-]+)", output)
            if match:
                task_id = match.group(1)
                print(f"Waiting for task {task_id} to complete...")
                success = wait_for_task(args.backend_url, task_id)
                if not success:
                    sys.exit(1)

        print("-" * 50)
        print(f"Done! Results saved to: {output_dir}")
        print(f"  - 3DGS PLY:   {output_dir}/gs_ply/")
        print(f"  - 3DGS Video: {output_dir}/gs_video/")
        print(f"  - Point Cloud: {output_dir}/scene.glb")

    except subprocess.CalledProcessError as e:
        print(f"Error running da3: {e}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        if not args.no_backend:
            print("\nHint: Make sure the backend is running:")
            print(f"  da3 backend --model-dir {args.model_dir}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: 'da3' command not found. Make sure Depth Anything 3 is installed.")
        print("  pip install -e .")
        sys.exit(1)


if __name__ == "__main__":
    main()
