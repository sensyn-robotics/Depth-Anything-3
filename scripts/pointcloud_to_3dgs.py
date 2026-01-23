#!/usr/bin/env python3
"""
Convert point cloud and poses to 3D Gaussian Splatting format and train.

This script:
1. Converts DA3-streaming output to COLMAP format
2. Trains 3D Gaussian Splatting model

Usage:
    python scripts/pointcloud_to_3dgs.py \\
        -p ./pointcloud/combined_pcd.ply \\
        --poses ./pointcloud/camera_poses.txt \\
        --intrinsics ./pointcloud/intrinsic.txt \\
        --images-dir ./cubemap \\
        -o ./3dgs

Output:
    - colmap/: COLMAP format data
    - model/: Trained 3DGS model
    - point_cloud/iteration_XXXXX/point_cloud.ply
"""

import argparse
import glob
import os
import re
import shutil
import struct
import subprocess
import sys
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# COLMAP data structures
CameraModel = namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
CAMERA_MODELS = {
    "PINHOLE": CameraModel(1, "PINHOLE", 4),
}


@dataclass
class DA3StreamingOutput:
    """Container for DA3-streaming output data."""
    pointcloud_path: str
    poses_path: str
    intrinsics_path: str
    images_dir: str

    # Loaded data
    points: np.ndarray = None  # [N, 3] XYZ
    colors: np.ndarray = None  # [N, 3] RGB (0-255)
    poses: np.ndarray = None   # [M, 4, 4] C2W matrices
    intrinsics: tuple = None   # (fx, fy, cx, cy)
    image_paths: list = None   # List of image paths


def load_da3_output(
    pointcloud_path: str,
    poses_path: str,
    intrinsics_path: str,
    images_dir: str,
) -> DA3StreamingOutput:
    """Load DA3-streaming output files.

    Args:
        pointcloud_path: Path to combined_pcd.ply
        poses_path: Path to camera_poses.txt
        intrinsics_path: Path to intrinsic.txt
        images_dir: Directory containing input images

    Returns:
        DA3StreamingOutput with loaded data
    """
    output = DA3StreamingOutput(
        pointcloud_path=pointcloud_path,
        poses_path=poses_path,
        intrinsics_path=intrinsics_path,
        images_dir=images_dir,
    )

    # Load point cloud
    print(f"Loading point cloud from {pointcloud_path}...")
    try:
        from plyfile import PlyData
        plydata = PlyData.read(pointcloud_path)
        vertices = plydata['vertex']

        output.points = np.vstack([
            vertices['x'],
            vertices['y'],
            vertices['z']
        ]).T

        if 'red' in vertices.data.dtype.names:
            output.colors = np.vstack([
                vertices['red'],
                vertices['green'],
                vertices['blue']
            ]).T.astype(np.uint8)
        else:
            output.colors = np.full((len(output.points), 3), 128, dtype=np.uint8)

        print(f"  Loaded {len(output.points)} points")
    except Exception as e:
        raise RuntimeError(f"Failed to load point cloud: {e}")

    # Load camera poses (4x4 C2W matrices)
    print(f"Loading poses from {poses_path}...")
    try:
        poses_data = np.loadtxt(poses_path)
        num_poses = len(poses_data) // 4
        output.poses = poses_data.reshape(num_poses, 4, 4)
        print(f"  Loaded {num_poses} camera poses")
    except Exception as e:
        raise RuntimeError(f"Failed to load poses: {e}")

    # Load intrinsics (fx, fy, cx, cy)
    print(f"Loading intrinsics from {intrinsics_path}...")
    try:
        intrinsics_data = np.loadtxt(intrinsics_path)
        output.intrinsics = tuple(intrinsics_data[:4])
        print(f"  Intrinsics: fx={output.intrinsics[0]:.2f}, fy={output.intrinsics[1]:.2f}, "
              f"cx={output.intrinsics[2]:.2f}, cy={output.intrinsics[3]:.2f}")
    except Exception as e:
        raise RuntimeError(f"Failed to load intrinsics: {e}")

    # Find input images
    print(f"Finding images in {images_dir}...")
    image_patterns = ["*.png", "*.jpg", "*.jpeg"]
    all_images = []
    for pattern in image_patterns:
        all_images.extend(glob.glob(os.path.join(images_dir, pattern)))
    output.image_paths = sorted(all_images)
    print(f"  Found {len(output.image_paths)} images")

    # Verify counts match
    if len(output.image_paths) != num_poses:
        print(f"  Warning: Number of images ({len(output.image_paths)}) != number of poses ({num_poses})")
        # Truncate to minimum
        min_count = min(len(output.image_paths), num_poses)
        output.image_paths = output.image_paths[:min_count]
        output.poses = output.poses[:min_count]
        print(f"  Using {min_count} images/poses")

    return output


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
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


def write_cameras_binary(cameras: dict, path: str):
    """Write cameras.bin in COLMAP binary format."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for camera_id, camera in cameras.items():
            f.write(struct.pack("<I", camera_id))
            f.write(struct.pack("<i", camera["model_id"]))
            f.write(struct.pack("<Q", camera["width"]))
            f.write(struct.pack("<Q", camera["height"]))
            for param in camera["params"]:
                f.write(struct.pack("<d", param))


def write_images_binary(images: dict, path: str):
    """Write images.bin in COLMAP binary format."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for image_id, image in images.items():
            f.write(struct.pack("<I", image_id))
            f.write(struct.pack("<d", image["qw"]))
            f.write(struct.pack("<d", image["qx"]))
            f.write(struct.pack("<d", image["qy"]))
            f.write(struct.pack("<d", image["qz"]))
            f.write(struct.pack("<d", image["tx"]))
            f.write(struct.pack("<d", image["ty"]))
            f.write(struct.pack("<d", image["tz"]))
            f.write(struct.pack("<I", image["camera_id"]))
            # Write image name as null-terminated string
            name_bytes = image["name"].encode("utf-8") + b"\x00"
            f.write(name_bytes)
            # Write empty 2D points (no feature matching)
            f.write(struct.pack("<Q", 0))


def write_points3d_binary(points: np.ndarray, colors: np.ndarray, path: str):
    """Write points3D.bin in COLMAP binary format."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(points)))
        for i, (point, color) in enumerate(zip(points, colors)):
            point_id = i + 1
            f.write(struct.pack("<Q", point_id))
            f.write(struct.pack("<d", point[0]))
            f.write(struct.pack("<d", point[1]))
            f.write(struct.pack("<d", point[2]))
            f.write(struct.pack("<B", int(color[0])))
            f.write(struct.pack("<B", int(color[1])))
            f.write(struct.pack("<B", int(color[2])))
            f.write(struct.pack("<d", 0.0))  # error
            # Empty track (no image observations)
            f.write(struct.pack("<Q", 0))


def convert_to_colmap_format(
    da3_output: DA3StreamingOutput,
    output_dir: str,
    force: bool = False,
) -> str:
    """Convert DA3-streaming output to COLMAP format.

    Args:
        da3_output: Loaded DA3-streaming output
        output_dir: Output directory for COLMAP data
        force: Force reconversion even if outputs exist

    Returns:
        Path to COLMAP sparse reconstruction directory

    Resume logic:
        - If colmap/sparse/0/*.bin all exist, skip conversion
    """
    colmap_dir = os.path.join(output_dir, "colmap")
    sparse_dir = os.path.join(colmap_dir, "sparse", "0")
    images_link_dir = os.path.join(colmap_dir, "images")

    # Check for existing outputs
    cameras_bin = os.path.join(sparse_dir, "cameras.bin")
    images_bin = os.path.join(sparse_dir, "images.bin")
    points3d_bin = os.path.join(sparse_dir, "points3D.bin")

    if not force and all(os.path.exists(p) for p in [cameras_bin, images_bin, points3d_bin]):
        print(f"COLMAP data already exists in {sparse_dir}")
        print("Use --force to reconvert")
        return colmap_dir

    print(f"\nConverting to COLMAP format...")
    os.makedirs(sparse_dir, exist_ok=True)

    # Get image dimensions from first image
    from PIL import Image
    with Image.open(da3_output.image_paths[0]) as img:
        width, height = img.size
    print(f"  Image size: {width}x{height}")

    # Create cameras.bin (single shared camera)
    fx, fy, cx, cy = da3_output.intrinsics
    cameras = {
        1: {
            "model_id": CAMERA_MODELS["PINHOLE"].model_id,
            "width": width,
            "height": height,
            "params": [fx, fy, cx, cy],
        }
    }
    write_cameras_binary(cameras, cameras_bin)
    print(f"  Created cameras.bin (1 camera)")

    # Create images.bin
    # Convert C2W poses to W2C (COLMAP format)
    images = {}
    for i, (pose, image_path) in enumerate(zip(da3_output.poses, da3_output.image_paths)):
        image_id = i + 1

        # C2W -> W2C
        c2w = pose
        w2c = np.linalg.inv(c2w)

        # Extract rotation and translation
        R = w2c[:3, :3]
        t = w2c[:3, 3]

        # Convert rotation to quaternion
        quat = rotation_matrix_to_quaternion(R)

        images[image_id] = {
            "qw": quat[0],
            "qx": quat[1],
            "qy": quat[2],
            "qz": quat[3],
            "tx": t[0],
            "ty": t[1],
            "tz": t[2],
            "camera_id": 1,
            "name": os.path.basename(image_path),
        }

    write_images_binary(images, images_bin)
    print(f"  Created images.bin ({len(images)} images)")

    # Create points3D.bin
    write_points3d_binary(da3_output.points, da3_output.colors, points3d_bin)
    print(f"  Created points3D.bin ({len(da3_output.points)} points)")

    # Create symlink or copy images to COLMAP images directory
    if os.path.exists(images_link_dir):
        if os.path.islink(images_link_dir):
            os.unlink(images_link_dir)
        else:
            shutil.rmtree(images_link_dir)

    # Try symlink first, fall back to copy
    try:
        os.symlink(os.path.abspath(da3_output.images_dir), images_link_dir)
        print(f"  Linked images directory")
    except OSError:
        # Copy images if symlink fails (e.g., on Windows)
        shutil.copytree(da3_output.images_dir, images_link_dir)
        print(f"  Copied images directory")

    return colmap_dir


def find_gaussian_splatting() -> Optional[str]:
    """Find gaussian-splatting installation.

    Returns:
        Path to gaussian-splatting directory, or None if not found
    """
    # Check environment variable
    gs_path = os.environ.get("GAUSSIAN_SPLATTING_PATH")
    if gs_path and os.path.exists(os.path.join(gs_path, "train.py")):
        return gs_path

    # Check common locations
    common_paths = [
        os.path.expanduser("~/gaussian-splatting"),
        os.path.expanduser("~/repos/gaussian-splatting"),
        "/opt/gaussian-splatting",
    ]

    for path in common_paths:
        if os.path.exists(os.path.join(path, "train.py")):
            return path

    return None


def train_gaussian_splatting(
    colmap_dir: str,
    output_dir: str,
    iterations: int = 30000,
    force: bool = False,
) -> str:
    """Train 3D Gaussian Splatting model.

    Args:
        colmap_dir: Path to COLMAP format data
        output_dir: Output directory for trained model
        iterations: Number of training iterations
        force: Force retraining even if model exists

    Returns:
        Path to trained model directory

    Resume logic:
        - If point_cloud/iteration_{iterations}/point_cloud.ply exists, skip training
        - gaussian-splatting has internal checkpoint resume
    """
    model_dir = os.path.join(output_dir, "model")
    final_ply = os.path.join(model_dir, "point_cloud", f"iteration_{iterations}", "point_cloud.ply")

    if not force and os.path.exists(final_ply):
        print(f"Trained model already exists: {final_ply}")
        print("Use --force to retrain")
        return model_dir

    # Find gaussian-splatting
    gs_path = find_gaussian_splatting()
    if gs_path is None:
        print("\nWarning: gaussian-splatting not found!")
        print("To train 3DGS, either:")
        print("  1. Set GAUSSIAN_SPLATTING_PATH environment variable")
        print("  2. Clone to ~/gaussian-splatting")
        print("  3. Install gsplat: pip install gsplat")
        print("\nSkipping training. COLMAP data is ready for manual training:")
        print(f"  {colmap_dir}")
        return None

    print(f"\nTraining 3D Gaussian Splatting...")
    print(f"  Using: {gs_path}")
    print(f"  Iterations: {iterations}")

    os.makedirs(model_dir, exist_ok=True)

    # Run training
    cmd = [
        sys.executable,
        os.path.join(gs_path, "train.py"),
        "-s", colmap_dir,
        "-m", model_dir,
        "--iterations", str(iterations),
    ]

    print(f"  Command: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        print(line, end='')

    process.wait()

    if process.returncode != 0:
        print(f"Warning: Training finished with return code {process.returncode}")

    return model_dir


def process_pointcloud_to_3dgs(
    pointcloud_path: str,
    poses_path: str,
    intrinsics_path: str,
    images_dir: str,
    output_dir: str = None,
    iterations: int = 30000,
    force: bool = False,
) -> dict:
    """
    Complete pipeline from point cloud to trained 3DGS.

    Args:
        pointcloud_path: Path to combined_pcd.ply
        poses_path: Path to camera_poses.txt
        intrinsics_path: Path to intrinsic.txt
        images_dir: Directory containing input images
        output_dir: If None, uses {pointcloud_parent}/3dgs/
        iterations: Number of training iterations
        force: If False, skip completed steps

    Returns:
        Dictionary with paths to outputs:
        - colmap_dir: Path to COLMAP format data
        - model_dir: Path to trained model (or None if training skipped)
        - final_ply: Path to final point cloud PLY
    """
    # Default output directory
    if output_dir is None:
        pointcloud_parent = os.path.dirname(os.path.abspath(pointcloud_path))
        output_dir = os.path.join(pointcloud_parent, "3dgs")

    os.makedirs(output_dir, exist_ok=True)

    # Load DA3-streaming output
    print("Loading DA3-streaming output...")
    da3_output = load_da3_output(
        pointcloud_path=pointcloud_path,
        poses_path=poses_path,
        intrinsics_path=intrinsics_path,
        images_dir=images_dir,
    )

    # Convert to COLMAP format
    colmap_dir = convert_to_colmap_format(da3_output, output_dir, force=force)

    # Train 3DGS
    model_dir = train_gaussian_splatting(colmap_dir, output_dir, iterations=iterations, force=force)

    # Find final PLY
    final_ply = None
    if model_dir:
        final_ply = os.path.join(model_dir, "point_cloud", f"iteration_{iterations}", "point_cloud.ply")
        if not os.path.exists(final_ply):
            # Try to find any iteration
            ply_pattern = os.path.join(model_dir, "point_cloud", "iteration_*", "point_cloud.ply")
            plys = sorted(glob.glob(ply_pattern))
            if plys:
                final_ply = plys[-1]

    return {
        "colmap_dir": colmap_dir,
        "model_dir": model_dir,
        "final_ply": final_ply,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert point cloud to 3D Gaussian Splatting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python scripts/pointcloud_to_3dgs.py \\
        -p ./pointcloud/combined_pcd.ply \\
        --poses ./pointcloud/camera_poses.txt \\
        --intrinsics ./pointcloud/intrinsic.txt \\
        --images-dir ./cubemap \\
        -o ./3dgs

    # Quick test with fewer iterations
    python scripts/pointcloud_to_3dgs.py \\
        -p ./pointcloud/combined_pcd.ply \\
        --poses ./pointcloud/camera_poses.txt \\
        --intrinsics ./pointcloud/intrinsic.txt \\
        --images-dir ./cubemap \\
        -o ./3dgs \\
        --iterations 1000

Environment Variables:
    GAUSSIAN_SPLATTING_PATH: Path to gaussian-splatting repository
        """
    )
    parser.add_argument(
        "--pointcloud", "-p",
        required=True,
        help="Path to combined_pcd.ply"
    )
    parser.add_argument(
        "--poses",
        required=True,
        help="Path to camera_poses.txt"
    )
    parser.add_argument(
        "--intrinsics",
        required=True,
        help="Path to intrinsic.txt"
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        help="Directory containing input images"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory (default: {pointcloud_parent}/3dgs/)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=30000,
        help="Number of training iterations (default: 30000)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if outputs exist"
    )

    args = parser.parse_args()

    result = process_pointcloud_to_3dgs(
        pointcloud_path=args.pointcloud,
        poses_path=args.poses,
        intrinsics_path=args.intrinsics,
        images_dir=args.images_dir,
        output_dir=args.output,
        iterations=args.iterations,
        force=args.force,
    )

    print(f"\nOutputs:")
    print(f"  COLMAP data: {result['colmap_dir']}")
    if result['model_dir']:
        print(f"  Model: {result['model_dir']}")
    if result['final_ply']:
        print(f"  Final PLY: {result['final_ply']}")


if __name__ == "__main__":
    main()
