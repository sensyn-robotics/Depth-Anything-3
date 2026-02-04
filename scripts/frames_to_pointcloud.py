#!/usr/bin/env python3
"""
Process image frames through DA3-streaming to generate aligned point cloud.

This script runs DA3-streaming on image frames to produce:
- Aligned merged point cloud
- Camera poses (4x4 C2W matrices)
- Camera intrinsics

Usage:
    python scripts/frames_to_pointcloud.py -i ./frames -o ./pointcloud

Output:
    - combined_pcd.ply: Aligned merged point cloud
    - camera_poses.txt: Camera poses
    - intrinsic.txt: Camera intrinsics
"""

import argparse
import os
import shutil
import subprocess
import sys

import yaml


def check_da3_streaming_weights(da3_streaming_dir: str) -> bool:
    """Check if DA3-streaming weights are downloaded."""
    weights_dir = os.path.join(da3_streaming_dir, "weights")
    required_files = ["model.safetensors", "config.json", "dino_salad.ckpt"]

    for f in required_files:
        if not os.path.exists(os.path.join(weights_dir, f)):
            return False
    return True


def download_da3_streaming_weights(da3_streaming_dir: str):
    """Download DA3-streaming weights if not present."""
    script_path = os.path.join(da3_streaming_dir, "scripts", "download_weights.sh")

    if os.path.exists(script_path):
        print("Downloading DA3-streaming weights...")
        result = subprocess.run(
            ["bash", script_path], cwd=da3_streaming_dir, capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"Warning: Weight download may have failed: {result.stderr}")
    else:
        print(f"Warning: Download script not found at {script_path}")
        print("Please download weights manually following da3_streaming/README.md")


def create_streaming_config(
    config_path: str,
    chunk_size: int = 60,
    overlap: int = 30,
    loop_enable: bool = True,
    salad_batch_size: int = 32,
    process_res: int = 504,
) -> str:
    """Create a config file for DA3-streaming.

    Args:
        config_path: Path to save the config file
        chunk_size: Number of images per chunk
        overlap: Overlap between chunks
        loop_enable: Enable loop closure detection
        salad_batch_size: SALAD batch size for loop detection
        process_res: DA3 internal processing resolution

    Returns:
        Path to the created config file
    """
    config = {
        "Weights": {
            "DA3": "./weights/model.safetensors",
            "DA3_CONFIG": "./weights/config.json",
            "SALAD": "./weights/dino_salad.ckpt",
        },
        "Model": {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "loop_chunk_size": 20,
            "loop_enable": loop_enable,
            "process_res": process_res,
            "useDBoW": False,
            "delete_temp_files": True,
            "align_lib": "triton",
            "align_method": "sim3",
            "scale_compute_method": "auto",
            "align_type": "dense",
            "ref_view_strategy": "saddle_balanced",
            "ref_view_strategy_loop": "saddle_balanced",
            "depth_threshold": 15.0,
            "save_depth_conf_result": False,
            "save_debug_info": False,
            "Sparse_Align": {
                "keypoint_select": "orb",
                "keypoint_num": 5000,
            },
            "IRLS": {
                "delta": 0.1,
                "max_iters": 5,
                "tol": "1e-9",
            },
            "Pointcloud_Save": {
                "sample_ratio": 0.015,
                "conf_threshold_coef": 0.75,
            },
        },
        "Loop": {
            "SALAD": {
                "image_size": [336, 336],
                "batch_size": salad_batch_size,
                "similarity_threshold": 0.85,
                "top_k": 5,
                "use_nms": True,
                "nms_threshold": 25,
            },
            "SIM3_Optimizer": {
                "lang_version": "cpp",
                "max_iterations": 30,
                "lambda_init": 1e-6,
            },
        },
    }

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return config_path


def run_da3_streaming(
    image_dir: str,
    output_dir: str,
    da3_streaming_dir: str,
    config_path: str = None,
    chunk_size: int = 60,
    overlap: int = 30,
    loop_enable: bool = True,
    salad_batch_size: int = 32,
    process_res: int = 504,
) -> str:
    """Run DA3-streaming on the cubemap images.

    Args:
        image_dir: Directory containing cubemap images
        output_dir: Output directory for results
        da3_streaming_dir: Path to DA3-streaming installation
        config_path: Optional path to existing config file
        chunk_size: Number of images per chunk
        overlap: Overlap between chunks
        loop_enable: Enable loop closure detection
        salad_batch_size: SALAD batch size
        process_res: DA3 processing resolution

    Returns:
        Path to output directory
    """
    # Convert all paths to absolute to avoid cwd issues
    da3_streaming_dir = os.path.abspath(da3_streaming_dir)
    image_dir = os.path.abspath(image_dir)
    output_dir = os.path.abspath(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Check weights
    if not check_da3_streaming_weights(da3_streaming_dir):
        download_da3_streaming_weights(da3_streaming_dir)

        if not check_da3_streaming_weights(da3_streaming_dir):
            raise RuntimeError(
                "DA3-streaming weights not found. Please download them manually:\n"
                f"  cd {da3_streaming_dir} && bash scripts/download_weights.sh"
            )

    # Create config if not provided
    if config_path is None:
        config_path = os.path.join(output_dir, "streaming_config.yaml")
        create_streaming_config(
            config_path,
            chunk_size=chunk_size,
            overlap=overlap,
            loop_enable=loop_enable,
            salad_batch_size=salad_batch_size,
            process_res=process_res,
        )
    config_path = os.path.abspath(config_path)

    # Run DA3-streaming
    script_path = "da3_streaming.py"

    cmd = [
        sys.executable,
        script_path,
        "--image_dir",
        image_dir,
        "--output_dir",
        output_dir,
        "--config",
        config_path,
    ]

    print(f"\nRunning DA3-streaming...")
    print(f"  Image dir: {image_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Config: {config_path}")

    # Set PYTHONPATH to include da3_streaming directory for module imports
    env = os.environ.copy()
    env["PYTHONPATH"] = da3_streaming_dir + os.pathsep + env.get("PYTHONPATH", "")

    # Run with real-time output
    process = subprocess.Popen(
        cmd,
        cwd=da3_streaming_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        print(line, end="")

    process.wait()

    if process.returncode != 0:
        raise RuntimeError(f"DA3-streaming failed with return code {process.returncode}")

    return output_dir


def process_frames_to_pointcloud(
    input_dir: str,
    output_dir: str = None,
    chunk_size: int = 60,
    overlap: int = 30,
    loop_enable: bool = True,
    salad_batch_size: int = 32,
    process_res: int = 504,
    force: bool = False,
) -> dict:
    """
    Process cubemap images through DA3-streaming.

    Args:
        input_dir: Directory containing cubemap images
        output_dir: If None, uses {input_dir}/../pointcloud/
        chunk_size: Number of images per chunk
        overlap: Overlap between chunks
        loop_enable: Enable loop closure detection
        salad_batch_size: SALAD batch size
        process_res: DA3 processing resolution
        force: If False and outputs exist, return existing paths

    Returns:
        Dictionary with paths to outputs:
        - pointcloud_path: Path to combined_pcd.ply
        - poses_path: Path to camera_poses.txt
        - intrinsics_path: Path to intrinsic.txt
        - streaming_output_dir: Path to full streaming output

    Resume logic:
        - If combined_pcd.ply, camera_poses.txt, intrinsic.txt all exist, skip
        - DA3-streaming has internal resume support for chunk processing
    """
    # Default output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(input_dir)), "pointcloud")

    os.makedirs(output_dir, exist_ok=True)

    # Check for existing outputs (resume support)
    pcd_path = os.path.join(output_dir, "combined_pcd.ply")
    poses_path = os.path.join(output_dir, "camera_poses.txt")
    intrinsics_path = os.path.join(output_dir, "intrinsic.txt")

    if not force and all(os.path.exists(p) for p in [pcd_path, poses_path, intrinsics_path]):
        print(f"Found existing outputs in {output_dir}")
        print("Use --force to reprocess")
        return {
            "pointcloud_path": pcd_path,
            "poses_path": poses_path,
            "intrinsics_path": intrinsics_path,
            "streaming_output_dir": os.path.join(output_dir, "streaming_output"),
        }

    # Find DA3-streaming directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    da3_streaming_dir = os.path.join(project_root, "da3_streaming")

    if not os.path.exists(da3_streaming_dir):
        raise RuntimeError(f"DA3-streaming directory not found: {da3_streaming_dir}")

    # Run DA3-streaming
    streaming_output = os.path.join(output_dir, "streaming_output")

    run_da3_streaming(
        image_dir=input_dir,
        output_dir=streaming_output,
        da3_streaming_dir=da3_streaming_dir,
        chunk_size=chunk_size,
        overlap=overlap,
        loop_enable=loop_enable,
        salad_batch_size=salad_batch_size,
        process_res=process_res,
    )

    # Copy main outputs to output dir root
    pcd_src = os.path.join(streaming_output, "pcd", "combined_pcd.ply")
    poses_src = os.path.join(streaming_output, "camera_poses.txt")
    intrinsics_src = os.path.join(streaming_output, "intrinsic.txt")

    if os.path.exists(pcd_src):
        shutil.copy(pcd_src, pcd_path)
        print(f"  Point cloud: {pcd_path}")

    if os.path.exists(poses_src):
        shutil.copy(poses_src, poses_path)
        print(f"  Camera poses: {poses_path}")

    if os.path.exists(intrinsics_src):
        shutil.copy(intrinsics_src, intrinsics_path)
        print(f"  Intrinsics: {intrinsics_path}")

    return {
        "pointcloud_path": pcd_path,
        "poses_path": poses_path,
        "intrinsics_path": intrinsics_path,
        "streaming_output_dir": streaming_output,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Process cubemap images through DA3-streaming",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python scripts/cubemap_to_pointcloud.py -i ./cubemap -o ./pointcloud

    # Low memory mode for 8-12GB VRAM GPUs
    python scripts/cubemap_to_pointcloud.py -i ./cubemap -o ./pointcloud --low-memory

    # Custom chunk settings
    python scripts/cubemap_to_pointcloud.py -i ./cubemap -o ./pointcloud \\
        --chunk-size 30 --overlap 15

    # Disable loop closure for faster processing
    python scripts/cubemap_to_pointcloud.py -i ./cubemap -o ./pointcloud --no-loop
        """,
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Input directory containing cubemap images"
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output directory (default: {input_dir}/../pointcloud/)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=60, help="DA3-streaming chunk size (default: 60)"
    )
    parser.add_argument(
        "--overlap", type=int, default=30, help="Overlap between chunks (default: 30)"
    )
    parser.add_argument("--no-loop", action="store_true", help="Disable loop closure detection")
    parser.add_argument(
        "--salad-batch-size",
        type=int,
        default=32,
        help="SALAD loop closure batch size (default: 32)",
    )
    parser.add_argument(
        "--process-res", type=int, default=504, help="DA3 processing resolution (default: 504)"
    )
    parser.add_argument(
        "--low-memory", action="store_true", help="Enable low-memory mode for 8-12GB VRAM GPUs"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force reprocessing even if outputs exist"
    )

    args = parser.parse_args()

    # Apply low-memory defaults
    chunk_size = args.chunk_size
    overlap = args.overlap
    salad_batch_size = args.salad_batch_size
    loop_enable = not args.no_loop
    process_res = args.process_res

    if args.low_memory:
        print("Low-memory mode enabled")
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

    result = process_frames_to_pointcloud(
        input_dir=args.input,
        output_dir=args.output,
        chunk_size=chunk_size,
        overlap=overlap,
        loop_enable=loop_enable,
        salad_batch_size=salad_batch_size,
        process_res=process_res,
        force=args.force,
    )

    print(f"\nOutputs:")
    print(f"  Point cloud: {result['pointcloud_path']}")
    print(f"  Camera poses: {result['poses_path']}")
    print(f"  Intrinsics: {result['intrinsics_path']}")


if __name__ == "__main__":
    main()
