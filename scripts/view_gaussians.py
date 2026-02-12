#!/usr/bin/env python3
"""Simple viser-based Gaussian splatting viewer."""

import argparse
import time

import numpy as np
import viser
from pathlib import Path
from plyfile import PlyData


def load_ply(path, max_points=500000):
    """Load Gaussian splatting PLY file."""
    print(f"Loading PLY: {path}")
    plydata = PlyData.read(path)
    vertex = plydata["vertex"]
    n_points = len(vertex)
    print(f"Total points: {n_points}")

    # Subsample if too many points
    if n_points > max_points:
        indices = np.random.choice(n_points, max_points, replace=False)
        print(f"Subsampled to {max_points} points")
    else:
        indices = np.arange(n_points)

    xyz = np.stack([vertex["x"][indices], vertex["y"][indices], vertex["z"][indices]], axis=1)

    # DC color (SH degree 0) - convert from SH to RGB
    SH_C0 = 0.28209479177387814
    f_dc = np.stack([
        vertex["f_dc_0"][indices],
        vertex["f_dc_1"][indices],
        vertex["f_dc_2"][indices]
    ], axis=1)
    colors = f_dc * SH_C0 + 0.5
    colors = np.clip(colors, 0, 1)

    # Opacity (stored as logit)
    opacity = 1.0 / (1.0 + np.exp(-vertex["opacity"][indices]))

    # Scale (stored as log)
    scales = np.exp(np.stack([
        vertex["scale_0"][indices],
        vertex["scale_1"][indices],
        vertex["scale_2"][indices]
    ], axis=1))
    avg_scale = np.mean(scales, axis=1)

    return {
        "xyz": xyz.astype(np.float32),
        "colors": colors.astype(np.float32),
        "opacity": opacity.astype(np.float32),
        "scales": avg_scale.astype(np.float32),
    }


def main():
    parser = argparse.ArgumentParser(description="View Gaussian splatting model with viser")
    parser.add_argument("-m", "--model", required=True, help="Path to model directory or PLY file")
    parser.add_argument("--max-points", type=int, default=300000, help="Max points to display")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--point-size", type=float, default=0.01, help="Point size")
    args = parser.parse_args()

    model_path = Path(args.model)

    # Find PLY file
    if model_path.suffix == ".ply":
        ply_path = model_path
    else:
        ply_paths = list(model_path.glob("**/point_cloud/iteration_*/point_cloud.ply"))
        if not ply_paths:
            ply_paths = list(model_path.glob("**/point_cloud.ply"))
        if not ply_paths:
            raise FileNotFoundError(f"No PLY file found in {model_path}")
        ply_path = sorted(ply_paths)[-1]  # Use latest iteration

    # Load data
    data = load_ply(ply_path, max_points=args.max_points)

    # Create viser server
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    print(f"\nViewer running at: http://localhost:{args.port}")
    print("Press Ctrl+C to stop\n")

    # Add point cloud
    colors_uint8 = (data["colors"] * 255).astype(np.uint8)

    server.scene.add_point_cloud(
        "/gaussians",
        points=data["xyz"],
        colors=colors_uint8,
        point_size=args.point_size,
        point_shape="circle",
    )

    # Add coordinate frame
    server.scene.add_frame("/origin", wxyz=np.array([1.0, 0.0, 0.0, 0.0]), position=np.array([0.0, 0.0, 0.0]), axes_length=1.0)

    # Add controls
    with server.gui.add_folder("Controls"):
        point_size_slider = server.gui.add_slider(
            "Point Size", min=0.001, max=0.1, step=0.001, initial_value=args.point_size
        )
        opacity_slider = server.gui.add_slider(
            "Opacity", min=0.1, max=1.0, step=0.05, initial_value=1.0
        )

    @point_size_slider.on_update
    def _(_) -> None:
        server.scene.add_point_cloud(
            "/gaussians",
            points=data["xyz"],
            colors=colors_uint8,
            point_size=point_size_slider.value,
            point_shape="circle",
        )

    # Keep server running
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
