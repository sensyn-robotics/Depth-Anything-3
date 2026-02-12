#!/usr/bin/env python3
"""Simple Gaussian splatting renderer using gsplat."""

import argparse
import struct

import numpy as np
import torch
from pathlib import Path
from PIL import Image
from plyfile import PlyData


def read_cameras_binary(path):
    """Read COLMAP cameras.bin file."""
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            num_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 4, 5: 5}.get(model_id, 4)
            params = struct.unpack(f"<{num_params}d", f.read(8 * num_params))
            cameras[camera_id] = {
                "id": camera_id,
                "model_id": model_id,
                "width": width,
                "height": height,
                "params": params,
            }
    return cameras


def read_images_binary(path):
    """Read COLMAP images.bin file."""
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            f.read(24 * num_points2D)  # Skip 2D points
            images[image_id] = {
                "id": image_id,
                "qvec": np.array([qw, qx, qy, qz]),
                "tvec": np.array([tx, ty, tz]),
                "camera_id": camera_id,
                "name": name.decode("utf-8"),
            }
    return images


def qvec2rotmat(qvec):
    """Convert quaternion to rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ])


def load_ply(path):
    """Load Gaussian splatting PLY file."""
    plydata = PlyData.read(path)
    vertex = plydata["vertex"]

    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)

    # DC color (SH degree 0)
    f_dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1)

    # Opacity (stored as logit)
    opacity = vertex["opacity"][:, None]

    # Scale (stored as log)
    scale = np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1)

    # Rotation quaternion
    rot = np.stack([vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]], axis=1)

    return {
        "xyz": torch.tensor(xyz, dtype=torch.float32),
        "f_dc": torch.tensor(f_dc, dtype=torch.float32),
        "opacity": torch.tensor(opacity, dtype=torch.float32),
        "scale": torch.tensor(scale, dtype=torch.float32),
        "rot": torch.tensor(rot, dtype=torch.float32),
    }


def render_frame(gaussians, viewmat, K, width, height, device="cuda", max_gaussians=200000):
    """Render a single frame using gsplat."""
    from gsplat import rasterization

    # Subsample if too many gaussians
    n_gaussians = len(gaussians["xyz"])
    if n_gaussians > max_gaussians:
        indices = torch.randperm(n_gaussians)[:max_gaussians]
        gaussians = {k: v[indices] for k, v in gaussians.items()}
        print(f"    Subsampled to {max_gaussians} gaussians")

    # Move to device
    means = gaussians["xyz"].to(device)
    quats = gaussians["rot"].to(device)
    scales = torch.exp(gaussians["scale"].to(device))
    opacities = torch.sigmoid(gaussians["opacity"].to(device)).squeeze(-1)

    # Convert SH DC to RGB
    SH_C0 = 0.28209479177387814
    colors = gaussians["f_dc"].to(device) * SH_C0 + 0.5
    colors = colors.clamp(0, 1)

    # View matrix (4x4)
    viewmat = torch.tensor(viewmat, dtype=torch.float32, device=device)

    # Intrinsics
    fx, fy, cx, cy = K
    Ks = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=device)

    # Rasterize with packed=True for memory efficiency
    renders, alphas, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat[None],
        Ks=Ks[None],
        width=width,
        height=height,
        packed=True,
        near_plane=0.01,
        far_plane=1000.0,
    )

    return renders[0].cpu().numpy(), alphas[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Render Gaussian splatting model")
    parser.add_argument("-m", "--model", required=True, help="Path to model directory")
    parser.add_argument("-o", "--output", default="rendered", help="Output directory")
    parser.add_argument("-n", "--num-frames", type=int, default=10, help="Number of frames to render")
    parser.add_argument("--width", type=int, default=800, help="Output width")
    parser.add_argument("--height", type=int, default=600, help="Output height")
    parser.add_argument("--max-gaussians", type=int, default=200000, help="Max gaussians to render")
    args = parser.parse_args()

    model_path = Path(args.model)

    # Find PLY file
    ply_paths = list(model_path.glob("**/point_cloud/iteration_*/point_cloud.ply"))
    if not ply_paths:
        ply_paths = list(model_path.glob("**/point_cloud.ply"))
    if not ply_paths:
        raise FileNotFoundError(f"No PLY file found in {model_path}")
    ply_path = sorted(ply_paths)[-1]  # Use latest iteration
    print(f"Loading PLY: {ply_path}")

    # Find COLMAP data
    colmap_paths = [
        model_path / "colmap" / "sparse" / "0",
        model_path / "sparse" / "0",
        model_path.parent / "colmap" / "sparse" / "0",
    ]
    colmap_path = None
    for p in colmap_paths:
        if (p / "cameras.bin").exists():
            colmap_path = p
            break

    if colmap_path is None:
        raise FileNotFoundError("Could not find COLMAP data")
    print(f"Using COLMAP data from: {colmap_path}")

    # Load data
    gaussians = load_ply(ply_path)
    print(f"Loaded {len(gaussians['xyz'])} Gaussians")

    cameras = read_cameras_binary(colmap_path / "cameras.bin")
    images = read_images_binary(colmap_path / "images.bin")

    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select frames to render
    image_list = sorted(images.values(), key=lambda x: x["name"])
    step = max(1, len(image_list) // args.num_frames)
    selected = image_list[::step][:args.num_frames]

    print(f"Rendering {len(selected)} frames...")

    for i, img_data in enumerate(selected):
        cam = cameras[img_data["camera_id"]]

        # Get camera parameters
        width = args.width or cam["width"]
        height = args.height or cam["height"]
        scale_x = width / cam["width"]
        scale_y = height / cam["height"]

        params = cam["params"]
        fx = params[0] * scale_x
        fy = params[1] * scale_y if len(params) > 1 else fx
        cx = params[2] * scale_x if len(params) > 2 else width / 2
        cy = params[3] * scale_y if len(params) > 3 else height / 2

        # Build view matrix (world to camera)
        R = qvec2rotmat(img_data["qvec"])
        t = img_data["tvec"]
        viewmat = np.eye(4)
        viewmat[:3, :3] = R
        viewmat[:3, 3] = t

        # Render
        rgb, alpha = render_frame(gaussians, viewmat, (fx, fy, cx, cy), width, height,
                                  max_gaussians=args.max_gaussians)

        # Save
        rgb_uint8 = (rgb * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(rgb_uint8).save(output_dir / f"frame_{i:04d}.png")
        print(f"  Saved frame_{i:04d}.png ({img_data['name']})")

    print(f"\nDone! Rendered images saved to: {output_dir}")


if __name__ == "__main__":
    main()
