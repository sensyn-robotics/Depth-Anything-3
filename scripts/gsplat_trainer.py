#!/usr/bin/env python3
"""
gsplat-based 3D Gaussian Splatting trainer.

Trains 3DGS from COLMAP binary data using gsplat, eliminating the need
for the external gaussian-splatting repository.

Expects COLMAP directory structure:
    colmap/
        sparse/0/
            cameras.bin
            images.bin
            points3D.bin
        images/
            *.png / *.jpg
"""

import math
import os
import struct
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# COLMAP binary readers
# ---------------------------------------------------------------------------

CameraModel = namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
CAMERA_MODELS = {
    1: CameraModel(1, "PINHOLE", 4),
    2: CameraModel(2, "SIMPLE_RADIAL", 4),
    3: CameraModel(3, "RADIAL", 5),
    4: CameraModel(4, "OPENCV", 8),
}


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray  # fx, fy, cx, cy for PINHOLE


@dataclass
class Image:
    id: int
    qvec: np.ndarray  # (w, x, y, z)
    tvec: np.ndarray  # (3,)
    camera_id: int
    name: str


def read_cameras_binary(path: str) -> dict[int, Camera]:
    cameras = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            cam_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            model = CAMERA_MODELS[model_id]
            params = np.array(
                struct.unpack(f"<{model.num_params}d", f.read(8 * model.num_params))
            )
            cameras[cam_id] = Camera(cam_id, model.model_name, width, height, params)
    return cameras


def read_images_binary(path: str) -> dict[int, Image]:
    images = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            img_id = struct.unpack("<I", f.read(4))[0]
            qvec = np.array(struct.unpack("<4d", f.read(32)))
            tvec = np.array(struct.unpack("<3d", f.read(24)))
            camera_id = struct.unpack("<I", f.read(4))[0]
            # null-terminated name
            name_chars = []
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_chars.append(c.decode("utf-8"))
            name = "".join(name_chars)
            # skip 2D points
            num_points2d = struct.unpack("<Q", f.read(8))[0]
            f.read(num_points2d * 24)  # each: x(d) y(d) id(Q)
            images[img_id] = Image(img_id, qvec, tvec, camera_id, name)
    return images


def read_points3d_binary(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (xyz [N,3], rgb [N,3] uint8)."""
    points, colors = [], []
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            _ = struct.unpack("<Q", f.read(8))[0]  # point_id
            xyz = np.array(struct.unpack("<3d", f.read(24)))
            rgb = np.array(struct.unpack("<3B", f.read(3)), dtype=np.uint8)
            _ = struct.unpack("<d", f.read(8))[0]  # error
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(track_len * 8)  # each: image_id(I) point2d_idx(I)
            points.append(xyz)
            colors.append(rgb)
    return np.array(points, dtype=np.float64), np.array(colors, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Quaternion / rotation helpers
# ---------------------------------------------------------------------------

def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """COLMAP quaternion (w,x,y,z) -> 3x3 rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y],
    ])


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def _fspecial_gauss(size: int, sigma: float, device):
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = torch.outer(g, g)
    return (g / g.sum()).unsqueeze(0).unsqueeze(0)


def ssim(img1, img2, window_size=11):
    """Compute SSIM between two [B,C,H,W] tensors."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ch = img1.shape[1]
    window = _fspecial_gauss(window_size, 1.5, img1.device).expand(ch, -1, -1, -1)
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=ch)
    mu2 = F.conv2d(img2, window, padding=pad, groups=ch)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=ch) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=ch) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=ch) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


# ---------------------------------------------------------------------------
# PLY writer (standard 3DGS format)
# ---------------------------------------------------------------------------

def save_ply(path: str, means: np.ndarray, scales: np.ndarray,
             rotations: np.ndarray, opacities: np.ndarray, sh_coeffs: np.ndarray):
    """Save Gaussians to PLY in standard 3DGS format."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    n = means.shape[0]
    sh_dim = sh_coeffs.shape[1]

    # Build dtype
    attrs = [("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
    for i in range(sh_dim):
        attrs.append((f"f_rest_{i}" if i > 0 else "f_dc_0", "f4"))
    # Actually, standard format uses f_dc_0..2 then f_rest_0..
    # Let's follow the convention properly:
    attrs = [("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
    # DC: 3 channels
    for i in range(3):
        attrs.append((f"f_dc_{i}", "f4"))
    # Rest of SH
    for i in range(sh_dim - 3):
        attrs.append((f"f_rest_{i}", "f4"))
    attrs.append(("opacity", "f4"))
    for i in range(3):
        attrs.append((f"scale_{i}", "f4"))
    for i in range(4):
        attrs.append((f"rot_{i}", "f4"))

    dtype = np.dtype(attrs)
    elements = np.empty(n, dtype=dtype)

    elements["x"] = means[:, 0]
    elements["y"] = means[:, 1]
    elements["z"] = means[:, 2]
    elements["nx"] = 0
    elements["ny"] = 0
    elements["nz"] = 0

    for i in range(3):
        elements[f"f_dc_{i}"] = sh_coeffs[:, i]
    for i in range(sh_dim - 3):
        elements[f"f_rest_{i}"] = sh_coeffs[:, i + 3]

    elements["opacity"] = opacities[:, 0]
    for i in range(3):
        elements[f"scale_{i}"] = scales[:, i]
    for i in range(4):
        elements[f"rot_{i}"] = rotations[:, i]

    # Write PLY
    with open(path, "wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n}\n"
        )
        for name, fmt in attrs:
            header += f"property float {name}\n"
        header += "end_header\n"
        f.write(header.encode("ascii"))
        f.write(elements.tobytes())

    print(f"  Saved PLY: {path} ({n} Gaussians)")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    colmap_dir: str,
    model_dir: str,
    iterations: int = 30000,
    resolution: int = -1,
    strategy: str = "mcmc",
):
    """
    Train 3DGS using gsplat.

    Args:
        colmap_dir: COLMAP directory with sparse/0/ and images/
        model_dir: Output model directory
        iterations: Training iterations
        resolution: Target image width (-1 = original)
        strategy: "mcmc" or "default" (ADC)
    """
    import gsplat

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    print(f"  Strategy: {strategy}")

    # ---- Load COLMAP data ----
    sparse_dir = os.path.join(colmap_dir, "sparse", "0")
    cameras = read_cameras_binary(os.path.join(sparse_dir, "cameras.bin"))
    images = read_images_binary(os.path.join(sparse_dir, "images.bin"))
    points_xyz, points_rgb = read_points3d_binary(os.path.join(sparse_dir, "points3D.bin"))

    print(f"  Cameras: {len(cameras)}, Images: {len(images)}, Points: {len(points_xyz)}")

    images_dir = os.path.join(colmap_dir, "images")

    # ---- Load training images and build camera tensors ----
    from PIL import Image as PILImage

    # Sort images by id for deterministic order
    sorted_images = sorted(images.values(), key=lambda im: im.id)

    cam = cameras[sorted_images[0].camera_id]
    orig_w, orig_h = cam.width, cam.height
    if resolution > 0 and resolution != orig_w:
        scale_factor = resolution / orig_w
        target_w = resolution
        target_h = int(orig_h * scale_factor)
    else:
        scale_factor = 1.0
        target_w, target_h = orig_w, orig_h

    print(f"  Training resolution: {target_w}x{target_h}")

    gt_images = []
    viewmats = []
    Ks = []

    for img_meta in sorted_images:
        # Load image
        img_path = os.path.join(images_dir, img_meta.name)
        img = PILImage.open(img_path).convert("RGB")
        if scale_factor != 1.0:
            img = img.resize((target_w, target_h), PILImage.LANCZOS)
        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0  # [H,W,3]
        gt_images.append(img_tensor)

        # Build view matrix (W2C) as 4x4
        R = qvec_to_rotmat(img_meta.qvec)
        t = img_meta.tvec
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = t
        viewmats.append(torch.from_numpy(w2c).float())

        # Build intrinsics
        c = cameras[img_meta.camera_id]
        fx, fy, cx, cy = c.params[:4]
        K = torch.tensor([
            [fx * scale_factor, 0, cx * scale_factor],
            [0, fy * scale_factor, cy * scale_factor],
            [0, 0, 1],
        ], dtype=torch.float32)
        Ks.append(K)

    gt_images = torch.stack(gt_images).to(device)       # [N, H, W, 3]
    viewmats = torch.stack(viewmats).to(device)          # [N, 4, 4]
    Ks = torch.stack(Ks).to(device)                      # [N, 3, 3]
    num_views = len(gt_images)

    print(f"  Loaded {num_views} training views")

    # ---- Initialize Gaussians ----
    N = len(points_xyz)
    means = torch.from_numpy(points_xyz).float().to(device)
    means.requires_grad_(True)

    # RGB to SH DC coefficient: color = SH_C0 * sh_dc + 0.5
    SH_C0 = 0.28209479177387814
    rgb_normalized = torch.from_numpy(points_rgb).float().to(device) / 255.0
    sh_dc = (rgb_normalized - 0.5) / SH_C0  # [N, 3]
    sh_coeffs = sh_dc.contiguous().requires_grad_(True)

    # Compute initial scales from local point density
    from scipy.spatial import KDTree
    tree = KDTree(points_xyz)
    dists, _ = tree.query(points_xyz, k=4)  # k=4: self + 3 neighbors
    avg_dist = np.mean(dists[:, 1:], axis=1)  # exclude self
    avg_dist = np.clip(avg_dist, 1e-7, None)
    log_scales_np = np.log(avg_dist * 0.5)
    log_scales = torch.from_numpy(log_scales_np).float().to(device)
    log_scales = log_scales.unsqueeze(-1).expand(-1, 3).contiguous().requires_grad_(True)

    # Rotations as quaternions (w,x,y,z), init to identity
    quats = torch.zeros(N, 4, device=device)
    quats[:, 0] = 1.0
    quats.requires_grad_(True)

    # Opacities (logit space)
    opacities_logit = torch.logit(torch.full((N,), 0.5, device=device)).requires_grad_(True)

    # ---- Strategy ----
    if strategy == "mcmc":
        from gsplat.strategy import MCMCStrategy
        strat = MCMCStrategy(verbose=True)
        # MCMC needs cap_max
        strat_state = strat.initialize_state(scene_scale=1.0)
    else:
        from gsplat.strategy import DefaultStrategy
        strat = DefaultStrategy(verbose=True)
        strat_state = strat.initialize_state()

    # ---- Optimizer ----
    params = [
        {"params": [means], "lr": 1.6e-4, "name": "means"},
        {"params": [sh_coeffs], "lr": 2.5e-3, "name": "sh_coeffs"},
        {"params": [log_scales], "lr": 5e-3, "name": "log_scales"},
        {"params": [quats], "lr": 1e-3, "name": "quats"},
        {"params": [opacities_logit], "lr": 5e-2, "name": "opacities_logit"},
    ]
    optimizer = torch.optim.Adam(params, eps=1e-15)

    # Learning rate schedule for means (same as original 3DGS)
    def lr_lambda_means(step):
        lr_init = 1.6e-4
        lr_final = 1.6e-6
        max_steps = iterations
        t = min(step / max_steps, 1.0)
        lr = math.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t)
        return lr / lr_init

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[lr_lambda_means, lambda s: 1.0, lambda s: 1.0,
                   lambda s: 1.0, lambda s: 1.0],
    )

    # Checkpoint iterations
    save_iters = set()
    if iterations >= 7000:
        save_iters.add(7000)
    if iterations >= 30000:
        save_iters.add(30000)
    save_iters.add(iterations)

    # ---- Training loop ----
    print(f"\n  Starting training for {iterations} iterations...")

    for step in range(1, iterations + 1):
        # Random view
        idx = torch.randint(0, num_views, (1,)).item()
        gt_img = gt_images[idx]          # [H, W, 3]
        viewmat = viewmats[idx]          # [4, 4]
        K = Ks[idx]                      # [3, 3]

        scales = torch.exp(log_scales)
        opacities = torch.sigmoid(opacities_logit)

        # Rasterize
        renders, alphas, info = gsplat.rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=sh_coeffs,
            viewmats=viewmat.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=target_w,
            height=target_h,
            sh_degree=0,
        )
        # renders: [1, H, W, 3]
        rendered = renders[0]  # [H, W, 3]

        # Loss
        l1_loss = F.l1_loss(rendered, gt_img)
        # SSIM on [B,C,H,W]
        rendered_bchw = rendered.permute(2, 0, 1).unsqueeze(0)
        gt_bchw = gt_img.permute(2, 0, 1).unsqueeze(0)
        ssim_val = ssim(rendered_bchw, gt_bchw)
        loss = 0.8 * l1_loss + 0.2 * (1.0 - ssim_val)

        loss.backward()

        with torch.no_grad():
            # Strategy step (densification / pruning)
            strat.step_pre_backward(
                params=params,
                optimizers=[optimizer],
                state=strat_state,
                step=step,
                info=info,
            )

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        # Strategy post-step
        with torch.no_grad():
            strat.step_post_backward(
                params=params,
                optimizers=[optimizer],
                state=strat_state,
                step=step,
                info=info,
            )

        # Refresh references after potential densification
        means = params[0]["params"][0]
        sh_coeffs = params[1]["params"][0]
        log_scales = params[2]["params"][0]
        quats = params[3]["params"][0]
        opacities_logit = params[4]["params"][0]

        # Log
        if step % 500 == 0 or step == 1:
            n_gs = means.shape[0]
            print(f"  [Step {step:>6d}/{iterations}] loss={loss.item():.4f} "
                  f"l1={l1_loss.item():.4f} ssim={ssim_val.item():.4f} "
                  f"n_gaussians={n_gs}")

        # Save checkpoint
        if step in save_iters:
            ply_path = os.path.join(
                model_dir, "point_cloud", f"iteration_{step}", "point_cloud.ply"
            )
            _save_checkpoint(ply_path, means, log_scales, quats, opacities_logit, sh_coeffs)

    print(f"  Training complete. Final Gaussians: {means.shape[0]}")
    return model_dir


def _save_checkpoint(ply_path, means, log_scales, quats, opacities_logit, sh_coeffs):
    with torch.no_grad():
        save_ply(
            ply_path,
            means=means.detach().cpu().numpy(),
            scales=torch.exp(log_scales).detach().cpu().numpy(),
            rotations=(quats / quats.norm(dim=-1, keepdim=True)).detach().cpu().numpy(),
            opacities=torch.sigmoid(opacities_logit).detach().cpu().numpy().reshape(-1, 1),
            sh_coeffs=sh_coeffs.detach().cpu().numpy(),
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train 3DGS with gsplat")
    parser.add_argument("--colmap-dir", required=True, help="COLMAP directory")
    parser.add_argument("--model-dir", required=True, help="Output model directory")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--strategy", choices=["mcmc", "default"], default="mcmc")
    args = parser.parse_args()
    train(args.colmap_dir, args.model_dir, args.iterations, args.resolution, args.strategy)
