"""VGGT back-end: feed-forward pose + depth prediction, dense fusion.

VGGT (Visual Geometry Grounded Transformer,
``facebookresearch/vggt-omega``) predicts, in a *single forward pass* over a
set of images, per-view camera extrinsics + intrinsics and dense depth, with
the world frame anchored to camera 0. Unlike the former MASt3R-SfM back-end
there is no pairwise matching or iterative global alignment: the network
outputs a consistent multi-view reconstruction directly.

This module runs VGGT and packs the result into the backend-agnostic
:class:`RawReconstruction` the orchestrator consumes. As with the old SfM
path, the reconstruction lives in VGGT's own (arbitrary-scale, camera-0)
frame; metric scale and gravity are recovered downstream by the ArUco Sim3 /
bundle-adjustment stage.

The library API differs slightly between the ``vggt-omega`` and the original
``vggt`` packages (class name, weight loading, pose-decoder helper). We select
the right imports from ``cfg.vggt.impl`` and keep defensive fallbacks so a
minor upstream rename can be handled from config instead of code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from PIL.ImageOps import exif_transpose

from ..config import ReconstructionConfig, VggtConfig
from .types import BackendInputs, RawReconstruction


# =============================================================================
# Image geometry (original ⇄ network pixel mapping for 'crop' preprocessing)
# =============================================================================


@dataclass(frozen=True)
class VggtImageGeometry:
    """Maps between original-resolution pixels and the network crop.

    VGGT 'crop' preprocessing locks the width to ``image_resolution`` (aspect
    preserved by a uniform resize), rounds the resized height to a multiple of
    the patch size, then centre-crops the height down to at most
    ``image_resolution``. Because the resize is uniform, intrinsics scale by
    ``scale_x``/``scale_y`` and shift by the crop offset — exactly like the
    MASt3R geometry helper this replaces.
    """

    original_height: int
    original_width: int
    network_height: int
    network_width: int
    scale_x: float
    scale_y: float
    crop_left: float
    crop_top: float

    def network_to_original_intrinsics(self, K_net: np.ndarray) -> np.ndarray:
        """Pull a network-frame intrinsic matrix back into original pixels."""
        A_inv = np.array(
            [
                [1.0 / self.scale_x, 0.0, self.crop_left / self.scale_x],
                [0.0, 1.0 / self.scale_y, self.crop_top / self.scale_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return A_inv @ K_net.astype(np.float32)

    def network_grid_to_original(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (uu, vv) original-pixel coords for every network pixel."""
        uu, vv = np.meshgrid(
            np.arange(self.network_width, dtype=np.float32),
            np.arange(self.network_height, dtype=np.float32),
        )
        uu = (uu + self.crop_left) / self.scale_x
        vv = (vv + self.crop_top) / self.scale_y
        return uu, vv


def _compute_crop_geometry(
    original_width: int,
    original_height: int,
    resolution: int,
    patch_size: int,
) -> VggtImageGeometry:
    """Replicate VGGT 'crop' preprocessing to recover the pixel mapping."""
    W0, H0 = int(original_width), int(original_height)
    new_width = resolution
    scale = resolution / float(W0)
    resized_height = int(round(H0 * scale / patch_size) * patch_size)

    crop_top = 0.0
    network_height = resized_height
    if resized_height > resolution:
        crop_top = float((resized_height - resolution) // 2)
        network_height = resolution

    return VggtImageGeometry(
        original_height=H0,
        original_width=W0,
        network_height=network_height,
        network_width=new_width,
        scale_x=new_width / float(W0),
        scale_y=resized_height / float(H0),
        crop_left=0.0,
        crop_top=crop_top,
    )


# =============================================================================
# Shared point-cloud helpers (mirror the former MASt3R back-end)
# =============================================================================


def _read_image_rgb(path: Path) -> np.ndarray:
    """Load an image as an [H, W, 3] uint8 RGB array with EXIF applied."""
    with Image.open(path) as pil_img:
        rgb = exif_transpose(pil_img).convert("RGB")
        return np.asarray(rgb, dtype=np.uint8)


def _weighted_voxel_downsample(
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(points) == 0 or voxel_size <= 0:
        return points, colors, confidence

    keys = np.floor(points / voxel_size).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    counts = np.bincount(inv, minlength=len(uniq))
    weights = np.clip(confidence.astype(np.float64), 1e-6, None)
    weight_sum = np.bincount(inv, weights=weights, minlength=len(uniq))

    out_pts = np.zeros((len(uniq), 3), dtype=np.float64)
    out_cols = np.zeros((len(uniq), 3), dtype=np.float64)
    out_conf = np.bincount(inv, weights=confidence.astype(np.float64), minlength=len(uniq)) / np.maximum(counts, 1)

    for d in range(3):
        out_pts[:, d] = np.bincount(inv, weights=weights * points[:, d], minlength=len(uniq)) / np.maximum(
            weight_sum, 1e-12
        )
        out_cols[:, d] = np.bincount(inv, weights=weights * colors[:, d], minlength=len(uniq)) / np.maximum(
            weight_sum, 1e-12
        )

    return out_pts.astype(np.float32), np.clip(out_cols, 0, 255).astype(np.uint8), out_conf.astype(np.float32)


def _pad_view_maps_for_stacking(
    point_map_world: List[np.ndarray],
    valid_masks: List[np.ndarray],
    images_network_uint8: List[np.ndarray],
    confidence_maps_network: List[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pad heterogeneous per-view maps to a common [V, Hmax, Wmax, ...] shape."""
    max_h = max(int(mask.shape[0]) for mask in valid_masks)
    max_w = max(int(mask.shape[1]) for mask in valid_masks)

    point_map_world_np = np.full((len(point_map_world), max_h, max_w, 3), np.nan, dtype=np.float32)
    valid_masks_np = np.zeros((len(valid_masks), max_h, max_w), dtype=bool)
    images_net_np = np.zeros((len(images_network_uint8), max_h, max_w, 3), dtype=np.uint8)
    conf_maps_np = np.zeros((len(confidence_maps_network), max_h, max_w), dtype=np.float32)

    for idx, (pts, mask, rgb, conf) in enumerate(
        zip(point_map_world, valid_masks, images_network_uint8, confidence_maps_network)
    ):
        h, w = int(mask.shape[0]), int(mask.shape[1])
        point_map_world_np[idx, :h, :w, :] = pts.astype(np.float32, copy=False)
        valid_masks_np[idx, :h, :w] = mask.astype(bool, copy=False)
        images_net_np[idx, :h, :w, :] = rgb.astype(np.uint8, copy=False)
        conf_maps_np[idx, :h, :w] = conf.astype(np.float32, copy=False)

    return point_map_world_np, valid_masks_np, images_net_np, conf_maps_np


def _unproject_depth_to_world(
    depth: np.ndarray,      # [H, W] float32
    K_net: np.ndarray,      # [3, 3] network intrinsics
    T_world_cam: np.ndarray,  # [4, 4] camera-to-world
) -> np.ndarray:
    """Back-project a depth map to world-frame 3D points, OpenCV convention.

    Uses the *network* intrinsics (the frame the depth was predicted in) and
    the camera-to-world pose. Returns an [H, W, 3] float32 point map.
    """
    H, W = depth.shape
    fx, fy = float(K_net[0, 0]), float(K_net[1, 1])
    cx, cy = float(K_net[0, 2]), float(K_net[1, 2])

    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    x_cam = (uu - cx) / fx * depth
    y_cam = (vv - cy) / fy * depth
    z_cam = depth
    cam_pts = np.stack([x_cam, y_cam, z_cam], axis=-1).reshape(-1, 3)  # [H*W, 3]

    R_wc = T_world_cam[:3, :3].astype(np.float32)
    t_wc = T_world_cam[:3, 3].astype(np.float32)
    world_pts = cam_pts @ R_wc.T + t_wc  # [H*W, 3]
    return world_pts.reshape(H, W, 3).astype(np.float32)


# =============================================================================
# Model loading + forward (impl-specific, defensive)
# =============================================================================


def _torch_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _load_model_and_preprocess(vcfg: VggtConfig, filelist: List[str], device: torch.device):
    """Load the VGGT model and preprocess images. Returns (model, images_tensor).

    ``images_tensor`` is [S, 3, Hn, Wn] float in [0, 1] on ``device``.
    """
    if vcfg.impl == "omega":
        try:
            from vggt_omega.models import VGGTOmega  # type: ignore
            from vggt_omega.utils.load_fn import load_and_preprocess_images  # type: ignore
        except Exception as exc:  # pragma: no cover - env-specific
            raise RuntimeError(
                "Could not import 'vggt_omega'. Install facebookresearch/vggt-omega "
                "into the environment, or set vggt.impl='vggt' to use the original "
                f"vggt package. Original error: {exc}"
            ) from exc

        if vcfg.checkpoint_path:
            model = VGGTOmega()
            state = torch.load(vcfg.checkpoint_path, map_location="cpu")
            state = state.get("model", state) if isinstance(state, dict) else state
            model.load_state_dict(state)
        else:
            # Prefer from_pretrained when the HF mixin is available.
            if hasattr(VGGTOmega, "from_pretrained"):
                model = VGGTOmega.from_pretrained(vcfg.model_name)
            else:
                raise RuntimeError(
                    "vggt-omega weights: set vggt.checkpoint_path to a local .pt "
                    "checkpoint (VGGTOmega has no from_pretrained)."
                )
       # vggt-omega usa un vocabolario diverso ("balanced"/"max_size") rispetto
        # al nostro crop/pad (che guida invece la logica di remap delle intrinsics
        # sotto, riga ~384). Traduciamo prima di chiamare la libreria omega.
        _omega_mode = "balanced" if vcfg.preprocess_mode == "crop" else "max_size"
        try:
            images = load_and_preprocess_images(
                filelist, image_resolution=vcfg.image_resolution, mode=_omega_mode
            )
        except TypeError:
            images = load_and_preprocess_images(filelist, image_resolution=vcfg.image_resolution)
    else:  # impl == "vggt"
        try:
            from vggt.models.vggt import VGGT  # type: ignore
            from vggt.utils.load_fn import load_and_preprocess_images  # type: ignore
        except Exception as exc:  # pragma: no cover - env-specific
            raise RuntimeError(
                "Could not import 'vggt'. Install facebookresearch/vggt, or set "
                f"vggt.impl='omega'. Original error: {exc}"
            ) from exc

        if vcfg.checkpoint_path:
            model = VGGT()
            state = torch.load(vcfg.checkpoint_path, map_location="cpu")
            state = state.get("model", state) if isinstance(state, dict) else state
            model.load_state_dict(state)
        else:
            model = VGGT.from_pretrained(vcfg.model_name)
        images = load_and_preprocess_images(filelist, mode=vcfg.preprocess_mode)

    model = model.to(device).eval()
    images = images.to(device)
    return model, images


def _decode_pose_encoding(vcfg: VggtConfig, pose_enc: torch.Tensor, image_hw: tuple[int, int]):
    """Decode the pose encoding to (extrinsic [B,S,3,4] w2c, intrinsic [B,S,3,3])."""
    if vcfg.impl == "omega":
        try:
            from vggt_omega.utils.pose_enc import encoding_to_camera  # type: ignore
        except Exception:
            # Fall back to the vggt spelling if omega re-exports it.
            from vggt.utils.pose_enc import (  # type: ignore
                pose_encoding_to_extri_intri as encoding_to_camera,
            )
        return encoding_to_camera(pose_enc, image_hw)

    from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # type: ignore

    return pose_encoding_to_extri_intri(pose_enc, image_hw)


def _to_numpy(t) -> np.ndarray:
    return t.detach().float().cpu().numpy()


# =============================================================================
# Main entry point
# =============================================================================


def run_vggt(cfg: ReconstructionConfig, inputs: BackendInputs) -> RawReconstruction:
    """Run VGGT feed-forward reconstruction and pack a ``RawReconstruction``."""
    vcfg: VggtConfig = cfg.vggt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    image_paths = list(inputs.image_paths)
    if len(image_paths) < 2:
        raise RuntimeError("VGGT backend needs at least two images.")

    # Optional memory guard: uniformly subsample frames.
    if vcfg.max_frames is not None and len(image_paths) > vcfg.max_frames:
        keep = np.linspace(0, len(image_paths) - 1, vcfg.max_frames).round().astype(int)
        keep = sorted(set(int(i) for i in keep))
        image_paths = [image_paths[i] for i in keep]
    num_views = len(image_paths)
    filelist = [str(p) for p in image_paths]

    # --- Load model + preprocess -------------------------------------------
    model, images = _load_model_and_preprocess(vcfg, filelist, device)
    network_h, network_w = int(images.shape[-2]), int(images.shape[-1])

    # --- Forward -----------------------------------------------------------
    use_amp = device.type == "cuda" and vcfg.dtype != "fp32"
    with torch.no_grad():
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=_torch_dtype(vcfg.dtype)):
                predictions = model(images)
        else:
            predictions = model(images)

        pose_enc = predictions["pose_enc"]
        if pose_enc.ndim == 2:  # [S,9] -> [1,S,9]
            pose_enc = pose_enc.unsqueeze(0)
        extrinsic, intrinsic = _decode_pose_encoding(vcfg, pose_enc, (network_h, network_w))

    extrinsic_np = _to_numpy(extrinsic).reshape(-1, 3, 4)      # [S,3,4] world->cam
    intrinsic_np = _to_numpy(intrinsic).reshape(-1, 3, 3)      # [S,3,3] network px

    def _squeeze_batch(arr: np.ndarray) -> np.ndarray:
        # predictions come as [B,S,...] with B=1; drop the batch dim.
        return arr[0] if arr.ndim >= 4 and arr.shape[0] == 1 else arr

    depth_np = _squeeze_batch(_to_numpy(predictions["depth"]))          # [S,H,W,1] or [S,H,W]
    if depth_np.ndim == 4 and depth_np.shape[-1] == 1:
        depth_np = depth_np[..., 0]
    depth_conf_np = _squeeze_batch(_to_numpy(predictions["depth_conf"]))  # [S,H,W]

    world_points_np = None
    world_conf_np = None
    if vcfg.point_source == "world":
        if "world_points" in predictions:
            world_points_np = _squeeze_batch(_to_numpy(predictions["world_points"]))  # [S,H,W,3]
            if "world_points_conf" in predictions:
                world_conf_np = _squeeze_batch(_to_numpy(predictions["world_points_conf"]))
        else:
            print(
                "[yellow]VGGT: point_source='world' requested but the model has no "
                "'world_points' head; falling back to depth unprojection.[/yellow]"
            )

    # Free GPU memory before the CPU-side fusion.
    del predictions, model, images
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- Camera-to-world poses (RawReconstruction expects cam2world) -------
    T_world_cam = np.zeros((num_views, 4, 4), dtype=np.float32)
    for i in range(num_views):
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :4] = extrinsic_np[i]
        T_world_cam[i] = np.linalg.inv(w2c).astype(np.float32)

    # --- Per-view geometry + original RGB ----------------------------------
    have_gt_intrinsics = vcfg.use_gt_intrinsics and inputs.K_orig_gt is not None
    if vcfg.preprocess_mode != "crop" and not have_gt_intrinsics:
        raise RuntimeError(
            "vggt.preprocess_mode='pad' has no network→original intrinsics remap; "
            "either use preprocess_mode='crop' or provide GT intrinsics."
        )

    orig_rgb = [_read_image_rgb(p) for p in image_paths]  # [H0,W0,3] uint8
    geometries = [
        _compute_crop_geometry(
            original_width=rgb.shape[1],
            original_height=rgb.shape[0],
            resolution=vcfg.image_resolution,
            patch_size=vcfg.patch_size,
        )
        for rgb in orig_rgb
    ]
    for i, geom in enumerate(geometries):
        if (geom.network_height, geom.network_width) != (network_h, network_w):
            raise RuntimeError(
                f"View {i}: computed crop geometry {(geom.network_height, geom.network_width)} "
                f"!= actual network tensor {(network_h, network_w)}. Check vggt.patch_size "
                f"({vcfg.patch_size}) / image_resolution ({vcfg.image_resolution}) match the checkpoint."
            )

    # --- Build per-view point maps, colours, confidence, masks -------------
    all_pts_list: List[np.ndarray] = []
    all_cols_list: List[np.ndarray] = []
    all_conf_list: List[np.ndarray] = []
    valid_masks: List[np.ndarray] = []
    point_map_world: List[np.ndarray] = []
    images_network_uint8: List[np.ndarray] = []
    confidence_maps_network: List[np.ndarray] = []

    for i, (geom, rgb) in enumerate(zip(geometries, orig_rgb)):
        conf_map = depth_conf_np[i].astype(np.float32)
        depth_map = depth_np[i].astype(np.float32)

        if world_points_np is not None:
            pts3d = world_points_np[i].astype(np.float32)
            if world_conf_np is not None:
                conf_map = world_conf_np[i].astype(np.float32)
        else:
            pts3d = _unproject_depth_to_world(depth_map, intrinsic_np[i], T_world_cam[i])

        # Colours: sample the ORIGINAL image at each network pixel (independent
        # of the model's input normalization; keeps crisp full-res colour).
        uu_orig, vv_orig = geom.network_grid_to_original()
        u_idx = np.clip(np.round(uu_orig).astype(np.int32), 0, geom.original_width - 1)
        v_idx = np.clip(np.round(vv_orig).astype(np.int32), 0, geom.original_height - 1)
        cols_full = rgb[v_idx, u_idx].astype(np.uint8)

        finite = np.isfinite(pts3d).all(axis=-1) & np.isfinite(depth_map) & (depth_map > 1e-6)
        if vcfg.conf_threshold is not None:
            thr = float(vcfg.conf_threshold)
        else:
            pool = conf_map[finite]
            thr = float(np.percentile(pool, vcfg.conf_percentile)) if pool.size else 0.0
        valid = finite & (conf_map >= thr)
        if not np.any(valid):
            valid = finite

        all_pts_list.append(pts3d[valid].astype(np.float32))
        all_cols_list.append(cols_full[valid].astype(np.uint8))
        all_conf_list.append(conf_map[valid].astype(np.float32))
        valid_masks.append(valid)
        point_map_world.append(pts3d)
        images_network_uint8.append(cols_full)
        confidence_maps_network.append(conf_map)

    if not any(len(p) > 0 for p in all_pts_list):
        raise RuntimeError("VGGT dense extraction produced no valid points.")

    fused_points = np.concatenate(all_pts_list, axis=0)
    fused_colors = np.concatenate(all_cols_list, axis=0)
    fused_confidence = np.concatenate(all_conf_list, axis=0)

    if vcfg.voxel_size > 0.0:
        fused_points, fused_colors, fused_confidence = _weighted_voxel_downsample(
            fused_points, fused_colors, fused_confidence, vcfg.voxel_size
        )
    if len(fused_points) > vcfg.max_points:
        keep = np.argsort(fused_confidence)[-vcfg.max_points:][::-1]
        fused_points = fused_points[keep]
        fused_colors = fused_colors[keep]
        fused_confidence = fused_confidence[keep]

    point_map_world_np, valid_masks_np, images_net_np, conf_maps_np = _pad_view_maps_for_stacking(
        point_map_world=point_map_world,
        valid_masks=valid_masks,
        images_network_uint8=images_network_uint8,
        confidence_maps_network=confidence_maps_network,
    )

    # --- Intrinsics in original pixel space --------------------------------
    K_net_per_view = intrinsic_np.astype(np.float32)
    if have_gt_intrinsics:
        K_orig_per_view = np.repeat(
            inputs.K_orig_gt[None].astype(np.float32), num_views, axis=0
        )
    else:
        K_orig_per_view = np.stack(
            [geom.network_to_original_intrinsics(K_net_per_view[i]) for i, geom in enumerate(geometries)],
            axis=0,
        ).astype(np.float32)

    network_image_sizes = np.asarray(
        [[g.network_width, g.network_height] for g in geometries], dtype=np.int32
    )
    original_image_sizes = np.asarray(
        [[g.original_width, g.original_height] for g in geometries], dtype=np.int32
    )

    return RawReconstruction(
        fused_points=fused_points,
        fused_colors=fused_colors,
        fused_confidence=fused_confidence,
        point_map_world=point_map_world_np,
        valid_masks=valid_masks_np,
        T_world_cam=T_world_cam,
        K_per_view_orig=K_orig_per_view,
        K_per_view_network=K_net_per_view,
        network_image_sizes=network_image_sizes,
        original_image_sizes=original_image_sizes,
        images_network_uint8=images_net_np,
        confidence_maps_network=conf_maps_np,
        frame_description="vggt",
        backend_name="vggt",
        alignment_info={
            "impl": vcfg.impl,
            "point_source": "world" if world_points_np is not None else "depth",
        },
        extra={
            "impl": vcfg.impl,
            "model_name": vcfg.model_name,
            "image_resolution": vcfg.image_resolution,
            "point_source": "world" if world_points_np is not None else "depth",
            "have_gt_intrinsics": bool(have_gt_intrinsics),
            "num_views": num_views,
        },
    )


__all__ = ["VggtImageGeometry", "run_vggt"]
