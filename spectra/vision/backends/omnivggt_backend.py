"""OmniVGGT back-end: pose/intrinsics-*conditioned* feed-forward reconstruction.

OmniVGGT (``Livioni/OmniVGGT``) is VGGT extended so that an arbitrary subset of
auxiliary geometric modalities — camera **extrinsics**, **intrinsics**, and
**depth** — can be injected as *conditioning* into the forward pass, instead of
being ignored. Where the plain VGGT back-end predicts everything from images
alone (and the known robot poses only enter downstream, in the ArUco stage),
this back-end feeds the known poses + intrinsics *into the network* so the
reconstruction is guided by them.

What actually happens under the hood (see the upstream aggregator):

* The conditioning extrinsics are **internally normalized** — re-anchored to the
  first conditioned camera and scale-normalized by the mean inter-camera
  distance. So OmniVGGT consumes the *relative* pose geometry (rotations +
  translation directions + relative scale), **not** an absolute metric frame.
  Metric scale and gravity are still recovered downstream by the ArUco Sim3 /
  bundle-adjustment stage, exactly as for the plain VGGT path.
* The conditioning intrinsics enter as field-of-view in the pose encoding, so
  they must be expressed in **network** pixels (the resized/cropped 518px
  image), not original-resolution pixels — we scale them here.
* Depth conditioning is optional and we do **not** supply it (the robot gives
  poses + intrinsics, not dense metric depth), so ``depth_gt_index`` is empty.

The decoded output (``pose_encoding_to_extri_intri`` → ``[S,3,4]`` world→cam +
``[S,3,3]`` network intrinsics + dense depth) is byte-for-byte the same shape as
the plain VGGT back-end, so the entire post-forward fusion is shared via
:func:`spectra.vision.backends.vggt_backend._pack_raw_reconstruction`.

IMPORTANT — coordinate convention: OmniVGGT expects **OpenCV** camera axes
(x-right, y-down, z-forward) and camera-to-world extrinsics in the ``.txt``
sense. We invert the robot's cam2world to world2cam before feeding it in. If the
hand-eye calibration (``camera2ee.npy``) defines the camera frame with a
different axis convention, the conditioning will fight the images — verify this
on a small scene first (see the module tests / README notes).
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
from PIL import Image

from ..config import ReconstructionConfig, VggtConfig
from .types import BackendInputs, RawReconstruction
from .vggt_backend import (
    VggtImageGeometry,
    _compute_crop_geometry,
    _pack_raw_reconstruction,
    _read_image_rgb,
    _to_numpy,
)

# OmniVGGT weights are trained at a fixed network resolution / patch size
# (see OmniVGGT.__init__ img_size=518, patch_size=14 and
# omnivggt.utils.load_fn.load_and_preprocess_images target_size=518). These are
# fixed by the checkpoint, so vggt.image_resolution / patch_size / dtype from
# config are ignored for impl='omni'.
OMNI_RESOLUTION = 518
OMNI_PATCH = 14


# =============================================================================
# Model loading
# =============================================================================


def _load_omnivggt(vcfg: VggtConfig, device: torch.device):
    """Instantiate OmniVGGT and load weights (safetensors checkpoint or HF)."""
    try:
        from omnivggt.models.omnivggt import OmniVGGT  # type: ignore
    except Exception as exc:  # pragma: no cover - env-specific
        raise RuntimeError(
            "Could not import 'omnivggt'. Install Livioni/OmniVGGT into the "
            "environment (pip install -e . inside the OmniVGGT repo), or choose "
            f"a different vggt.impl. Original error: {exc}"
        ) from exc

    if vcfg.checkpoint_path:
        model = OmniVGGT()
        ckpt = str(vcfg.checkpoint_path)
        if ckpt.endswith(".safetensors"):
            from safetensors.torch import load_file  # type: ignore

            state = load_file(ckpt)
        else:
            state = torch.load(ckpt, map_location="cpu")
            state = state.get("model", state) if isinstance(state, dict) else state
        model.load_state_dict(state, strict=True)
    else:
        if not hasattr(OmniVGGT, "from_pretrained"):
            raise RuntimeError(
                "OmniVGGT has no from_pretrained; set vggt.checkpoint_path to a "
                "local OmniVGGT.safetensors instead."
            )
        model = OmniVGGT.from_pretrained(vcfg.model_name)

    return model.to(device).eval()


# =============================================================================
# Preprocessing + conditioning tensors
# =============================================================================


def _preprocess_images(orig_rgb: List[np.ndarray]) -> torch.Tensor:
    """Replicate OmniVGGT's 'crop' preprocessing (width→518, height rounded to a
    multiple of 14, centre-crop to 518), returning an ``[S,3,H,W]`` float tensor
    in ``[0, 1]``.

    We replicate it inline (rather than call the upstream loader) so the frame
    order is exactly the caller's order — the upstream loader re-sorts its file
    list, which would silently misalign images against the per-view GT poses.
    ResNet mean/std normalization is applied *inside* the model aggregator, so
    a plain [0,1] tensor is what the network expects here.
    """
    from torchvision import transforms as TF  # local import, torchvision is heavy

    to_tensor = TF.ToTensor()
    tensors: List[torch.Tensor] = []
    shapes = set()
    for rgb in orig_rgb:
        pil = Image.fromarray(rgb).convert("RGB")
        w0, h0 = pil.size
        new_w = OMNI_RESOLUTION
        new_h = round(h0 * (new_w / w0) / OMNI_PATCH) * OMNI_PATCH
        pil = pil.resize((new_w, new_h), Image.Resampling.BICUBIC)
        t = to_tensor(pil)  # [3, new_h, new_w] in [0,1]
        if new_h > OMNI_RESOLUTION:
            start_y = (new_h - OMNI_RESOLUTION) // 2
            t = t[:, start_y : start_y + OMNI_RESOLUTION, :]
        tensors.append(t)
        shapes.add((int(t.shape[1]), int(t.shape[2])))

    if len(shapes) > 1:
        raise RuntimeError(
            "OmniVGGT back-end requires all input images to share one resolution "
            f"(aspect ratio); got network shapes {sorted(shapes)}. Resize/crop the "
            "inputs to a common size first."
        )
    return torch.stack(tensors, dim=0)


def _original_to_network_intrinsics(geom: VggtImageGeometry, K_orig: np.ndarray) -> np.ndarray:
    """Scale an original-pixel intrinsic matrix into the network crop's pixels.

    This is the exact inverse of
    :meth:`VggtImageGeometry.network_to_original_intrinsics` and mirrors the
    scaling done by OmniVGGT's own ``load_images_and_cameras``:
    ``fx,cx *= scale_x``; ``fy,cy *= scale_y``; then subtract the crop offset
    from the principal point.
    """
    K = K_orig.astype(np.float32).copy()
    K[0, 0] *= geom.scale_x
    K[0, 2] = K[0, 2] * geom.scale_x - geom.crop_left
    K[1, 1] *= geom.scale_y
    K[1, 2] = K[1, 2] * geom.scale_y - geom.crop_top
    return K


def _build_conditioning(
    inputs: BackendInputs,
    geometries: List[VggtImageGeometry],
    num_views: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Build the (extrinsics, intrinsics, camera_gt_index) conditioning inputs.

    Camera conditioning needs *both* GT poses and GT intrinsics (the pose
    encoding fuses them). When either is missing we return an empty index list,
    which makes OmniVGGT fall back to pure image-only prediction (i.e. it
    behaves like VGGT with the OmniVGGT weights).

    Shapes match OmniVGGT's ``load_images_and_cameras``:
    ``extrinsics [1,S,3,4]`` world→cam, ``intrinsics [1,S,3,3]`` network pixels.
    """
    have_poses = inputs.T_world_cam_gt is not None
    have_intr = inputs.K_orig_gt is not None

    extrinsics = np.zeros((num_views, 3, 4), dtype=np.float32)
    intrinsics = np.zeros((num_views, 3, 3), dtype=np.float32)
    camera_gt_index: list[int] = []

    if have_poses and have_intr:
        for i in range(num_views):
            # robot pose is camera-to-world; OmniVGGT wants world-to-camera.
            w2c = np.linalg.inv(inputs.T_world_cam_gt[i].astype(np.float32))
            extrinsics[i] = w2c[:3, :4]
            intrinsics[i] = _original_to_network_intrinsics(geometries[i], inputs.K_orig_gt)
        camera_gt_index = list(range(num_views))

    ext_t = torch.from_numpy(extrinsics[None]).float().to(device)   # [1,S,3,4]
    intr_t = torch.from_numpy(intrinsics[None]).float().to(device)  # [1,S,3,3]
    return ext_t, intr_t, camera_gt_index


# =============================================================================
# Main entry point
# =============================================================================


def run_omnivggt(cfg: ReconstructionConfig, inputs: BackendInputs) -> RawReconstruction:
    """Run OmniVGGT (pose/intrinsics-conditioned) reconstruction.

    Mirrors :func:`run_vggt` but (a) always uses the fixed 518px/patch-14 'crop'
    preprocessing OmniVGGT was trained on, and (b) feeds the known camera poses
    and intrinsics into ``model.inference`` as conditioning when they are
    available. Packs the decoded predictions into a ``RawReconstruction`` via
    the shared fusion helper.
    """
    vcfg: VggtConfig = cfg.vggt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    image_paths = list(inputs.image_paths)
    if len(image_paths) < 2:
        raise RuntimeError("OmniVGGT backend needs at least two images.")

    # Optional memory guard: uniformly subsample frames, keeping the GT poses in
    # lock-step so conditioning stays aligned with the images.
    T_world_cam_gt = inputs.T_world_cam_gt
    if vcfg.max_frames is not None and len(image_paths) > vcfg.max_frames:
        keep = np.linspace(0, len(image_paths) - 1, vcfg.max_frames).round().astype(int)
        keep = sorted(set(int(i) for i in keep))
        image_paths = [image_paths[i] for i in keep]
        if T_world_cam_gt is not None:
            T_world_cam_gt = T_world_cam_gt[keep]
    num_views = len(image_paths)

    # Work on a shallow copy of inputs so the (possibly subsampled) poses are
    # what the conditioning + shared fusion helper see.
    from dataclasses import replace

    sub_inputs = replace(inputs, image_paths=image_paths, T_world_cam_gt=T_world_cam_gt)

    # --- Original RGB + crop geometry (fixed 518/14 for OmniVGGT) ----------
    orig_rgb = [_read_image_rgb(p) for p in image_paths]  # [H0,W0,3] uint8
    geometries = [
        _compute_crop_geometry(
            original_width=rgb.shape[1],
            original_height=rgb.shape[0],
            resolution=OMNI_RESOLUTION,
            patch_size=OMNI_PATCH,
        )
        for rgb in orig_rgb
    ]

    # --- Load model + preprocess -------------------------------------------
    model = _load_omnivggt(vcfg, device)
    images = _preprocess_images(orig_rgb).to(device)  # [S,3,H,W] in [0,1]
    network_h, network_w = int(images.shape[-2]), int(images.shape[-1])

    # --- Conditioning tensors ----------------------------------------------
    extrinsics_t, intrinsics_t, camera_gt_index = _build_conditioning(
        sub_inputs, geometries, num_views, device
    )
    # We do not supply GT depth; still need real (zero) tensors because
    # OmniVGGT.inference reads depth.device even when depth_gt_index is empty.
    depth_t = torch.zeros((1, num_views, network_h, network_w, 1), dtype=torch.float32, device=device)
    mask_t = torch.zeros((1, num_views, network_h, network_w), dtype=torch.float32, device=device)
    depth_gt_index: list[int] = []

    if camera_gt_index:
        print(
            f"[green]OmniVGGT: conditioning on {num_views} known camera poses + "
            "intrinsics.[/green]"
        )
    else:
        print(
            "[yellow]OmniVGGT: no GT poses+intrinsics available; running "
            "image-only (unconditioned, like VGGT).[/yellow]"
        )

    # --- Forward -----------------------------------------------------------
    # Follow the upstream inference.py: plain fp32, no autocast (the model
    # disables autocast around its heads internally anyway).
    with torch.no_grad():
        predictions = model.inference(
            images=images,
            extrinsics=extrinsics_t,
            intrinsics=intrinsics_t,
            depth=depth_t,
            mask=mask_t,
            depth_gt_index=depth_gt_index,
            camera_gt_index=camera_gt_index,
        )

        from omnivggt.utils.pose_enc import pose_encoding_to_extri_intri  # type: ignore

        pose_enc = predictions["pose_enc"]
        if pose_enc.ndim == 2:  # [S,9] -> [1,S,9]
            pose_enc = pose_enc.unsqueeze(0)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, (network_h, network_w))

    extrinsic_np = _to_numpy(extrinsic).reshape(-1, 3, 4)   # [S,3,4] world->cam
    intrinsic_np = _to_numpy(intrinsic).reshape(-1, 3, 3)   # [S,3,3] network px

    def _squeeze_batch(arr: np.ndarray) -> np.ndarray:
        return arr[0] if arr.ndim >= 4 and arr.shape[0] == 1 else arr

    depth_np = _squeeze_batch(_to_numpy(predictions["depth"]))   # [S,H,W,1] or [S,H,W]
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
                "[yellow]OmniVGGT: point_source='world' requested but the model has "
                "no 'world_points' head; falling back to depth unprojection.[/yellow]"
            )

    # Free GPU memory before the CPU-side fusion.
    del predictions, model, images, extrinsics_t, intrinsics_t, depth_t, mask_t
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return _pack_raw_reconstruction(
        vcfg=vcfg,
        inputs=sub_inputs,
        orig_rgb=orig_rgb,
        geometries=geometries,
        network_h=network_h,
        network_w=network_w,
        extrinsic_np=extrinsic_np,
        intrinsic_np=intrinsic_np,
        depth_np=depth_np,
        depth_conf_np=depth_conf_np,
        world_points_np=world_points_np,
        world_conf_np=world_conf_np,
        backend_name="omnivggt",
        impl_label="omni",
        extra_info={
            "model_name": vcfg.model_name,
            "image_resolution": OMNI_RESOLUTION,
            "num_views": num_views,
            "conditioned_on_cameras": bool(camera_gt_index),
        },
    )


__all__ = ["run_omnivggt"]
