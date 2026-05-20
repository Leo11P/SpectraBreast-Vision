"""Mode dispatcher for the registration pipeline.

This module is the single entry point used by ``spectra registration`` (and by
``spectra full`` via :mod:`spectra.orchestrator`). It reads a
:class:`~spectra.config.RegistrationConfig` and runs the right backend:

    mode="single" → :func:`spectra.registration.pipeline_roi.run_full_pipeline_roi`
                    with ``liveview_png_path=None`` (delegates to
                    ``run_full_pipeline`` internally — see pipeline_roi.py)
    mode="roi"    → :func:`spectra.registration.pipeline_roi.run_full_pipeline_roi`
                    with the LiveView PNG enabled
    mode="batch"  → :func:`spectra.registration.batch.run_batch`
    mode="sweep"  → :func:`spectra.registration.sweep.run_sweep`

The dispatcher does NOT auto-derive paths: the orchestrator
(:mod:`spectra.orchestrator`) is responsible for filling
``cfg.paths.{hsi_hdr,mesh,aruco_json,liveview_png,output_dir}`` from
``sample_name`` + ``data_root`` + ``results_root`` BEFORE calling this. When
:func:`run_registration` is invoked directly with missing paths it raises a
clean error.

Behavior is otherwise identical to the legacy ``main.py``.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2

from ..config import RegistrationConfig

# Render GPU is optional (torch / cupy may be absent).
try:
    import torch
    from .render_gpu import render_orthographic_topview_gpu
    _TORCH_AVAILABLE = True
    _CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:
    render_orthographic_topview_gpu = None
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False
    _CUDA_AVAILABLE = False


_ARUCO_DICT_MAP = {
    "4X4_50":  cv2.aruco.DICT_4X4_50,
    "4X4_100": cv2.aruco.DICT_4X4_100,
    "5X5_50":  cv2.aruco.DICT_5X5_50,
    "5X5_100": cv2.aruco.DICT_5X5_100,
    "6X6_50":  cv2.aruco.DICT_6X6_50,
    "6X6_100": cv2.aruco.DICT_6X6_100,
}


# =============================================================================
# Path validation
# =============================================================================

def _missing_paths(cfg: RegistrationConfig) -> list[str]:
    """Return human-readable names of registration paths that are still ``None``."""
    p = cfg.paths
    missing: list[str] = []
    # Always required:
    if p.hsi_hdr is None:
        missing.append("paths.hsi_hdr")
    if cfg.mode != "batch":
        # batch discovers per-sample paths itself
        if p.mesh is None:
            missing.append("paths.mesh")
        if p.aruco_json is None:
            missing.append("paths.aruco_json")
    if p.output_dir is None:
        missing.append("paths.output_dir")
    needs_liveview = (
        cfg.mode == "roi"
        or (cfg.mode == "sweep" and cfg.sweep.roi_mode)
    )
    if needs_liveview and p.liveview_png is None:
        missing.append("paths.liveview_png")
    return missing


def _validate_inputs_exist(cfg: RegistrationConfig) -> list[str]:
    """Return human-readable error messages for missing files on disk."""
    p = cfg.paths
    errors: list[str] = []
    checks = [(p.hsi_hdr, "HSI .hdr")]
    if cfg.mode in ("single", "roi", "sweep"):
        checks += [(p.mesh, "Mesh"), (p.aruco_json, "ArUco JSON")]
    if cfg.mode == "roi" or (cfg.mode == "sweep" and cfg.sweep.roi_mode):
        checks += [(p.liveview_png, "LiveView PNG")]
    for path, label in checks:
        if path is not None and not Path(path).exists():
            errors.append(f"{label} not found: {path}")
    return errors


# =============================================================================
# Helpers shared by single/roi runs
# =============================================================================

def _device_string(force_cpu: bool = False) -> Optional[str]:
    if not _TORCH_AVAILABLE:
        return None
    if force_cpu:
        return "cpu"
    return "cuda" if _CUDA_AVAILABLE else "cpu"


def _print_summary(cfg: RegistrationConfig, sample_name: str) -> None:
    print("\n" + "=" * 68)
    print("SPECTRA REGISTRATION — SUMMARY")
    print("=" * 68)
    print(f"  Mode          : {cfg.mode.upper()}")
    print(f"  Sample        : {sample_name}")
    print(f"  HSI           : {cfg.paths.hsi_hdr}")
    if cfg.mode != "batch":
        print(f"  Mesh          : {cfg.paths.mesh}")
        print(f"  ArUco JSON    : {cfg.paths.aruco_json}")
        if cfg.mode == "roi" or (cfg.mode == "sweep" and cfg.sweep.roi_mode):
            print(f"  LiveView PNG  : {cfg.paths.liveview_png}")
    print(f"  Marker side   : {cfg.detection.marker_side_mm} mm")

    if cfg.mode == "sweep":
        print(f"  Render        : sweep over {len(cfg.sweep.resolution_pairs)} pairs")
        print(f"  ROI sweep     : {'YES' if cfg.sweep.roi_mode else 'NO'}")
    else:
        print(f"  Render        : reg={cfg.render.resolution_reg_mm_per_px} "
              f"pc={cfg.render.resolution_mm_per_px} mm/px")

    print(f"  Save PC       : {'YES' if cfg.export.save_pointcloud else 'NO'}")
    print(f"  Save images   : {'YES' if cfg.export.save_images else 'NO'}")
    print(f"  Output        : {cfg.paths.output_dir}")
    print("=" * 68)


# =============================================================================
# Dispatcher
# =============================================================================

def run_registration(
    cfg: RegistrationConfig,
    *,
    sample_name: str = "sample",
    data_root: Optional[Path] = None,
    results_root: Optional[Path] = None,
    force_cpu: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the registration stage in the mode declared by ``cfg.mode``.

    Parameters
    ----------
    cfg :
        A fully-resolved ``RegistrationConfig`` (paths must already be filled in
        by the caller — the orchestrator does that automatically for
        ``spectra full``).
    sample_name :
        Used as the filename prefix for outputs (Excel report, point cloud,
        registration PNGs).
    data_root, results_root :
        Required only for ``mode == "batch"`` (the batch discoverer scans
        ``data_root`` and writes per-sample outputs under
        ``results_root/<sample>/registration``).
    force_cpu :
        If true, disables GPU raycasting even if torch/cupy are available.
    dry_run :
        If true, validates paths and prints the summary but does not run.

    Returns
    -------
    dict
        Backend-specific result dictionary. For batch/sweep, contains the
        Excel summary path under ``"summary_xlsx"``.
    """
    missing = _missing_paths(cfg)
    if missing:
        raise ValueError(
            "Registration config is missing required paths "
            f"(none of these can be null when running standalone): {missing}. "
            "Either set them explicitly under `registration.paths:` in the YAML, "
            "or run via `spectra full` so the orchestrator derives them from "
            "sample_name + data_root + results_root."
        )

    errors = _validate_inputs_exist(cfg)
    if errors:
        for e in errors:
            print(f"[Validate] MISSING — {e}")
        raise FileNotFoundError(
            f"Registration cannot start: {len(errors)} input file(s) missing."
        )

    _print_summary(cfg, sample_name)

    if dry_run:
        print("\n[Dry-run] Configuration valid; pipeline NOT executed.")
        return {"dry_run": True}

    aruco_dict_cv = _ARUCO_DICT_MAP.get(cfg.detection.aruco_dict.upper(), cv2.aruco.DICT_4X4_50)
    device_str = _device_string(force_cpu=force_cpu)
    use_gpu_render = _TORCH_AVAILABLE and (render_orthographic_topview_gpu is not None)

    os.makedirs(cfg.paths.output_dir, exist_ok=True)  # type: ignore[arg-type]

    # ----- mode == "batch" --------------------------------------------------
    if cfg.mode == "batch":
        if data_root is None or results_root is None:
            raise ValueError(
                "Batch mode requires data_root and results_root to be passed. "
                "These are normally provided by the unified CLI / orchestrator."
            )
        from .batch import run_batch
        xlsx_path = run_batch(
            cfg=cfg,
            aruco_dict_cv=aruco_dict_cv,
            torch_device=device_str,
            use_torch_render=use_gpu_render,
            data_root=Path(data_root),
            results_root=Path(results_root),
            render_fn=render_orthographic_topview_gpu if use_gpu_render else None,
        )
        return {"summary_xlsx": xlsx_path}

    # ----- mode == "sweep" --------------------------------------------------
    if cfg.mode == "sweep":
        from .sweep import run_sweep
        xlsx_path = run_sweep(
            cfg=cfg,
            aruco_dict_cv=aruco_dict_cv,
            torch_device=device_str,
            use_torch_render=use_gpu_render,
            sample_name=sample_name,
            render_fn=render_orthographic_topview_gpu if use_gpu_render else None,
        )
        return {"summary_xlsx": xlsx_path}

    # ----- mode == "single" / "roi" -----------------------------------------
    return _run_single_or_roi(
        cfg=cfg,
        sample_name=sample_name,
        aruco_dict_cv=aruco_dict_cv,
        device_str=device_str,
        use_gpu_render=use_gpu_render,
    )


def _run_single_or_roi(
    cfg: RegistrationConfig,
    sample_name: str,
    aruco_dict_cv: int,
    device_str: Optional[str],
    use_gpu_render: bool,
) -> dict[str, Any]:
    """Single / ROI path — mirrors the second half of legacy main.py."""
    from .pipeline import load_mesh, save_render, save_turbo_render
    from .pipeline_roi import run_full_pipeline_roi

    res_pc = cfg.render.resolution_mm_per_px
    res_reg = cfg.render.resolution_reg_mm_per_px
    dual = abs(res_reg - res_pc) > 1e-6

    output_dir = str(cfg.paths.output_dir)
    save_pointcloud = cfg.export.save_pointcloud
    save_images = cfg.export.save_images

    precomputed_render = None
    precomputed_render_reg = None

    # Pre-render on GPU once, so single+roi pipelines reuse them.
    if use_gpu_render and render_orthographic_topview_gpu is not None:
        mesh = load_mesh(str(cfg.paths.mesh), scale_m_to_mm=True)

        print(f"\n[Render PC] {res_pc} mm/px on {device_str}...")
        t0 = time.time()
        precomputed_render = render_orthographic_topview_gpu(
            mesh,
            resolution_mm_per_px=res_pc,
            margin_mm=cfg.render.margin_mm,
            device=device_str,
        )
        print(f"[Render PC] Completed in {time.time() - t0:.1f}s")
        if save_images:
            r_rgb, d_map, xyz_map, _, _ = precomputed_render
            save_render(r_rgb, d_map, output_dir=output_dir,
                        prefix=f"{sample_name}_render_pc")
            save_turbo_render(
                xyz_map,
                str(Path(output_dir) / f"{sample_name}_render_pc_turbo.png"),
            )

        if dual:
            print(f"\n[Render REG] {res_reg} mm/px on {device_str}...")
            t0 = time.time()
            precomputed_render_reg = render_orthographic_topview_gpu(
                mesh,
                resolution_mm_per_px=res_reg,
                margin_mm=cfg.render.margin_mm,
                device=device_str,
            )
            print(f"[Render REG] Completed in {time.time() - t0:.1f}s")
            if save_images:
                r_rgb, d_map, xyz_map, _, _ = precomputed_render_reg
                save_render(r_rgb, d_map, output_dir=output_dir,
                            prefix=f"{sample_name}_render_reg")
                save_turbo_render(
                    xyz_map,
                    str(Path(output_dir) / f"{sample_name}_render_reg_turbo.png"),
                )
        else:
            precomputed_render_reg = precomputed_render

    # ROI-only params
    liveview_png_path = str(cfg.paths.liveview_png) if cfg.mode == "roi" else None
    roi_align_cfg = cfg.roi.model_dump() if cfg.mode == "roi" else None

    t0 = time.time()
    result = run_full_pipeline_roi(
        hsi_hdr_path             = str(cfg.paths.hsi_hdr),
        mesh_path                = str(cfg.paths.mesh),
        aruco_json_path          = str(cfg.paths.aruco_json),
        output_dir               = output_dir,
        liveview_png_path        = liveview_png_path,
        roi_align_cfg            = roi_align_cfg,
        hsi_extraction_method    = cfg.detection.hsi_extraction_method,
        aruco_dict_type          = aruco_dict_cv,
        suspicious_pixels_hsi    = None,
        render_resolution_mm     = res_pc,
        render_resolution_reg_mm = res_reg,
        render_margin_mm         = cfg.render.margin_mm,
        marker_side_mm           = cfg.detection.marker_side_mm,
        use_subpix               = cfg.detection.use_subpix,
        subpix_winsize           = cfg.detection.subpix_winsize,
        border_px                = cfg.pointcloud.border_px,
        reflectance_norm         = cfg.pointcloud.reflectance_norm,
        pc_chunk_size            = cfg.pointcloud.pc_chunk_size,
        export_ply_file          = cfg.export.ply  if save_pointcloud else False,
        export_npy_file          = cfg.export.npy  if save_pointcloud else False,
        export_csv_file          = cfg.export.csv  if save_pointcloud else False,
        csv_max_points           = cfg.export.csv_max_points,
        save_pointcloud          = save_pointcloud,
        save_images              = save_images,
        sample_name              = sample_name,
        precomputed_render       = precomputed_render,
        precomputed_render_reg   = precomputed_render_reg,
    )
    elapsed = time.time() - t0

    print("\n" + "=" * 68)
    print("REGISTRATION COMPLETED")
    print("=" * 68)
    print(f"  Total time     : {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    n_pts = result.get("n_valid", 0)
    n_bands = result.get("bands", 0)
    if save_pointcloud:
        print(f"  Cloud points   : {n_pts:,}")
        print(f"  Spectral bands : {n_bands}")
    if result.get("wavelengths"):
        wl = result["wavelengths"]
        print(f"  Wavelengths    : {wl[0]:.1f} — {wl[-1]:.1f} nm")
    if result.get("is_roi_mode"):
        info = result["roi_align_info"]
        print(f"  ROI→PNG match  : {info['n_inliers']}/{info['n_good_matches']} "
              f"inliers, reproj={info['reproj_error_mean_px']:.3f} px")
    print(f"  Output         : {output_dir}")
    print("=" * 68)

    return result


__all__ = ["run_registration"]
