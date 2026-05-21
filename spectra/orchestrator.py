"""End-to-end orchestrator: ``spectra full`` (vision → registration).

Responsibilities
================
1. **Path resolution.** For a given ``UnifiedConfig`` and (optionally) a
   sample name, fill in any per-sample paths that the user left ``None`` in
   the YAML, following the layout convention documented in
   ``configs/default.yaml``.

2. **Stage skipping.** When ``spectra full`` is invoked and the vision
   outputs already exist (``RESULTS/<sample>/vision/surface_mesh.ply`` AND
   ``RESULTS/<sample>/vision/aruco_markers_3d.json``), the vision stage is
   skipped automatically.

3. **Cross-stage I/O.** After vision runs, the orchestrator injects the
   freshly-produced ``surface_mesh.ply`` and ``aruco_markers_3d.json`` into
   the registration config, so registration always sees the right inputs.

The standalone CLI sub-commands (``spectra vision``, ``spectra registration``)
also call :func:`resolve_unified_paths` so the layout convention works the
same way for them.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

from .config import (
    RegistrationConfig,
    UnifiedConfig,
    VisionConfig,
)


# =============================================================================
# Layout convention
# =============================================================================
# These small helpers exist as a SINGLE source of truth: changing any of them
# changes path conventions globally.

def vision_input_dir(data_root: Path, sample: str) -> Path:
    return Path(data_root) / "input_vision" / sample


def registration_input_dir(data_root: Path, sample: str) -> Path:
    return Path(data_root) / "input_registration" / sample


def vision_output_dir(results_root: Path, sample: str) -> Path:
    return Path(results_root) / sample / "vision"


def registration_output_dir(results_root: Path, sample: str) -> Path:
    return Path(results_root) / sample / "registration"


def vision_mesh_path(results_root: Path, sample: str) -> Path:
    return vision_output_dir(results_root, sample) / "surface_mesh.ply"


def vision_aruco_path(results_root: Path, sample: str) -> Path:
    return vision_output_dir(results_root, sample) / "aruco_markers_3d.json"


def default_hsi_hdr_path(data_root: Path, sample: str) -> Path:
    return registration_input_dir(data_root, sample) / f"{sample}_raw.hdr"


def default_liveview_png_path(data_root: Path, sample: str) -> Path:
    return registration_input_dir(data_root, sample) / f"{sample}_raw.png"


# =============================================================================
# Path resolution
# =============================================================================

def _is_unset(value: Any) -> bool:
    """Treat None and empty strings as unset; everything else is user-provided."""
    if value is None:
        return True
    if isinstance(value, (str, Path)) and str(value).strip() == "":
        return True
    return False


def resolve_unified_paths(cfg: UnifiedConfig) -> UnifiedConfig:
    """Fill in any unset path under ``vision.input`` / ``registration.paths``
    using the layout convention. Returns a new ``UnifiedConfig`` (the input is
    not mutated)."""
    sample = cfg.sample_name
    overrides: dict[str, Any] = {}

    # ------- vision.input -------------------------------------------------
    v_in = cfg.vision.input
    if _is_unset(v_in.rgb_dir):
        overrides["vision.input.rgb_dir"] = str(vision_input_dir(cfg.data_root, sample) / "rgb")
    # pose_dir / camera_params_dir auto-fill only if the folders exist.
    if _is_unset(v_in.pose_dir):
        candidate = vision_input_dir(cfg.data_root, sample) / "poses"
        if candidate.is_dir():
            overrides["vision.input.pose_dir"] = str(candidate)
    if _is_unset(v_in.camera_params_dir):
        candidate = vision_input_dir(cfg.data_root, sample) / "camera_parameters"
        if candidate.is_dir():
            overrides["vision.input.camera_params_dir"] = str(candidate)

    # ------- vision.output ------------------------------------------------
    if _is_unset(cfg.vision.output.root):
        overrides["vision.output.root"] = str(vision_output_dir(cfg.results_root, sample).parent)
    if _is_unset(cfg.vision.output.run_name):
        overrides["vision.output.run_name"] = "vision"

    # ------- registration.paths ------------------------------------------
    r_paths = cfg.registration.paths
    if _is_unset(r_paths.hsi_hdr):
        overrides["registration.paths.hsi_hdr"] = str(default_hsi_hdr_path(cfg.data_root, sample))
    if _is_unset(r_paths.mesh):
        overrides["registration.paths.mesh"] = str(vision_mesh_path(cfg.results_root, sample))
    if _is_unset(r_paths.aruco_json):
        overrides["registration.paths.aruco_json"] = str(vision_aruco_path(cfg.results_root, sample))
    if _is_unset(r_paths.output_dir):
        overrides["registration.paths.output_dir"] = str(registration_output_dir(cfg.results_root, sample))

    # liveview_png: only auto-derive when the file exists (don't force a
    # path that might not exist for a non-ROI run, so validation stays clean).
    needs_liveview = (
        cfg.registration.mode == "roi"
        or (cfg.registration.mode == "sweep" and cfg.registration.sweep.roi_mode)
    )
    if needs_liveview and _is_unset(r_paths.liveview_png):
        overrides["registration.paths.liveview_png"] = str(
            default_liveview_png_path(cfg.data_root, sample)
        )
    # In non-ROI single/sweep, leave None untouched.

    return cfg.with_overrides(overrides) if overrides else cfg


# =============================================================================
# Skip-if-exists helper
# =============================================================================

def vision_outputs_present(results_root: Path, sample: str) -> bool:
    """True iff both surface_mesh.ply AND aruco_markers_3d.json exist."""
    return (
        vision_mesh_path(results_root, sample).is_file()
        and vision_aruco_path(results_root, sample).is_file()
    )


# =============================================================================
# Stage runners (thin wrappers — used by CLI and by run_full)
# =============================================================================

def run_vision_stage(
    cfg: UnifiedConfig,
    *,
    sample_name: Optional[str] = None,
) -> dict[str, Any]:
    """Run only the vision stage. Returns ``{"run_dir": Path, "skipped": False}``."""
    from .vision.pipeline import run_reconstruction

    cfg = resolve_unified_paths(cfg)
    sample = sample_name or cfg.sample_name
    print(f"\n[orchestrator] Vision stage for sample={sample!r}")
    t0 = time.time()
    result = run_reconstruction(cfg.vision)
    dt = time.time() - t0
    print(f"[orchestrator] Vision finished in {dt:.1f}s — run_dir={result.run_dir}")
    return {"run_dir": Path(result.run_dir), "skipped": False, "elapsed_s": dt}


def run_registration_stage(
    cfg: UnifiedConfig,
    *,
    sample_name: Optional[str] = None,
    force_cpu: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run only the registration stage (mode is read from cfg.registration.mode)."""
    from .registration.runner import run_registration

    cfg = resolve_unified_paths(cfg)
    sample = sample_name or cfg.sample_name
    print(f"\n[orchestrator] Registration stage for sample={sample!r} "
          f"(mode={cfg.registration.mode!r})")
    t0 = time.time()
    result = run_registration(
        cfg.registration,
        sample_name=sample,
        data_root=cfg.data_root,
        results_root=cfg.results_root,
        force_cpu=force_cpu,
        dry_run=dry_run,
    )
    dt = time.time() - t0
    print(f"[orchestrator] Registration finished in {dt:.1f}s")
    result["elapsed_s"] = dt
    return result


# =============================================================================
# Full pipeline
# =============================================================================

def run_full(
    cfg: UnifiedConfig,
    *,
    sample_name: Optional[str] = None,
    force_cpu: bool = False,
    force_vision: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run vision (if needed) and then registration on a single sample.

    Parameters
    ----------
    cfg :
        Unified config. Path resolution is performed internally — anything
        ``None`` is filled in via the layout convention.
    sample_name :
        Overrides ``cfg.sample_name`` for this call.
    force_cpu :
        Forwarded to registration.
    force_vision :
        If True, runs vision even when its outputs already exist.
    dry_run :
        If True, validates configs and prints what would happen, no execution.

    Returns
    -------
    dict
        ``{"vision": ..., "registration": ..., "sample": ..., "skipped_vision": bool}``
    """
    cfg = resolve_unified_paths(cfg)
    sample = sample_name or cfg.sample_name

    print("\n" + "=" * 68)
    print(f"  SPECTRA FULL — sample={sample!r}")
    print(f"  data_root    : {cfg.data_root}")
    print(f"  results_root : {cfg.results_root}")
    print(f"  stages       : {cfg.stages}")
    print(f"  reg mode     : {cfg.registration.mode}")
    print("=" * 68)

    do_vision = "vision" in cfg.stages
    do_registration = "registration" in cfg.stages

    out: dict[str, Any] = {"sample": sample, "skipped_vision": False}

    if do_vision:
        if not force_vision and vision_outputs_present(cfg.results_root, sample):
            print(f"\n[orchestrator] Vision outputs already exist at "
                  f"{vision_output_dir(cfg.results_root, sample)} — skipping vision stage.")
            print(f"  (use --force-vision to re-run vision anyway)")
            out["skipped_vision"] = True
            out["vision"] = {
                "skipped": True,
                "run_dir": vision_output_dir(cfg.results_root, sample),
            }
        else:
            if dry_run:
                print("[orchestrator][dry-run] Would run vision stage.")
                out["vision"] = {"skipped": False, "dry_run": True}
            else:
                out["vision"] = run_vision_stage(cfg, sample_name=sample)
    else:
        print("[orchestrator] Vision stage disabled (not in cfg.stages).")

    if do_registration:
        # Re-resolve in case vision wrote outputs into a custom run_name and
        # registration.paths.mesh / .aruco_json were derived assuming the
        # canonical 'vision' run_name. This is a no-op when nothing changed.
        cfg = resolve_unified_paths(cfg)
        if dry_run:
            print("[orchestrator][dry-run] Would run registration stage.")
            out["registration"] = {"dry_run": True}
        else:
            out["registration"] = run_registration_stage(
                cfg, sample_name=sample, force_cpu=force_cpu, dry_run=False,
            )
    else:
        print("[orchestrator] Registration stage disabled (not in cfg.stages).")

    print("\n" + "=" * 68)
    print("SPECTRA FULL COMPLETED")
    print("=" * 68)
    print(f"  Sample        : {sample}")
    print(f"  Vision skipped: {out['skipped_vision']}")
    if "vision" in out and "run_dir" in out["vision"]:
        print(f"  Vision output : {out['vision']['run_dir']}")
    if "registration" in out:
        print(f"  Reg. output   : {cfg.registration.paths.output_dir}")
    print("=" * 68)

    return out


__all__ = [
    "default_hsi_hdr_path",
    "default_liveview_png_path",
    "registration_input_dir",
    "registration_output_dir",
    "resolve_unified_paths",
    "run_full",
    "run_registration_stage",
    "run_vision_stage",
    "vision_aruco_path",
    "vision_input_dir",
    "vision_mesh_path",
    "vision_output_dir",
    "vision_outputs_present",
]
