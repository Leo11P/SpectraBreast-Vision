#!/usr/bin/env python3
"""Grid-search over SpectraBreast vision hyperparameters (mesh + bundle adjustment).

Runs `spectra full` once per hyperparameter combination, forcing a fresh vision
stage each time (so the new vggt/aruco params actually take effect), then parses
each run's stdout for the metrics we care about:

  - Bundle-adjustment reprojection RMSE (px), initial + final
  - ArUco geometric QC: plane RMS (mm), reproj max/p95, n observations
  - Rejected views (if any)
  - Registration 3D error on markers (mm), best pair from the sweep
  - Registration 2D reprojection error (px)
  - Peak GPU memory (MiB), sampled via nvidia-smi during the run
  - Wall-clock time per combo (vision + registration)

Design choices:
  * Two SEPARATE grids, not a full cartesian product:
      Grid A varies mesh params (conf_threshold x voxel_size), BA fixed at baseline.
      Grid B varies BA params, mesh fixed at baseline.
    This isolates the effect of each group instead of confounding them, and keeps
    the run count (~27) inside one SLURM allocation. Change BASELINE / the grid
    lists below to taste.
  * A failing combo (OOM, etc.) is recorded as FAILED and the grid continues.
  * Nothing is computed from priors — every number in the report comes from a real
    run's stdout on this machine.

Outputs (under --out-dir):
  grid_report.xlsx   # sheets: results, timings, memory, failures
  grid_report.md     # human-readable summary, sorted by the two key metrics
  raw_logs/          # full stdout+stderr per combo, for auditing
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Hyperparameter space
# ---------------------------------------------------------------------------
# Baseline: the values used when a given axis is NOT being swept.
BASELINE: Dict[str, Any] = {
    "vision.vggt.conf_threshold": 3.0,
    "vision.vggt.voxel_size": 0.0005,
    "vision.aruco.ba_huber_delta_px": 2.0,
    "vision.aruco.max_view_alignment_reproj_rmse_px": 10.0,
    "vision.aruco.min_views_per_marker": 3,
}

# Grid A — mesh quality (BA held at baseline).
GRID_A = {
    "vision.vggt.conf_threshold": [2.0, 3.0, 5.0],
    "vision.vggt.voxel_size": [0.0005, 0.001, 0.0015],
}

# Grid B — bundle adjustment / ArUco (mesh held at baseline).
GRID_B = {
    "vision.aruco.ba_huber_delta_px": [1.0, 2.0, 4.0],
    "vision.aruco.max_view_alignment_reproj_rmse_px": [6.0, 8.0, 10.0],
    "vision.aruco.min_views_per_marker": [2, 3],
}

#GRID_A = {
#    "vision.vggt.conf_threshold": [3.0],
#    "vision.vggt.voxel_size": [0.0005],
#}
#GRID_B = {
#    "vision.aruco.ba_huber_delta_px": [2.0],
#    "vision.aruco.max_view_alignment_reproj_rmse_px": [6.0],
#    "vision.aruco.min_views_per_marker": [3],
#}


def _combos_from_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    out = []
    for values in itertools.product(*(grid[k] for k in keys)):
        combo = dict(BASELINE)
        combo.update(dict(zip(keys, values)))
        out.append(combo)
    return out


def build_grid() -> List[Dict[str, Any]]:
    """Grid A then Grid B, de-duplicated (the shared baseline point can appear in both)."""
    combos: List[Dict[str, Any]] = []
    seen = set()
    for combo in _combos_from_grid(GRID_A) + _combos_from_grid(GRID_B):
        key = tuple(sorted(combo.items()))
        if key not in seen:
            seen.add(key)
            combos.append(combo)
    return combos


# ---------------------------------------------------------------------------
# GPU memory sampler (nvidia-smi polled in a background thread)
# ---------------------------------------------------------------------------
class GpuMemorySampler:
    def __init__(self, interval_s: float = 1.0):
        self.interval_s = interval_s
        self._peak_mib = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._available = self._probe()

    @staticmethod
    def _probe() -> bool:
        try:
            subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, check=True, timeout=5,
            )
            return True
        except Exception:
            return False

    def _loop(self):
        while not self._stop.is_set():
            try:
                res = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                # Sum across visible GPUs (usually 1 here); take the max over time.
                used = sum(int(x) for x in res.stdout.strip().splitlines() if x.strip())
                self._peak_mib = max(self._peak_mib, used)
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._peak_mib = 0
        if self._available:
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=5)

    @property
    def peak_mib(self) -> Optional[int]:
        return self._peak_mib if self._available else None


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------
@dataclass
class RunMetrics:
    ba_rmse_initial_px: Optional[float] = None
    ba_rmse_final_px: Optional[float] = None
    ba_num_markers: Optional[int] = None
    ba_num_obs: Optional[int] = None
    ba_num_iters: Optional[int] = None
    qc_plane_rms_mm: Optional[float] = None
    qc_reproj_max_px: Optional[float] = None
    qc_reproj_p95_px: Optional[float] = None
    rejected_views: Optional[int] = None
    reg_best_res_reg: Optional[float] = None
    reg_best_res_pc: Optional[float] = None
    reg_3d_error_mm: Optional[float] = None      # best-pair 3D marker error
    reg_2d_error_px: Optional[float] = None      # best-pair 2D reprojection mean
    fused_points: Optional[int] = None
    mesh_vertices: Optional[int] = None


_num = r"([-+]?\d+(?:\.\d+)?)"


def parse_metrics(stdout: str) -> RunMetrics:
    m = RunMetrics()

    # Bundle adjustment: "4 markers, 392 obs, reproj RMSE 42.693 -> 5.299 px (1000 iters...)"
    ba = re.search(
        rf"Bundle adjustment:\s*(\d+)\s*markers,\s*(\d+)\s*obs,\s*"
        rf"reproj RMSE\s*{_num}\s*->\s*{_num}\s*px\s*\((\d+)\s*iters",
        stdout,
    )
    if ba:
        m.ba_num_markers = int(ba.group(1))
        m.ba_num_obs = int(ba.group(2))
        m.ba_rmse_initial_px = float(ba.group(3))
        m.ba_rmse_final_px = float(ba.group(4))
        m.ba_num_iters = int(ba.group(5))

    # QC: "best-fit plane RMS 0.25 mm; ... RMSE 5.299 px (max 19.169, p95 9.164, n=392)"
    qc = re.search(
        rf"best-fit plane RMS\s*{_num}\s*mm;.*?RMSE\s*{_num}\s*px\s*"
        rf"\(max\s*{_num},\s*p95\s*{_num},\s*n=(\d+)\)",
        stdout, re.DOTALL,
    )
    if qc:
        m.qc_plane_rms_mm = float(qc.group(1))
        m.qc_reproj_max_px = float(qc.group(3))
        m.qc_reproj_p95_px = float(qc.group(4))

    # Rejected views: "Rejecting N view(s) with per-view marker RMSE > ..."
    rej = re.search(r"Rejecting\s+(\d+)\s+view", stdout)
    m.rejected_views = int(rej.group(1)) if rej else 0

    # Fused points: "Backend produced 2,000,000 fused points"
    fp = re.search(r"Backend produced\s*([\d,]+)\s*fused points", stdout)
    if fp:
        m.fused_points = int(fp.group(1).replace(",", ""))

    # Mesh: "Surface: 1,040,932 vertices"
    sv = re.search(r"Surface:\s*([\d,]+)\s*vertices", stdout)
    if sv:
        m.mesh_vertices = int(sv.group(1).replace(",", ""))

    # Best sweep pair: "★ Best pair (row N): res_reg=2.0  res_pc=0.5  -> 3D error min = 1.9655 mm"
    best = re.search(
        rf"Best pair.*?res_reg={_num}\s+res_pc={_num}\s*.*?3D error min\s*=\s*{_num}\s*mm",
        stdout, re.DOTALL,
    )
    if best:
        m.reg_best_res_reg = float(best.group(1))
        m.reg_best_res_pc = float(best.group(2))
        m.reg_3d_error_mm = float(best.group(3))

    # 2D error for the best pair is harder to tie exactly; take the minimum 2D mean
    # reported across the sweep as a proxy (best-registered pair).
    twod = re.findall(rf"2D error mean\s*:\s*{_num}\s*px", stdout)
    if twod:
        m.reg_2d_error_px = min(float(x) for x in twod)

    return m


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
@dataclass
class ComboResult:
    idx: int
    params: Dict[str, Any]
    status: str = "OK"
    error: str = ""
    wall_s: float = 0.0
    peak_gpu_mib: Optional[int] = None
    metrics: RunMetrics = field(default_factory=RunMetrics)


def run_combo(
    idx: int,
    params: Dict[str, Any],
    config: str,
    sample: str,
    raw_log_dir: Path,
    extra_set: List[str],
) -> ComboResult:
    cmd = ["uv", "run", "--no-sync", "spectra", "full",
           "-c", config, "--sample", sample, "--force-vision"]
    for key, val in params.items():
        cmd += ["-s", f"{key}={_fmt(val)}"]
    for s in extra_set:
        cmd += ["-s", s]

    result = ComboResult(idx=idx, params=dict(params))
    t0 = time.perf_counter()
    with GpuMemorySampler(interval_s=1.0) as sampler:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60 * 60)
            stdout, stderr = proc.stdout, proc.stderr
            if proc.returncode != 0:
                result.status = "FAILED"
                result.error = _tail(stderr or stdout, 500)
        except subprocess.TimeoutExpired as e:
            stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
            result.status = "TIMEOUT"
            result.error = "combo exceeded 60 min"
        except Exception as e:  # noqa: BLE001
            stdout, stderr = "", str(e)
            result.status = "FAILED"
            result.error = str(e)
    result.wall_s = time.perf_counter() - t0
    result.peak_gpu_mib = sampler.peak_mib

    (raw_log_dir / f"combo_{idx:02d}.log").write_text(
        f"CMD: {' '.join(cmd)}\n\n=== STDOUT ===\n{stdout}\n\n=== STDERR ===\n{stderr}\n"
    )
    if result.status == "OK":
        result.metrics = parse_metrics(stdout)
        # Sanity: if we couldn't parse the BA line, the run probably didn't get there.
        if result.metrics.ba_rmse_final_px is None and result.metrics.reg_3d_error_mm is None:
            result.status = "OK_NO_METRICS"
            result.error = "run completed but no metrics parsed (check raw log)"
    return result


def _fmt(v: Any) -> str:
    # JSON-compatible so the CLI's json.loads parses ints/floats/bools correctly.
    if isinstance(v, bool):
        return "true" if v else "false"
    return json.dumps(v)


def _tail(s: str, n_lines: int) -> str:
    return "\n".join(s.splitlines()[-n_lines:])


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def write_reports(results: List[ComboResult], out_dir: Path) -> None:
    rows = []
    for r in results:
        row = {"combo": r.idx, "status": r.status}
        row.update({k.split(".")[-1]: v for k, v in r.params.items()})
        md = r.metrics
        row.update({
            "ba_rmse_final_px": md.ba_rmse_final_px,
            "ba_rmse_initial_px": md.ba_rmse_initial_px,
            "ba_num_obs": md.ba_num_obs,
            "reg_3d_error_mm": md.reg_3d_error_mm,
            "reg_2d_error_px": md.reg_2d_error_px,
            "reg_best_res_reg": md.reg_best_res_reg,
            "qc_plane_rms_mm": md.qc_plane_rms_mm,
            "qc_reproj_p95_px": md.qc_reproj_p95_px,
            "qc_reproj_max_px": md.qc_reproj_max_px,
            "rejected_views": md.rejected_views,
            "fused_points": md.fused_points,
            "mesh_vertices": md.mesh_vertices,
            "wall_s": round(r.wall_s, 1),
            "peak_gpu_mib": r.peak_gpu_mib,
        })
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by the two key metrics affiancate: 3D error (mm) then BA RMSE (px).
    df_sorted = df.sort_values(
        by=["reg_3d_error_mm", "ba_rmse_final_px"], na_position="last"
    ).reset_index(drop=True)

    timings = df[["combo", "status", "wall_s", "peak_gpu_mib"]].copy()
    memory = df[["combo", "status", "peak_gpu_mib", "fused_points", "mesh_vertices"]].copy()
    failures = df[df["status"] != "OK"][["combo", "status"]].copy()
    failures["error"] = [
        next((r.error for r in results if r.idx == c), "") for c in failures["combo"]
    ]

    xlsx = out_dir / "grid_report.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        df_sorted.to_excel(xw, sheet_name="results", index=False)
        timings.to_excel(xw, sheet_name="timings", index=False)
        memory.to_excel(xw, sheet_name="memory", index=False)
        failures.to_excel(xw, sheet_name="failures", index=False)

    _write_markdown(df_sorted, results, out_dir / "grid_report.md")
    print(f"\n[report] Excel  -> {xlsx}")
    print(f"[report] Markdown -> {out_dir / 'grid_report.md'}")


def _write_markdown(df_sorted: pd.DataFrame, results: List[ComboResult], path: Path) -> None:
    ok = [r for r in results if r.status in ("OK", "OK_NO_METRICS")]
    total_time = sum(r.wall_s for r in results)
    peak = max((r.peak_gpu_mib or 0) for r in results) if results else 0

    lines = [
        "# SpectraBreast Vision — Grid Search Report",
        "",
        f"- Combos run: **{len(results)}**  (OK: {len(ok)}, "
        f"failed/other: {len(results) - len(ok)})",
        f"- Total wall-clock: **{total_time/60:.1f} min**",
        f"- Peak GPU memory across grid: **{peak} MiB**",
        "",
        "Sorted by 3D marker error (mm), then BA reprojection RMSE (px).",
        "",
        "| # | conf_thr | voxel | huber | rej_thr | min_views "
        "| 3D err (mm) | BA RMSE (px) | 2D err (px) | plane RMS (mm) "
        "| rej views | GPU (MiB) | t (s) | status |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df_sorted.iterrows():
        lines.append(
            "| {combo} | {conf_threshold} | {voxel_size} | {ba_huber_delta_px} "
            "| {max_view_alignment_reproj_rmse_px} | {min_views_per_marker} "
            "| {reg_3d_error_mm} | {ba_rmse_final_px} | {reg_2d_error_px} "
            "| {qc_plane_rms_mm} | {rejected_views} | {peak_gpu_mib} "
            "| {wall_s} | {status} |".format(**{k: r.get(k, "") for k in [
                "combo", "conf_threshold", "voxel_size", "ba_huber_delta_px",
                "max_view_alignment_reproj_rmse_px", "min_views_per_marker",
                "reg_3d_error_mm", "ba_rmse_final_px", "reg_2d_error_px",
                "qc_plane_rms_mm", "rejected_views", "peak_gpu_mib", "wall_s", "status",
            ]})
        )
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-c", "--config", default="configs/default.yaml")
    ap.add_argument("--sample", default="SAMPLE2")
    ap.add_argument("--out-dir", default="RESULTS/grid_search")
    ap.add_argument("--extra-set", action="append", default=["vision.rerun.enabled=false"],
                    help="Extra -s overrides applied to every combo (repeatable).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the combos and commands without running.")
    args = ap.parse_args()

    combos = build_grid()
    out_dir = Path(args.out_dir)
    raw_log_dir = out_dir / "raw_logs"
    raw_log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[grid] {len(combos)} combos; sample={args.sample}; config={args.config}")
    if args.dry_run:
        for i, c in enumerate(combos):
            sets = " ".join(f"-s {k}={_fmt(v)}" for k, v in c.items())
            print(f"  [{i:02d}] {sets}")
        return 0

    results: List[ComboResult] = []
    for i, combo in enumerate(combos):
        print(f"\n{'='*70}\n[grid] Combo {i+1}/{len(combos)}\n"
              f"       {json.dumps(combo)}\n{'='*70}")
        res = run_combo(i, combo, args.config, args.sample, raw_log_dir, args.extra_set)
        status_str = res.status
        m = res.metrics
        print(f"[grid] -> {status_str}  "
              f"3D={m.reg_3d_error_mm}mm  BA={m.ba_rmse_final_px}px  "
              f"t={res.wall_s:.0f}s  gpu={res.peak_gpu_mib}MiB")
        results.append(res)
        # Write incrementally so a crash mid-grid still leaves a partial report.
        write_reports(results, out_dir)

    print(f"\n[grid] DONE. {sum(r.status=='OK' for r in results)}/{len(results)} OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
