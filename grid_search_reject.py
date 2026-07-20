#!/usr/bin/env python3
"""Targeted sweep: push the view-rejection threshold below 6 px.

Prior grid found the best 3D marker error (2.61 mm) at:
    ba_huber_delta_px = 1.0
    max_view_alignment_reproj_rmse_px = 6.0  (rejecting ~7-8 of 26 views)
where LOWER reject threshold -> more views dropped -> better 3D error.

This run holds everything at that winning config and varies ONLY the reject
threshold, stepping below 6 px, to find where the improvement stops — i.e.
whether 2.6 mm is a floor because too few views remain for stable marker
triangulation.

Two failure modes we must be able to tell apart in the report:
  (A) Statistical floor: error rises again because few views remain.
  (B) Safety cutoff fires: the pipeline refuses the rejection entirely
      ("View rejection skipped: would keep X views (minimum Y)") and falls
      back to 0 rejected views — error jumps back to the no-reject value.
      This is a mechanism artifact, NOT a real result.
So we parse BOTH the "Rejecting N views" and the "rejection skipped" messages,
and record views_kept + reject_skipped explicitly.

Outputs (under --out-dir):
  reject_sweep_report.xlsx  # sheets: results, timings, memory, failures
  reject_sweep_report.md
  raw_logs/                 # full stdout+stderr per combo
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
import threading
import time
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Hyperparameter space — winning config fixed, only reject threshold varies
# ---------------------------------------------------------------------------
BASELINE: Dict[str, Any] = {
    "vision.vggt.conf_threshold": 3.0,
    "vision.vggt.voxel_size": 0.0015,        # high on purpose: irrelevant to 3D error, saves points/time
    "vision.aruco.ba_huber_delta_px": 1.0,   # clear winner from the previous grid
    "vision.aruco.max_view_alignment_reproj_rmse_px": 6.0,
    "vision.aruco.min_views_per_marker": 3,
}

# The single axis under test. 6.0 included as the known reference (2.61 mm).
GRID_A = {
    "vision.aruco.max_view_alignment_reproj_rmse_px": [6.0, 5.0, 4.5, 4.0, 3.5, 3.0],
}
GRID_B: Dict[str, List[Any]] = {}   # intentionally empty this run


def _combos_from_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    if not grid:                       # empty grid -> no combos (avoid a spurious baseline dup)
        return []
    keys = list(grid.keys())
    out = []
    for values in itertools.product(*(grid[k] for k in keys)):
        combo = dict(BASELINE)
        combo.update(dict(zip(keys, values)))
        out.append(combo)
    return out


def build_grid() -> List[Dict[str, Any]]:
    combos: List[Dict[str, Any]] = []
    seen = set()
    for combo in _combos_from_grid(GRID_A) + _combos_from_grid(GRID_B):
        key = tuple(sorted(combo.items()))
        if key not in seen:
            seen.add(key)
            combos.append(combo)
    return combos


# ---------------------------------------------------------------------------
# GPU memory sampler
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
    ba_num_obs: Optional[int] = None
    qc_plane_rms_mm: Optional[float] = None
    qc_reproj_p95_px: Optional[float] = None
    qc_reproj_max_px: Optional[float] = None
    rejected_views: Optional[int] = None
    reject_skipped: Optional[bool] = None       # True if the safety cutoff fired
    views_kept: Optional[int] = None            # how many views survived
    views_total: Optional[int] = None
    reg_best_res_reg: Optional[float] = None
    reg_3d_error_mm: Optional[float] = None
    reg_2d_error_px: Optional[float] = None
    fused_points: Optional[int] = None
    mesh_vertices: Optional[int] = None


_num = r"([-+]?\d+(?:\.\d+)?)"


def parse_metrics(stdout: str) -> RunMetrics:
    m = RunMetrics()

    ba = re.search(
        rf"Bundle adjustment:\s*(\d+)\s*markers,\s*(\d+)\s*obs,\s*"
        rf"reproj RMSE\s*{_num}\s*->\s*{_num}\s*px\s*\((\d+)\s*iters",
        stdout,
    )
    if ba:
        m.ba_num_obs = int(ba.group(2))
        m.ba_rmse_initial_px = float(ba.group(3))
        m.ba_rmse_final_px = float(ba.group(4))

    qc = re.search(
        rf"best-fit plane RMS\s*{_num}\s*mm;.*?RMSE\s*{_num}\s*px\s*"
        rf"\(max\s*{_num},\s*p95\s*{_num},\s*n=(\d+)\)",
        stdout, re.DOTALL,
    )
    if qc:
        m.qc_plane_rms_mm = float(qc.group(1))
        m.qc_reproj_max_px = float(qc.group(3))
        m.qc_reproj_p95_px = float(qc.group(4))

    # Total views: "Loaded 26 images"
    tot = re.search(r"Loaded\s+(\d+)\s+images", stdout)
    if tot:
        m.views_total = int(tot.group(1))

    # --- View-rejection outcome: three mutually exclusive cases -------------
    # 1) Rejection happened:  "Rejecting N view(s) ..."
    rej = re.search(r"Rejecting\s+(\d+)\s+view", stdout)
    # 2) Safety cutoff fired: "View rejection skipped: would keep X views (minimum Y)"
    skip = re.search(
        r"View rejection skipped:\s*would keep\s+(\d+)\s+views\s*\(minimum\s+(\d+)\)",
        stdout,
    )
    if skip:
        m.reject_skipped = True
        m.rejected_views = 0
        m.views_kept = int(skip.group(1))     # what it WOULD have kept (cutoff blocked it)
    elif rej:
        m.reject_skipped = False
        m.rejected_views = int(rej.group(1))
        if m.views_total is not None:
            m.views_kept = m.views_total - m.rejected_views
    else:
        # No message at all -> nothing exceeded the threshold, nothing dropped.
        m.reject_skipped = False
        m.rejected_views = 0
        m.views_kept = m.views_total

    fp = re.search(r"Backend produced\s*([\d,]+)\s*fused points", stdout)
    if fp:
        m.fused_points = int(fp.group(1).replace(",", ""))

    sv = re.search(r"Surface:\s*([\d,]+)\s*vertices", stdout)
    if sv:
        m.mesh_vertices = int(sv.group(1).replace(",", ""))

    best = re.search(
        rf"Best pair.*?res_reg={_num}\s+res_pc={_num}\s*.*?3D error min\s*=\s*{_num}\s*mm",
        stdout, re.DOTALL,
    )
    if best:
        m.reg_best_res_reg = float(best.group(1))
        m.reg_3d_error_mm = float(best.group(3))

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


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return json.dumps(v)


def _tail(s: str, n_lines: int) -> str:
    return "\n".join(s.splitlines()[-n_lines:])


def run_combo(idx, params, config, sample, raw_log_dir, extra_set) -> ComboResult:
    cmd = ["uv", "run", "--no-sync", "spectra", "full",
           "-c", config, "--sample", sample, "--force-vision"]
    for key, val in params.items():
        cmd += ["-s", f"{key}={_fmt(val)}"]
    for s in extra_set:
        cmd += ["-s", s]

    result = ComboResult(idx=idx, params=dict(params))
    t0 = time.perf_counter()
    with GpuMemorySampler(1.0) as sampler:
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
        if result.metrics.ba_rmse_final_px is None and result.metrics.reg_3d_error_mm is None:
            result.status = "OK_NO_METRICS"
            result.error = "run completed but no metrics parsed (check raw log)"
    return result


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
            "reg_3d_error_mm": md.reg_3d_error_mm,
            "ba_rmse_final_px": md.ba_rmse_final_px,
            "reg_2d_error_px": md.reg_2d_error_px,
            "rejected_views": md.rejected_views,
            "views_kept": md.views_kept,
            "views_total": md.views_total,
            "reject_skipped": md.reject_skipped,
            "qc_plane_rms_mm": md.qc_plane_rms_mm,
            "qc_reproj_p95_px": md.qc_reproj_p95_px,
            "reg_best_res_reg": md.reg_best_res_reg,
            "mesh_vertices": md.mesh_vertices,
            "wall_s": round(r.wall_s, 1),
            "peak_gpu_mib": r.peak_gpu_mib,
        })
        rows.append(row)

    df = pd.DataFrame(rows)
    # Keep the natural sweep order (descending threshold) for readability,
    # but also mark the best.
    df_sorted = df.sort_values(
        by=["max_view_alignment_reproj_rmse_px"], ascending=False, na_position="last"
    ).reset_index(drop=True)

    timings = df[["combo", "status", "wall_s", "peak_gpu_mib"]].copy()
    memory = df[["combo", "status", "peak_gpu_mib", "fused_points", "mesh_vertices"]].copy() \
        if "fused_points" in df.columns else df[["combo", "status", "peak_gpu_mib"]].copy()
    failures = df[df["status"] != "OK"][["combo", "status"]].copy()
    failures["error"] = [
        next((r.error for r in results if r.idx == c), "") for c in failures["combo"]
    ]

    xlsx = out_dir / "reject_sweep_report.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        df_sorted.to_excel(xw, sheet_name="results", index=False)
        timings.to_excel(xw, sheet_name="timings", index=False)
        memory.to_excel(xw, sheet_name="memory", index=False)
        failures.to_excel(xw, sheet_name="failures", index=False)

    _write_markdown(df_sorted, results, out_dir / "reject_sweep_report.md")
    print(f"\n[report] Excel    -> {xlsx}")
    print(f"[report] Markdown -> {out_dir / 'reject_sweep_report.md'}")


def _write_markdown(df_sorted, results, path: Path) -> None:
    ok = [r for r in results if r.status in ("OK", "OK_NO_METRICS")]
    total_time = sum(r.wall_s for r in results)

    lines = [
        "# Reject-Threshold Sweep Report",
        "",
        f"- Combos run: **{len(results)}** (OK: {len(ok)})",
        f"- Total wall-clock: **{total_time/60:.1f} min**",
        "",
        "Fixed: ba_huber_delta_px=1.0, voxel_size=0.0015, conf_threshold=3.0.",
        "Varying only `max_view_alignment_reproj_rmse_px`.",
        "",
        "**Read `reject_skipped`**: if True, the safety cutoff (min_kept_views) fired "
        "and the rejection was refused — that row's error is the no-reject fallback, "
        "not a real aggressive-reject result.",
        "",
        "| rej_thr (px) | 3D err (mm) | rejected | kept/total | reject_skipped "
        "| BA RMSE (px) | plane RMS (mm) | t (s) | status |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df_sorted.iterrows():
        kt = f"{r.get('views_kept','')}/{r.get('views_total','')}"
        lines.append(
            "| {thr} | {err} | {rej} | {kt} | {skip} | {ba} | {plane} | {t} | {st} |".format(
                thr=r.get("max_view_alignment_reproj_rmse_px", ""),
                err=r.get("reg_3d_error_mm", ""),
                rej=r.get("rejected_views", ""),
                kt=kt,
                skip=r.get("reject_skipped", ""),
                ba=r.get("ba_rmse_final_px", ""),
                plane=r.get("qc_plane_rms_mm", ""),
                t=r.get("wall_s", ""),
                st=r.get("status", ""),
            )
        )
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-c", "--config", default="configs/default.yaml")
    ap.add_argument("--sample", default="SAMPLE2")
    ap.add_argument("--out-dir", default="RESULTS/reject_sweep")
    ap.add_argument("--extra-set", action="append", default=["vision.rerun.enabled=false"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    combos = build_grid()
    out_dir = Path(args.out_dir)
    raw_log_dir = out_dir / "raw_logs"
    raw_log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[sweep] {len(combos)} combos; sample={args.sample}; config={args.config}")
    if args.dry_run:
        for i, c in enumerate(combos):
            sets = " ".join(f"-s {k}={_fmt(v)}" for k, v in c.items())
            print(f"  [{i:02d}] {sets}")
        return 0

    results: List[ComboResult] = []
    for i, combo in enumerate(combos):
        thr = combo["vision.aruco.max_view_alignment_reproj_rmse_px"]
        print(f"\n{'='*70}\n[sweep] Combo {i+1}/{len(combos)}  rej_thr={thr}\n{'='*70}")
        res = run_combo(i, combo, args.config, args.sample, raw_log_dir, args.extra_set)
        mm = res.metrics
        print(f"[sweep] -> {res.status}  3D={mm.reg_3d_error_mm}mm  "
              f"rejected={mm.rejected_views} kept={mm.views_kept}/{mm.views_total} "
              f"skipped={mm.reject_skipped}  t={res.wall_s:.0f}s")
        results.append(res)
        write_reports(results, out_dir)   # incremental

    print(f"\n[sweep] DONE. {sum(r.status=='OK' for r in results)}/{len(results)} OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
