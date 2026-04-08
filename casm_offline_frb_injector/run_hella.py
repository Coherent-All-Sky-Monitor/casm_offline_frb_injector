#!/usr/bin/env python3
"""
Resumable multi-GPU Hella batch runner with candidate matching against
injection ground truth.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from casm_io.candidates import CandidateReader
from casm_io.filterbank import FilterbankFile

from .inject_frb import (
    _matched_filter_snr,
    CASM_FCH1,
    CASM_FOFF,
    CASM_NCHANS,
    CASM_TSAMP,
    CASM_NSAMPLES,
)

# Default gulp matches CASM_NSAMPLES
CASM_GULP: int = CASM_NSAMPLES


# ===================================================================
# HellaVersion
# ===================================================================
@dataclass(frozen=True)
class HellaVersion:
    """A Hella executable with version tag."""

    tag: str
    exe: Path


# ===================================================================
# ExpectedBoxcar
# ===================================================================
class ExpectedBoxcar:
    """Compute expected ibox from DM smearing + intrinsic width.

    Parameters
    ----------
    fch1 : float
        Top frequency in MHz.
    foff : float
        Channel bandwidth in MHz (negative for descending).
    nchans : int
        Number of frequency channels.
    tsamp : float
        Sampling time in seconds.
    """

    def __init__(
        self,
        fch1: float = CASM_FCH1,
        foff: float = CASM_FOFF,
        nchans: int = CASM_NCHANS,
        tsamp: float = CASM_TSAMP,
    ) -> None:
        self._fch1 = fch1
        self._foff = foff
        self._nchans = nchans
        self._tsamp = tsamp

        freq_hi = fch1
        freq_lo = fch1 + nchans * foff
        self._nu_center = 0.5 * (freq_hi + freq_lo)
        self._chan_bw = abs(foff)

    # The v*v squaring in smooth.cpp narrows the effective kernel FWHM
    # to ~0.677*sm (measured empirically). A pulse of width w_eff is
    # best matched by the kernel where 0.677*sm ~ w_eff, i.e.,
    # sm ~ w_eff / 0.677 ~ 1.48 * w_eff. We find the nearest power-of-2
    # sm to that value, then convert to boxcar index.
    _KERNEL_NARROWING = 0.677

    def compute(self, width_ms: float, dm: float) -> dict:
        """Compute expected boxcar index.

        Accounts for the v*v kernel narrowing: the kernel at sm has
        effective FWHM of ~0.677*sm (not sm), so the best-matching
        kernel for a pulse of width w_eff is sm ~ w_eff / 0.677.

        Parameters
        ----------
        width_ms : float
            Intrinsic pulse width in milliseconds.
        dm : float
            Dispersion measure in pc cm-3.

        Returns
        -------
        dict
            Keys: ibox, w_intr_samples, tau_dm_samples, w_eff_samples.
        """
        w_intr = (width_ms / 1000.0) / self._tsamp
        tau_dm = 8.3e3 * dm * self._chan_bw / (self._nu_center ** 3) / self._tsamp
        w_eff = max(math.sqrt(w_intr ** 2 + tau_dm ** 2), 1.0)

        # Best-matching sm accounts for kernel narrowing
        best_sm = w_eff / self._KERNEL_NARROWING
        nearest_pow2 = 2 ** int(round(math.log(max(best_sm, 1), 2)))
        ibox = int(round(math.log(nearest_pow2, 2))) if nearest_pow2 >= 1 else 0

        return {
            "ibox": ibox,
            "w_intr_samples": w_intr,
            "tau_dm_samples": tau_dm,
            "w_eff_samples": w_eff,
        }


# ===================================================================
# CandidateMatcher
# ===================================================================
class CandidateMatcher:
    """Match Hella output candidates against injection truth.

    Uses nearest-to-expected matching: filters by DM window AND time
    window around the expected pulse position, then picks highest SNR.

    Parameters
    ----------
    dm_window : float
        Floor of the DM matching window in pc cm-3.
    dm_window_frac : float
        Fractional DM window. Effective window is
        ``max(dm_window, dm_window_frac * dm_true)``.

        The default 0.06 accounts for:
        - 4.63% systematic offset from the TIME_RESOLUTION bug
          (dt=1.0e-3 vs actual 1.048576e-3, ratio=0.9537)
        - ~1.4% margin for DM trial spacing scatter (dm_tol=1.3)
    time_window_fwhm_factor : float
        Time window is ``max(time_window_fwhm_factor * fwhm_samples, 10)``
        samples around the expected pulse position.
    """

    def __init__(
        self,
        dm_window: float = 15.0,
        dm_window_frac: float = 0.06,
        time_window_fwhm_factor: float = 2.0,
    ) -> None:
        self._dm_window = dm_window
        self._dm_window_frac = dm_window_frac
        self._time_window_fwhm_factor = time_window_fwhm_factor

    def effective_dm_window(self, dm_true: float) -> float:
        """Compute effective DM matching window."""
        return max(self._dm_window, self._dm_window_frac * abs(dm_true))

    def effective_time_window(self, fwhm_samples: float) -> float:
        """Compute effective time matching window in samples."""
        return max(self._time_window_fwhm_factor * fwhm_samples, 10.0)

    def match(
        self,
        cand_df: pd.DataFrame,
        dm_true: float,
        expected_sample: int | None = None,
        fwhm_samples: float | None = None,
    ) -> dict:
        """Find best candidate within DM and time windows.

        Parameters
        ----------
        cand_df : pandas.DataFrame
            Candidate table with columns from CandidateReader.
        dm_true : float
            True injection DM.
        expected_sample : int, optional
            Expected pulse sample index. If provided with fwhm_samples,
            applies a time window filter to avoid matching noise peaks.
        fwhm_samples : float, optional
            Pulse FWHM in samples. Used to set the time window width.

        Returns
        -------
        dict
            Keys: detected (0/1), n_matches, best (Series or None).
        """
        if cand_df.empty:
            return {"detected": 0, "n_matches": 0, "best": None}

        dm_win = self.effective_dm_window(dm_true)
        dm_mask = (cand_df["dm"] - dm_true).abs() <= dm_win

        if expected_sample is not None and fwhm_samples is not None:
            time_win = self.effective_time_window(fwhm_samples)
            time_mask = (cand_df["sample_index"] - expected_sample).abs() <= time_win
            within = cand_df.loc[dm_mask & time_mask]
        else:
            within = cand_df.loc[dm_mask]

        n_matches = len(within)
        if n_matches == 0:
            return {"detected": 0, "n_matches": 0, "best": None}

        best_idx = within["snr"].idxmax()
        return {"detected": 1, "n_matches": n_matches, "best": within.loc[best_idx]}


# ===================================================================
# PulseVerifier
# ===================================================================
class PulseVerifier:
    """Independently verify pulse presence via matched-filter on filterbank.

    Reads the filterbank, dedisperses at the known DM, and measures
    matched-filter SNR using the exact injected pulse width. This gives
    a ground-truth detection answer independent of hella's normalization.

    Parameters
    ----------
    snr_threshold : float
        Minimum matched-filter SNR to count as verified.
    """

    def __init__(self, snr_threshold: float = 6.5) -> None:
        self._threshold = snr_threshold

    def verify(
        self,
        fil_path: str | Path,
        dm: float,
        fwhm_samples: float,
        fch1: float = CASM_FCH1,
        foff: float = CASM_FOFF,
        tsamp: float = CASM_TSAMP,
    ) -> dict:
        """Dedisperse filterbank at known DM and measure matched-filter SNR.

        Parameters
        ----------
        fil_path : str | Path
            Path to the filterbank file.
        dm : float
            True dispersion measure.
        fwhm_samples : float
            True pulse FWHM in samples.
        fch1 : float
            Top frequency in MHz.
        foff : float
            Channel bandwidth in MHz.
        tsamp : float
            Sampling time in seconds.

        Returns
        -------
        dict
            Keys: verified (bool), mf_snr (float), mf_peak_idx (int).
        """
        try:
            fb = FilterbankFile(str(fil_path), verbose=False)
            data = fb.data  # (nsamples, nchans)
            freq_mhz = fb.freq_mhz

            # Dedisperse: shift each channel by DM delay, sum across channels
            from casm_io.filterbank.plotting import _dedisperse
            dd = _dedisperse(data, dm, freq_mhz, tsamp)
            # dd is (nsamples_trimmed, nchans) — average across channels
            if dd.ndim == 2:
                ts = dd.mean(axis=1)
            else:
                ts = dd

            snr, peak_val, peak_idx = _matched_filter_snr(
                ts.astype(np.float64), fwhm_samples
            )
            return {
                "verified": bool(snr >= self._threshold),
                "mf_snr": float(snr),
                "mf_peak_idx": int(peak_idx),
            }
        except Exception as e:
            return {
                "verified": False,
                "mf_snr": np.nan,
                "mf_peak_idx": -1,
            }


# ===================================================================
# HellaRunner
# ===================================================================

HELLA_CFG_TEMPLATE = """\
INPUT FILTERBANK
INPUT_PATH {input_path}
DM_MIN {dm_min}
DM_MAX {dm_max}
WIDTH_MIN {width_min}
WIDTH_MAX {width_max}
SNR {snr_min}
GULP {gulp}
NBEAM {nbeam}
OUTPUT FILE
BEAM0 0
BEAM_OFFSET 0
GPU {gpu}
SPEC_MIN {spec_min}
SPEC_MAX {spec_max}
OUTPUT_BANDPASS 0
OUTPUTPATH {output_path}
SPECFLAGS /dev/null
BEAMFLAGS /dev/null
SCRUNCH 2
4 4 4.0 1
1 16 4.0 1
"""

SUMMARY_HEADER = [
    "id", "version_tag", "gpu_used",
    "hella_executable", "input_fil", "config_file", "candidate_file",
    "hella_returncode", "n_candidates", "n_matches",
    "dm_true", "snr_injected", "width_ms", "position",
    "expected_ibox", "w_intr_samples", "tau_dm_samples", "w_eff_samples",
    "detected", "snr_rec", "recovered_fraction",
    "dm_rec", "dm_diff", "abs_dm_diff",
    "sample_index_rec", "boxcar_width_rec", "time_start_rec", "beam_index_rec",
    "verified", "mf_snr", "mf_peak_idx",
]


class HellaRunner:
    """Resumable multi-GPU Hella batch runner.

    Parameters
    ----------
    input_dir : Path
        Directory containing ``injections_manifest.csv`` and ``.fil`` files.
    versions : list[HellaVersion]
        Hella executables to benchmark.
    gpu_list : list[int]
        GPU device IDs to use.
    max_workers : int | None
        Thread pool size (default: ``len(gpu_list)``).
    manifest_name : str
        Manifest CSV filename.
    dm_min : float
        Hella config DM lower bound.
    dm_max : float
        Hella config DM upper bound.
    width_min : int
        Hella config minimum boxcar width.
    width_max : int
        Hella config maximum boxcar width.
    snr_min : float
        Hella config SNR threshold.
    gulp : int
        Hella gulp size.
    nbeam : int
        Number of beams (1 for single-beam filterbanks).
    spec_min : float
        Hella spectral index min.
    spec_max : float
        Hella spectral index max.
    dm_window : float
        DM matching window.
    fch1 : float
        Top frequency in MHz (for expected boxcar).
    foff : float
        Channel bandwidth in MHz.
    nchans : int
        Number of channels.
    tsamp : float
        Sampling time in seconds.
    resume_mode : str
        ``'ok'``, ``'any'``, or ``'none'``.
    stop_on_error : bool
        Abort on first failure.
    """

    def __init__(
        self,
        input_dir: Path,
        versions: list[HellaVersion],
        gpu_list: list[int],
        max_workers: int | None = None,
        manifest_name: str = "injections_manifest.csv",
        dm_min: float = 20.0,
        dm_max: float = 2000.0,
        width_min: int = 1,
        width_max: int = 65,
        snr_min: float = 6.5,
        gulp: int = CASM_GULP,
        nbeam: int = 1,
        spec_min: float = -0.07,
        spec_max: float = 0.08,
        dm_window: float = 15.0,
        dm_window_frac: float = 0.06,
        fch1: float = CASM_FCH1,
        foff: float = CASM_FOFF,
        nchans: int = CASM_NCHANS,
        tsamp: float = CASM_TSAMP,
        resume_mode: str = "ok",
        stop_on_error: bool = False,
        verify: bool = True,
    ) -> None:
        self._input_dir = Path(input_dir).resolve()
        self._versions = versions
        self._gpu_list = gpu_list
        self._max_workers = max_workers if max_workers is not None else len(gpu_list)
        self._manifest_name = manifest_name

        self._dm_min = dm_min
        self._dm_max = dm_max
        self._width_min = width_min
        self._width_max = width_max
        self._snr_min = snr_min
        self._gulp = gulp
        self._nbeam = nbeam
        self._spec_min = spec_min
        self._spec_max = spec_max

        self._matcher = CandidateMatcher(dm_window, dm_window_frac)
        self._boxcar = ExpectedBoxcar(fch1, foff, nchans, tsamp)
        self._resume_mode = resume_mode
        self._stop_on_error = stop_on_error
        self._verify = verify
        self._verifier = PulseVerifier(snr_min) if verify else None

    def run(self) -> None:
        """Execute the Hella search across all versions and injections."""
        manifest_path = (self._input_dir / self._manifest_name).resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        inj = pd.read_csv(manifest_path)
        if "status" in inj.columns:
            inj = inj[inj["status"] == "OK"].copy()
        inj = inj.reset_index(drop=True)

        cfg_root = self._input_dir / "hella_configs"
        res_root = self._input_dir / "hella_results"
        sum_root = self._input_dir / "hella_summaries"
        log_root = self._input_dir / "hella_logs"
        for d in (cfg_root, res_root, sum_root, log_root):
            d.mkdir(parents=True, exist_ok=True)

        for v in self._versions:
            if not v.exe.exists():
                raise FileNotFoundError(f"Hella executable not found: {v.exe}")

            cfg_dir = cfg_root / v.tag
            res_dir = res_root / v.tag
            cfg_dir.mkdir(parents=True, exist_ok=True)
            res_dir.mkdir(parents=True, exist_ok=True)

            log_path = log_root / f"{v.tag}.log"
            log_path.write_text(f"Run started for {v.tag}\n")

            summary_path = sum_root / f"{v.tag}_summary.csv"
            completed = self._load_completed(summary_path)

            todo = inj[~inj["id"].astype(str).isin(completed)].reset_index(drop=True)
            print(f"{v.tag}: total={len(inj)}, completed={len(completed)}, remaining={len(todo)}")
            if len(todo) == 0:
                continue

            write_lock = threading.Lock()
            futures = []

            with ThreadPoolExecutor(max_workers=self._max_workers) as ex:
                for i, (_, row) in enumerate(todo.iterrows()):
                    gpu_id = self._gpu_list[i % len(self._gpu_list)]
                    futures.append(ex.submit(
                        self._run_one, v, row, gpu_id,
                        cfg_dir, res_dir, log_path,
                    ))

                for fut in as_completed(futures):
                    try:
                        result_row = fut.result()
                        self._atomic_append(summary_path, result_row, write_lock)
                    except Exception as e:
                        with log_path.open("a") as lf:
                            lf.write(f"\n!!! EXCEPTION: {e!r}\n")
                        if self._stop_on_error:
                            raise

            print(f"Wrote/updated summary: {summary_path}")

    # -- internals ----------------------------------------------------------

    def _run_one(
        self,
        version: HellaVersion,
        inj_row: pd.Series,
        gpu_id: int,
        cfg_dir: Path,
        res_dir: Path,
        log_path: Path,
    ) -> dict:
        inj_id = str(inj_row["id"])
        input_fil = Path(str(inj_row["output_fil"])).resolve()

        cfg_file = cfg_dir / f"{inj_id}.cfg"
        out_file = res_dir / f"{inj_id}_candidate_output.out"

        cfg_text = HELLA_CFG_TEMPLATE.format(
            input_path=str(input_fil),
            dm_min=f"{self._dm_min:g}",
            dm_max=f"{self._dm_max:g}",
            width_min=str(self._width_min),
            width_max=str(self._width_max),
            snr_min=f"{self._snr_min:g}",
            gulp=str(self._gulp),
            nbeam=str(self._nbeam),
            gpu=str(gpu_id),
            spec_min=f"{self._spec_min:g}",
            spec_max=f"{self._spec_max:g}",
            output_path=str(out_file),
        )
        cfg_file.write_text(cfg_text)

        cmd = [str(version.exe), "-c", str(cfg_file)]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)

        with log_path.open("a") as lf:
            lf.write(f"\n=== {inj_id} ===\nCMD: {' '.join(cmd)}\n"
                     f"RETURN: {proc.returncode}\n")
            if proc.stdout:
                lf.write(f"STDOUT:\n{proc.stdout}\n")
            if proc.stderr:
                lf.write(f"STDERR:\n{proc.stderr}\n")

        # Read candidates using casm_io CandidateReader
        try:
            if out_file.exists() and out_file.stat().st_size > 0:
                cand_df = CandidateReader(str(out_file)).df
            else:
                cand_df = pd.DataFrame()
        except Exception:
            cand_df = pd.DataFrame()

        dm_true = float(inj_row["dm"])
        match = self._matcher.match(cand_df, dm_true)

        width_ms = float(inj_row["width_ms"])
        expected = self._boxcar.compute(width_ms, dm_true)

        snr_inj = float(inj_row["snr"])
        best = match["best"]
        detected = match["detected"]

        snr_rec = float(best["snr"]) if detected else 0.0
        recovered_fraction = (snr_rec / snr_inj) if snr_inj > 0 else np.nan
        dm_rec = float(best["dm"]) if detected else np.nan
        dm_diff = (dm_rec - dm_true) if not np.isnan(dm_rec) else np.nan
        abs_dm_diff = abs(dm_diff) if not np.isnan(dm_diff) else np.nan

        out: dict = {
            "id": inj_id,
            "version_tag": version.tag,
            "gpu_used": gpu_id,
            "hella_executable": str(version.exe),
            "input_fil": str(input_fil),
            "config_file": str(cfg_file),
            "candidate_file": str(out_file),
            "hella_returncode": proc.returncode,
            "n_candidates": len(cand_df),
            "n_matches": match["n_matches"],
            "dm_true": dm_true,
            "snr_injected": snr_inj,
            "width_ms": width_ms,
            "position": float(inj_row["position"]),
            "expected_ibox": expected["ibox"],
            "w_intr_samples": expected["w_intr_samples"],
            "tau_dm_samples": expected["tau_dm_samples"],
            "w_eff_samples": expected["w_eff_samples"],
            "detected": detected,
            "snr_rec": snr_rec,
            "recovered_fraction": recovered_fraction,
            "dm_rec": dm_rec,
            "dm_diff": dm_diff,
            "abs_dm_diff": abs_dm_diff,
            "sample_index_rec": np.nan,
            "boxcar_width_rec": np.nan,
            "time_start_rec": np.nan,
            "beam_index_rec": np.nan,
        }

        if detected and best is not None:
            for key, col in [("sample_index_rec", "sample_index"),
                             ("boxcar_width_rec", "boxcar_width"),
                             ("time_start_rec", "time_start"),
                             ("beam_index_rec", "beam_index")]:
                if col in best.index and not pd.isna(best[col]):
                    out[key] = int(best[col]) if col != "time_start" else float(best[col])

        # Independent matched-filter verification
        if self._verifier is not None:
            fwhm_samp = float(inj_row["fwhm_samples"])
            vr = self._verifier.verify(input_fil, dm_true, fwhm_samp)
            out["verified"] = vr["verified"]
            out["mf_snr"] = vr["mf_snr"]
            out["mf_peak_idx"] = vr["mf_peak_idx"]
        else:
            out["verified"] = np.nan
            out["mf_snr"] = np.nan
            out["mf_peak_idx"] = np.nan

        return out

    def _load_completed(self, summary_path: Path) -> set[str]:
        if self._resume_mode == "none":
            return set()
        if not summary_path.exists() or summary_path.stat().st_size == 0:
            return set()
        try:
            df = pd.read_csv(summary_path, usecols=["id", "hella_returncode"])
        except Exception:
            return set()
        if self._resume_mode == "any":
            return set(df["id"].astype(str))
        return set(df.loc[df["hella_returncode"] == 0, "id"].astype(str))

    @staticmethod
    def _atomic_append(csv_path: Path, row: dict, lock: threading.Lock) -> None:
        with lock:
            needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
            with csv_path.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=SUMMARY_HEADER, extrasaction="ignore")
                if needs_header:
                    w.writeheader()
                w.writerow(row)
                f.flush()
                os.fsync(f.fileno())


# ===================================================================
# CLI
# ===================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Resumable multi-GPU Hella batch runner.",
    )
    ap.add_argument("--input_dir", required=True, type=Path,
                    help="Directory with injections_manifest.csv and .fil files")
    ap.add_argument("--manifest", default="injections_manifest.csv",
                    help="Manifest filename (default: injections_manifest.csv)")

    # Hella executables
    ap.add_argument("--hella_v1_exe",
                    default="/home/casm/software/fourier-space/opt/casm-hella/bin/casm_hella_refactored")
    ap.add_argument("--hella_v1_tag", default="casm_hella_refactored")
    ap.add_argument("--hella_v2_exe",
                    default="/home/casm/software/dev/casm-hella-ldunn/build/src/apps/casm_hella_test")
    ap.add_argument("--hella_v2_tag", default="casm_hella_test")
    ap.add_argument("--versions_to_run", default="v1",
                    help="Comma-separated: v1,v2 (default: v1)")

    # Config
    ap.add_argument("--dm_min", type=float, default=20.0)
    ap.add_argument("--dm_max", type=float, default=2000.0)
    ap.add_argument("--width_min", type=int, default=1)
    ap.add_argument("--width_max", type=int, default=65)
    ap.add_argument("--snr_min", type=float, default=6.5)
    ap.add_argument("--gulp", type=int, default=CASM_GULP)
    ap.add_argument("--nbeam", type=int, default=1,
                    help="Number of beams (default: 1)")
    ap.add_argument("--spec_min", type=float, default=-0.07)
    ap.add_argument("--spec_max", type=float, default=0.08)

    # Matching
    ap.add_argument("--dm_window", type=float, default=15.0)
    ap.add_argument("--dm_window_frac", type=float, default=0.06,
                    help="Fractional DM window (default: 0.06 = 6%% of DM)")

    # Filterbank params for expected ibox
    ap.add_argument("--fch1", type=float, default=CASM_FCH1)
    ap.add_argument("--foff", type=float, default=CASM_FOFF)
    ap.add_argument("--nchans", type=int, default=CASM_NCHANS)
    ap.add_argument("--tsamp", type=float, default=CASM_TSAMP)

    # Parallelism
    ap.add_argument("--gpus", default="0",
                    help="Comma-separated GPU ids (default: '0')")
    ap.add_argument("--max_workers", type=int, default=None)

    # Resume
    ap.add_argument("--resume_mode", choices=["ok", "any", "none"], default="ok")
    ap.add_argument("--stop_on_error", action="store_true")
    ap.add_argument("--no_verify", action="store_true",
                    help="Disable independent matched-filter verification")

    args = ap.parse_args()

    version_map = {
        "v1": HellaVersion(tag=args.hella_v1_tag,
                           exe=Path(args.hella_v1_exe).resolve()),
        "v2": HellaVersion(tag=args.hella_v2_tag,
                           exe=Path(args.hella_v2_exe).resolve()),
    }
    chosen = [v.strip() for v in args.versions_to_run.split(",") if v.strip()]
    versions = []
    for key in chosen:
        if key not in version_map:
            raise ValueError(f"Unknown version key {key!r}. Use v1 or v2.")
        versions.append(version_map[key])

    gpu_list = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    if not gpu_list:
        raise ValueError("No GPUs specified.")

    runner = HellaRunner(
        input_dir=args.input_dir,
        versions=versions,
        gpu_list=gpu_list,
        max_workers=args.max_workers,
        manifest_name=args.manifest,
        dm_min=args.dm_min,
        dm_max=args.dm_max,
        width_min=args.width_min,
        width_max=args.width_max,
        snr_min=args.snr_min,
        gulp=args.gulp,
        nbeam=args.nbeam,
        spec_min=args.spec_min,
        spec_max=args.spec_max,
        dm_window=args.dm_window,
        dm_window_frac=args.dm_window_frac,
        fch1=args.fch1,
        foff=args.foff,
        nchans=args.nchans,
        tsamp=args.tsamp,
        resume_mode=args.resume_mode,
        stop_on_error=args.stop_on_error,
        verify=not args.no_verify,
    )
    runner.run()


if __name__ == "__main__":
    main()
