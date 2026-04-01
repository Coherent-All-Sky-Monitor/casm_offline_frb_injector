#!/usr/bin/env python3
"""
Batch driver: generate many FRB injection filterbanks by calling
``inject_frb_dedigitized.py`` as a subprocess for crash-isolation.

Supports parallel execution via ``--nworkers``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import subprocess
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .inject_frb import (
    CASM_FCH1,
    CASM_FOFF,
    CASM_NCHANS,
    CASM_TSAMP,
    CASM_NSAMPLES,
    CASM_NBITS,
    CASM_TELESCOPE_ID,
)


# ===================================================================
# InjectionParameterSampler
# ===================================================================
class InjectionParameterSampler:
    """Draw random FRB parameters from configured distributions.

    Parameters
    ----------
    n : int
        Number of injections to draw.
    rng_seed : int
        Seed for the parameter RNG.
    snr_range : tuple[float, float]
        (min, max) for SNR.
    dm_range : tuple[float, float]
        (min, max) for DM in pc cm-3.
    width_ms_range : tuple[float, float]
        (min, max) for pulse width in ms.
    pos_range : tuple[float, float]
        (min, max) for pulse position fraction.
    dist_snr : str
        Distribution for SNR: ``'uniform'`` or ``'loguniform'``.
    dist_dm : str
        Distribution for DM.
    dist_width : str
        Distribution for width.
    tsamp : float
        Sampling time in seconds (to convert width_ms to fwhm_samples).
    """

    def __init__(
        self,
        n: int,
        rng_seed: int = 12345,
        snr_range: tuple[float, float] = (6.5, 65.0),
        dm_range: tuple[float, float] = (20.0, 2000.0),
        width_ms_range: tuple[float, float] = (0.262, 20.0),
        pos_range: tuple[float, float] = (0.1, 0.9),
        dist_snr: str = "uniform",
        dist_dm: str = "uniform",
        dist_width: str = "loguniform",
        tsamp: float = CASM_TSAMP,
    ) -> None:
        self._n = int(n)
        self._rng = np.random.default_rng(rng_seed)
        self._snr_range = snr_range
        self._dm_range = dm_range
        self._width_ms_range = width_ms_range
        self._pos_range = pos_range
        self._dist_snr = dist_snr
        self._dist_dm = dist_dm
        self._dist_width = dist_width
        self._tsamp = float(tsamp)

        self._parameters: list[dict] | None = None

    @property
    def parameters(self) -> list[dict]:
        """List of dicts with keys: dm, snr, width_ms, fwhm_samples, position, seed."""
        if self._parameters is None:
            self._parameters = self._draw()
        return self._parameters

    def _draw_dist(self, dist: str, lo: float, hi: float) -> np.ndarray:
        if dist == "uniform":
            return self._rng.uniform(lo, hi, size=self._n)
        if dist == "loguniform":
            return np.exp(self._rng.uniform(np.log(lo), np.log(hi), size=self._n))
        raise ValueError(f"Unknown distribution: {dist!r}")

    def _draw(self) -> list[dict]:
        snrs = self._draw_dist(self._dist_snr, *self._snr_range)
        dms = self._draw_dist(self._dist_dm, *self._dm_range)
        widths = self._draw_dist(self._dist_width, *self._width_ms_range)
        positions = self._rng.uniform(*self._pos_range, size=self._n)
        seeds = self._rng.integers(1, 2_147_483_647, size=self._n, dtype=np.int64)

        params = []
        for i in range(self._n):
            width_ms = float(widths[i])
            fwhm_samples = (width_ms * 1e-3) / self._tsamp
            params.append({
                "dm": float(dms[i]),
                "snr": float(snrs[i]),
                "width_ms": width_ms,
                "fwhm_samples": fwhm_samples,
                "position": float(positions[i]),
                "seed": int(seeds[i]),
            })
        return params


# ===================================================================
# BatchInjector
# ===================================================================

def _run_one_injection(cmd: list[str], out_fil: str) -> tuple[int, bool, str, str]:
    """Run a single injection subprocess (top-level for pickling).

    Returns
    -------
    tuple[int, bool, str, str]
        (returncode, success, stdout, stderr).
    """
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    out_path = Path(out_fil)
    success = proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0
    return proc.returncode, success, proc.stdout, proc.stderr


class BatchInjector:
    """Run batch FRB injections via subprocess calls.

    Parameters
    ----------
    outdir : Path
        Output directory for ``.fil`` files and manifest.
    sampler : InjectionParameterSampler
        Parameter sampler instance.
    fch1 : float
        Top frequency in MHz.
    foff : float
        Channel bandwidth in MHz.
    nchans : int
        Number of channels.
    tsamp : float
        Sampling time in seconds.
    nsamples : int
        Number of time samples.
    nbits : int
        Bits per sample.
    telescope_id : int
        SIGPROC telescope ID.
    nworkers : int
        Number of parallel workers (default: 1 = sequential).
    dry_run : bool
        If True, print commands but do not execute.
    stop_on_error : bool
        Abort on first failed injection.
    verbose : bool
        Print progress.
    """

    _MANIFEST_FIELDS = [
        "id", "output_fil",
        "dm", "snr", "width_ms", "tsamp_s", "fwhm_samples",
        "seed", "position",
        "cmd", "returncode", "status", "timestamp_utc",
    ]

    def __init__(
        self,
        outdir: Path,
        sampler: InjectionParameterSampler,
        fch1: float = CASM_FCH1,
        foff: float = CASM_FOFF,
        nchans: int = CASM_NCHANS,
        tsamp: float = CASM_TSAMP,
        nsamples: int = CASM_NSAMPLES,
        nbits: int = CASM_NBITS,
        telescope_id: int = CASM_TELESCOPE_ID,
        nworkers: int = 1,
        dry_run: bool = False,
        stop_on_error: bool = False,
        verbose: bool = True,
    ) -> None:
        self._outdir = Path(outdir).resolve()
        self._sampler = sampler
        self._fch1 = fch1
        self._foff = foff
        self._nchans = nchans
        self._tsamp = tsamp
        self._nsamples = nsamples
        self._nbits = nbits
        self._telescope_id = telescope_id
        self._nworkers = max(1, min(nworkers, os.cpu_count() or 16))
        self._dry_run = dry_run
        self._stop_on_error = stop_on_error
        self._verbose = verbose

    def run(self) -> Path:
        """Execute all injections and return the manifest path.

        Returns
        -------
        Path
            Path to the written ``injections_manifest.csv``.
        """
        self._outdir.mkdir(parents=True, exist_ok=True)
        manifest_path = self._outdir / "injections_manifest.csv"
        errlog_path = self._outdir / "injections_errors.log"

        params = self._sampler.parameters
        timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Build all jobs first
        jobs = []
        for i, p in enumerate(params):
            inj_id = self._make_id(i, p)
            out_fil = self._outdir / f"{inj_id}.fil"

            cmd = [
                sys.executable, "-m", "casm_offline_frb_injector.inject_frb_dedigitized",
                "--output", str(out_fil),
                "--dm", f"{p['dm']:.6f}",
                "--fwhm", f"{p['fwhm_samples']:.6f}",
                "--snr", f"{p['snr']:.6f}",
                "--seed", str(p["seed"]),
                "--fch1", str(self._fch1),
                "--foff", str(self._foff),
                "--nchans", str(self._nchans),
                "--tsamp", str(self._tsamp),
                "--nsamples", str(self._nsamples),
                "--nbits", str(self._nbits),
            ]

            row = {
                "id": inj_id,
                "output_fil": str(out_fil),
                "dm": f"{p['dm']:.6f}",
                "snr": f"{p['snr']:.6f}",
                "width_ms": f"{p['width_ms']:.6f}",
                "tsamp_s": f"{self._tsamp:.12g}",
                "fwhm_samples": f"{p['fwhm_samples']:.6f}",
                "seed": str(p["seed"]),
                "position": f"{p['position']:.6f}",
                "cmd": " ".join(cmd),
                "returncode": "",
                "status": "DRY_RUN" if self._dry_run else "",
                "timestamp_utc": timestamp,
            }
            jobs.append((inj_id, cmd, str(out_fil), row))

        if self._dry_run:
            with manifest_path.open("w", newline="") as fcsv:
                writer = csv.DictWriter(fcsv, fieldnames=self._MANIFEST_FIELDS)
                writer.writeheader()
                for inj_id, cmd, out_fil, row in jobs:
                    if self._verbose:
                        print(row["cmd"])
                    writer.writerow(row)
            return manifest_path

        # Write manifest header
        write_lock = threading.Lock()
        with manifest_path.open("w", newline="") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=self._MANIFEST_FIELDS)
            writer.writeheader()

        n_ok = 0
        n_fail = 0

        if self._nworkers <= 1:
            # Sequential execution
            for inj_id, cmd, out_fil, row in jobs:
                rc, success, stdout, stderr = _run_one_injection(cmd, out_fil)
                row["returncode"] = str(rc)
                row["status"] = "OK" if success else "FAIL"
                self._append_row(manifest_path, row, write_lock)
                if success:
                    n_ok += 1
                else:
                    n_fail += 1
                    self._write_error(errlog_path, timestamp, inj_id,
                                      row["cmd"], rc, stdout, stderr)
                    if self._stop_on_error:
                        raise RuntimeError(f"Injection failed: {inj_id}")
                if self._verbose:
                    print(f"  [{n_ok + n_fail}/{len(jobs)}] {inj_id}: {'OK' if success else 'FAIL'}")
        else:
            # Parallel execution
            if self._verbose:
                print(f"Running {len(jobs)} injections with {self._nworkers} workers...")
            futures = {}
            failed = False
            with ProcessPoolExecutor(max_workers=self._nworkers) as ex:
                for inj_id, cmd, out_fil, row in jobs:
                    if failed:
                        break
                    fut = ex.submit(_run_one_injection, cmd, out_fil)
                    futures[fut] = (inj_id, row)

                for fut in as_completed(futures):
                    inj_id, row = futures[fut]
                    try:
                        rc, success, stdout, stderr = fut.result()
                    except Exception as e:
                        rc, success, stdout, stderr = -1, False, "", str(e)
                    row["returncode"] = str(rc)
                    row["status"] = "OK" if success else "FAIL"
                    self._append_row(manifest_path, row, write_lock)
                    if success:
                        n_ok += 1
                    else:
                        n_fail += 1
                        self._write_error(errlog_path, timestamp, inj_id,
                                          row["cmd"], rc, stdout, stderr)
                        if self._stop_on_error:
                            failed = True
                            ex.shutdown(wait=False, cancel_futures=True)
                            break
                    if self._verbose:
                        print(f"  [{n_ok + n_fail}/{len(jobs)}] {inj_id}: {'OK' if success else 'FAIL'}")
            if failed:
                raise RuntimeError(f"Injection failed: {inj_id}")

        if self._verbose:
            print(f"Wrote manifest: {manifest_path}")
            print(f"Generated: {n_ok} OK, {n_fail} FAIL out of {len(params)} injections")
            print(f"Output dir: {self._outdir}")

        return manifest_path

    def _append_row(self, manifest_path: Path, row: dict,
                    lock: threading.Lock) -> None:
        with lock:
            with manifest_path.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self._MANIFEST_FIELDS,
                                   extrasaction="ignore")
                w.writerow(row)
                f.flush()
                os.fsync(f.fileno())

    @staticmethod
    def _make_id(i: int, p: dict) -> str:
        dm_i = int(round(p["dm"] * 10))
        snr_i = int(round(p["snr"] * 10))
        w_i = int(round(p["width_ms"] * 1000))
        p_i = int(round(p["position"] * 1000))
        return f"inj{i:05d}_dm{dm_i:05d}_snr{snr_i:04d}_w{w_i:05d}_p{p_i:04d}_seed{p['seed']:d}"

    @staticmethod
    def _write_error(errlog: Path, timestamp: str, inj_id: str,
                     cmd_str: str, returncode: int | None,
                     stdout: str, stderr: str) -> None:
        with errlog.open("a") as ef:
            ef.write(f"\n[{timestamp}] {inj_id}\n")
            ef.write(f"CMD: {cmd_str}\n")
            if returncode is not None:
                ef.write(f"RETURN: {returncode}\n")
            ef.write(f"STDOUT:\n{stdout}\n")
            ef.write(f"STDERR:\n{stderr}\n")


# ===================================================================
# CLI
# ===================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Batch driver to call inject_frb_dedigitized.py for many injections.",
    )
    ap.add_argument("--outdir", required=True, type=Path,
                    help="Output directory for .fil files")
    ap.add_argument("--manifest", default=None, type=Path,
                    help="CSV manifest path (default: <outdir>/injections_manifest.csv)")
    ap.add_argument("--ninj", type=int, default=2000,
                    help="Number of injections (default: 2000)")

    # Ranges
    ap.add_argument("--snr_min", type=float, default=6.5)
    ap.add_argument("--snr_max", type=float, default=65.0)
    ap.add_argument("--dm_min", type=float, default=20.0)
    ap.add_argument("--dm_max", type=float, default=2000.0)
    ap.add_argument("--wms_min", type=float, default=0.262,
                    help="Width min in ms (default: 0.262)")
    ap.add_argument("--wms_max", type=float, default=20.0,
                    help="Width max in ms (default: 20)")
    ap.add_argument("--pos_min", type=float, default=0.1)
    ap.add_argument("--pos_max", type=float, default=0.9)

    # Distributions
    ap.add_argument("--dist_snr", choices=["uniform", "loguniform"], default="uniform")
    ap.add_argument("--dist_dm", choices=["uniform", "loguniform"], default="uniform")
    ap.add_argument("--dist_width", choices=["uniform", "loguniform"], default="loguniform")

    # Reproducibility
    ap.add_argument("--rng_seed", type=int, default=12345)

    # CASM defaults (overridable)
    ap.add_argument("--fch1", type=float, default=CASM_FCH1)
    ap.add_argument("--foff", type=float, default=CASM_FOFF)
    ap.add_argument("--nchans", type=int, default=CASM_NCHANS)
    ap.add_argument("--tsamp", type=float, default=CASM_TSAMP)
    ap.add_argument("--nsamples", type=int, default=CASM_NSAMPLES)
    ap.add_argument("--nbits", type=int, default=CASM_NBITS)

    # Parallelism
    ap.add_argument("--nworkers", type=int, default=1,
                    help="Number of parallel workers (default: 1 = sequential)")

    # Execution
    ap.add_argument("--dry_run", action="store_true",
                    help="Print commands, do not execute")
    ap.add_argument("--stop_on_error", action="store_true",
                    help="Abort on first failure")
    ap.add_argument("--quiet", "-q", action="store_true")

    args = ap.parse_args()

    sampler = InjectionParameterSampler(
        n=args.ninj,
        rng_seed=args.rng_seed,
        snr_range=(args.snr_min, args.snr_max),
        dm_range=(args.dm_min, args.dm_max),
        width_ms_range=(args.wms_min, args.wms_max),
        pos_range=(args.pos_min, args.pos_max),
        dist_snr=args.dist_snr,
        dist_dm=args.dist_dm,
        dist_width=args.dist_width,
        tsamp=args.tsamp,
    )

    injector = BatchInjector(
        outdir=args.outdir,
        sampler=sampler,
        fch1=args.fch1,
        foff=args.foff,
        nchans=args.nchans,
        tsamp=args.tsamp,
        nsamples=args.nsamples,
        nbits=args.nbits,
        nworkers=args.nworkers,
        dry_run=args.dry_run,
        stop_on_error=args.stop_on_error,
        verbose=not args.quiet,
    )
    injector.run()


if __name__ == "__main__":
    main()
