#!/usr/bin/env python3
"""
Batch driver: generate many FRB injection filterbanks by calling
``inject_frb.py`` as a subprocess for crash-isolation.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import subprocess
import sys
from pathlib import Path

import numpy as np

from inject_frb import (
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
        (min, max) for DM in pc cm⁻³.
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
        Sampling time in seconds (to convert width_ms → fwhm_samples).
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
class BatchInjector:
    """Run batch FRB injections via subprocess calls to ``inject_frb.py``.

    Parameters
    ----------
    inject_script : Path
        Path to ``inject_frb.py``.
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
        inject_script: Path,
        outdir: Path,
        sampler: InjectionParameterSampler,
        fch1: float = CASM_FCH1,
        foff: float = CASM_FOFF,
        nchans: int = CASM_NCHANS,
        tsamp: float = CASM_TSAMP,
        nsamples: int = CASM_NSAMPLES,
        nbits: int = CASM_NBITS,
        telescope_id: int = CASM_TELESCOPE_ID,
        dry_run: bool = False,
        stop_on_error: bool = False,
        verbose: bool = True,
    ) -> None:
        self._script = Path(inject_script).resolve()
        self._outdir = Path(outdir).resolve()
        self._sampler = sampler
        self._fch1 = fch1
        self._foff = foff
        self._nchans = nchans
        self._tsamp = tsamp
        self._nsamples = nsamples
        self._nbits = nbits
        self._telescope_id = telescope_id
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

        with manifest_path.open("w", newline="") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=self._MANIFEST_FIELDS)
            writer.writeheader()

            for i, p in enumerate(params):
                inj_id = self._make_id(i, p)
                out_fil = self._outdir / f"{inj_id}.fil"

                cmd = [
                    sys.executable, str(self._script),
                    "--output", str(out_fil),
                    "--dm", f"{p['dm']:.6f}",
                    "--fwhm", f"{p['fwhm_samples']:.6f}",
                    "--snr", f"{p['snr']:.6f}",
                    "--seed", str(p["seed"]),
                    "--position", f"{p['position']:.6f}",
                    "--fch1", str(self._fch1),
                    "--foff", str(self._foff),
                    "--nchans", str(self._nchans),
                    "--tsamp", str(self._tsamp),
                    "--nsamples", str(self._nsamples),
                    "--nbits", str(self._nbits),
                    "--telescope_id", str(self._telescope_id),
                ]
                cmd_str = " ".join(cmd)

                row: dict = {
                    "id": inj_id,
                    "output_fil": str(out_fil),
                    "dm": f"{p['dm']:.6f}",
                    "snr": f"{p['snr']:.6f}",
                    "width_ms": f"{p['width_ms']:.6f}",
                    "tsamp_s": f"{self._tsamp:.12g}",
                    "fwhm_samples": f"{p['fwhm_samples']:.6f}",
                    "seed": str(p["seed"]),
                    "position": f"{p['position']:.6f}",
                    "cmd": cmd_str,
                    "returncode": "",
                    "status": "DRY_RUN" if self._dry_run else "",
                    "timestamp_utc": timestamp,
                }

                if self._dry_run:
                    if self._verbose:
                        print(cmd_str)
                    writer.writerow(row)
                    continue

                proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
                row["returncode"] = str(proc.returncode)

                if proc.returncode == 0 and out_fil.exists() and out_fil.stat().st_size > 0:
                    row["status"] = "OK"
                else:
                    row["status"] = "FAIL"
                    self._write_error(errlog_path, timestamp, inj_id, cmd_str,
                                      proc.returncode, proc.stdout, proc.stderr)
                    if self._stop_on_error:
                        writer.writerow(row)
                        raise RuntimeError(f"Injection failed: {inj_id}")

                writer.writerow(row)

        if self._verbose:
            print(f"Wrote manifest: {manifest_path}")
            print(f"Generated: {len(params)} injections")
            print(f"Output dir: {self._outdir}")
            if self._dry_run:
                print("Dry run: no files were created.")

        return manifest_path

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
        description="Batch driver to call inject_frb.py for many injections.",
    )
    ap.add_argument("--inject_script", type=Path,
                    default=Path(__file__).resolve().parent / "inject_frb.py",
                    help="Path to inject_frb.py (default: sibling of this script)")
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
    ap.add_argument("--telescope_id", type=int, default=CASM_TELESCOPE_ID)

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
        inject_script=args.inject_script,
        outdir=args.outdir,
        sampler=sampler,
        fch1=args.fch1,
        foff=args.foff,
        nchans=args.nchans,
        tsamp=args.tsamp,
        nsamples=args.nsamples,
        nbits=args.nbits,
        telescope_id=args.telescope_id,
        dry_run=args.dry_run,
        stop_on_error=args.stop_on_error,
        verbose=not args.quiet,
    )
    injector.run()


if __name__ == "__main__":
    main()
