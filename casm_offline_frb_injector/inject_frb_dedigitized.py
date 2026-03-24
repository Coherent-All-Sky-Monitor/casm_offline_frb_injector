#!/usr/bin/env python3
"""
FRB injection using probabilistic de-digitization.

Injects a dispersed Gaussian pulse into uint8 filterbank data with
accurate SNR preservation. Works in two modes:

  1. Synthetic: generate uint8 Gaussian noise, then inject.
  2. Existing file: read a real filterbank, inject into it.

Both modes share the same core pipeline:
  uint8 -> de-digitize -> add signal -> re-quantize -> uint8

The de-digitization step replaces each uint8 value with a continuous
sample from a truncated normal distribution within its quantization
cell [v-0.5, v+0.5). This ensures the signal is added in continuous
space, avoiding quantization SNR loss.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import truncnorm

from casm_io.filterbank import FilterbankFile, write_filterbank

# Reuse pulse generation and SNR calibration from existing injector
from casm_offline_frb_injector.inject_frb import (
    GaussianPulse,
    SNRCalibrator,
    _dedisperse,
    _matched_filter_snr,
    _quantize,
    K_DM,
    CASM_FCH1,
    CASM_FOFF,
    CASM_NCHANS,
    CASM_TSAMP,
    CASM_NSAMPLES,
    CASM_NBITS,
    CASM_TELESCOPE_ID,
)


# ===================================================================
# De-digitization helpers
# ===================================================================

def compute_filterbank_stats(data: np.ndarray) -> tuple[float, float]:
    """Compute global noise statistics from uint8 filterbank data.

    Parameters
    ----------
    data : numpy.ndarray
        Filterbank data, shape ``(nsamples, nchans)``, dtype uint8.

    Returns
    -------
    mean : float
        Median of per-channel means.
    std : float
        Median of per-channel standard deviations.
    """
    chan_mean = np.mean(data.astype(np.float64), axis=0)
    chan_std = np.std(data.astype(np.float64), axis=0)
    mask = chan_std > 0
    return float(np.median(chan_mean[mask])), float(np.median(chan_std[mask]))


def de_digitize(
    data: np.ndarray,
    mean: float,
    std: float,
    seed: int | None = None,
) -> np.ndarray:
    """Replace each uint8 value with a continuous sample from its
    quantization cell.

    For each unique value ``v`` in the data, samples are drawn from
    ``TruncatedNormal(mean, std, v - 0.5, v + 0.5)``.

    Parameters
    ----------
    data : numpy.ndarray
        uint8 filterbank data.
    mean : float
        Noise distribution mean (in uint8 counts).
    std : float
        Noise distribution std (in uint8 counts).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Float32 de-digitized data, same shape as input.
    """
    rs = np.random.RandomState(seed)
    result = np.empty(data.shape, dtype=np.float32)
    for val in range(int(data.min()), int(data.max()) + 1):
        mask = data == val
        n = int(np.sum(mask))
        if n == 0:
            continue
        lo = (val - 0.5 - mean) / std
        hi = (val + 0.5 - mean) / std
        samples = truncnorm.rvs(a=lo, b=hi, loc=mean, scale=std, size=n,
                                random_state=rs)
        result[mask] = samples.astype(np.float32)
    return result


def re_quantize(data: np.ndarray, nbits: int = 8) -> np.ndarray:
    """Round and clip continuous data back to integer type.

    Parameters
    ----------
    data : numpy.ndarray
        Continuous-valued data.
    nbits : int
        Bit depth (8 or 16).

    Returns
    -------
    numpy.ndarray
        Quantized data (uint8 or uint16).
    """
    max_val = (1 << nbits) - 1
    dtype = np.uint8 if nbits == 8 else np.uint16
    return np.clip(np.round(data), 0, max_val).astype(dtype)


# ===================================================================
# Injection core
# ===================================================================

def inject_frb(
    data_uint8: np.ndarray,
    fb_mean: float,
    fb_std: float,
    pulse: GaussianPulse,
    target_snr: float,
    fwhm_samples: float,
    nbits: int = 8,
    seed: int | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, dict]:
    """Inject a dispersed FRB into uint8 filterbank data.

    Parameters
    ----------
    data_uint8 : numpy.ndarray
        Input uint8 data, shape ``(nsamples, nchans)``.
    fb_mean : float
        Noise mean in uint8 counts.
    fb_std : float
        Noise std in uint8 counts.
    pulse : GaussianPulse
        Pre-built pulse object.
    target_snr : float
        Target matched-filter SNR after dedispersion.
    fwhm_samples : float
        Intrinsic pulse FWHM in samples.
    nbits : int
        Output bit depth.
    seed : int or None
        Random seed.
    verbose : bool
        Print diagnostics.

    Returns
    -------
    injected : numpy.ndarray
        uint8 data with FRB injected.
    metadata : dict
        Injection metadata including measured SNR.
    """
    # SNR calibration: compute initial scale factor, then iterate
    # to correct for quantization effects.
    dedispersed_signal = pulse.dedisperse()
    calibrator = SNRCalibrator(dedispersed_signal, fwhm_samples, data_uint8.shape[1])
    noise_std_float = calibrator.noise_std_for_snr(target_snr)
    scale_factor = fb_std / noise_std_float

    dispersed = pulse.dispersed_data
    analog = de_digitize(data_uint8, fb_mean, fb_std, seed=seed)

    # Iterative correction: adjust scale until measured SNR ≈ target
    measured_snr = 0.0
    for iteration in range(5):
        signal_scaled = dispersed * scale_factor
        injected = re_quantize(analog + signal_scaled, nbits)

        dd = _dedisperse(
            injected.astype(np.float32),
            pulse._dm, pulse._fch1, pulse._foff, pulse._tsamp,
        )
        measured_snr = calibrator.measure_snr(dd)

        if verbose:
            print(f"  Iteration {iteration}: scale={scale_factor:.2f}, "
                  f"measured S/N={measured_snr:.1f}")

        if abs(measured_snr / target_snr - 1.0) < 0.05:
            break

        # SNR scales linearly with signal amplitude
        if measured_snr > 0:
            scale_factor *= target_snr / measured_snr

    metadata = {
        "dm": pulse._dm,
        "fwhm_samples": fwhm_samples,
        "target_snr": target_snr,
        "measured_snr": float(measured_snr),
        "optimal_fwhm": calibrator.optimal_fwhm,
        "pulse_center": pulse.pulse_center,
        "sweep_samples": pulse.sweep_samples,
        "noise_std_float": float(noise_std_float),
        "fb_mean": fb_mean,
        "fb_std": fb_std,
    }
    return injected, metadata


# ===================================================================
# High-level modes
# ===================================================================

def inject_synthetic(
    dm: float,
    target_snr: float,
    fwhm_samples: float,
    fch1: float = CASM_FCH1,
    foff: float = CASM_FOFF,
    nchans: int = CASM_NCHANS,
    tsamp: float = CASM_TSAMP,
    nsamples: int = CASM_NSAMPLES,
    nbits: int = CASM_NBITS,
    nbeams: int = 1,
    ibeam: int = 0,
    seed: int | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, dict, dict]:
    """Generate a synthetic filterbank with an injected FRB.

    Returns
    -------
    data : numpy.ndarray
        Injected data. Shape ``(nsamples, nchans)`` for single-beam,
        ``(nbeams, nsamples, nchans)`` for multi-beam.
    metadata : dict
        Injection metadata.
    header : dict
        SIGPROC header dict.
    """
    rng = np.random.default_rng(seed)

    pulse = GaussianPulse(
        fwhm_samples=fwhm_samples, dm=dm,
        fch1=fch1, foff=foff, nchans=nchans,
        tsamp=tsamp, nsamples=nsamples,
    )

    if pulse.sweep_samples >= nsamples:
        raise ValueError(
            f"Dispersion sweep ({pulse.sweep_samples} samples) exceeds "
            f"nsamples ({nsamples}). Increase --nsamples or lower the DM."
        )

    # SNR calibration to get the float32 noise_std
    dedispersed_signal = pulse.dedisperse()
    calibrator = SNRCalibrator(dedispersed_signal, fwhm_samples, nchans)
    noise_std = calibrator.noise_std_for_snr(target_snr)

    if verbose:
        freqs = fch1 + np.arange(nchans) * foff
        print(f"Creating filterbank via probabilistic de-digitization")
        print(f"  DM: {dm} pc/cm^3")
        print(f"  Intrinsic FWHM: {fwhm_samples} samples "
              f"({fwhm_samples * tsamp * 1000:.3f} ms)")
        print(f"  Target S/N: {target_snr}")
        print(f"  Dispersion sweep: {pulse.sweep_samples} samples "
              f"({pulse.sweep_samples * tsamp * 1000:.1f} ms)")
        print(f"  Frequency range: {np.min(freqs):.3f} - {np.max(freqs):.3f} MHz")
        print(f"  Channels: {nchans}, Samples: {nsamples}")
        print(f"  Pulse center (high freq): sample {pulse.pulse_center}")
        print(f"  Matched filter peak: {calibrator.matched_filter_peak:.2f} "
              f"(optimal FWHM: {calibrator.optimal_fwhm:.1f} samples)")
        print(f"  Noise std per channel (float32): {noise_std:.6f}")
        if nbeams > 1:
            print(f"  Multi-beam mode: {nbeams} beams, FRB in beam {ibeam}")

    # Generate uint8 noise for signal beam
    noise_float = rng.normal(0, noise_std, (nsamples, nchans)).astype(np.float32)
    noise_uint8 = _quantize(noise_float, nbits)

    # Inject FRB into the signal beam
    fb_mean, fb_std = compute_filterbank_stats(noise_uint8)
    signal_beam, metadata = inject_frb(
        noise_uint8, fb_mean, fb_std, pulse, target_snr, fwhm_samples,
        nbits=nbits, seed=seed, verbose=verbose,
    )

    # Build multi-beam output
    if nbeams > 1:
        beams = []
        for b in range(nbeams):
            if b == ibeam:
                beams.append(signal_beam)
            else:
                beam_noise_float = rng.normal(
                    0, noise_std, (nsamples, nchans),
                ).astype(np.float32)
                beams.append(_quantize(beam_noise_float, nbits))
        data = np.stack(beams, axis=0)
    else:
        data = signal_beam

    header = {
        "source_name": "FRB_injection",
        "telescope_id": CASM_TELESCOPE_ID,
        "data_type": 1,
        "fch1": fch1,
        "foff": foff,
        "nchans": nchans,
        "nbits": nbits,
        "tsamp": tsamp,
        "tstart": 59000.0,
        "nifs": 1,
        "nsamples": nsamples,
        "nbeams": nbeams,
        "ibeam": ibeam,
    }

    return data, metadata, header


def inject_into_file(
    input_path: str,
    dm: float,
    target_snr: float,
    fwhm_samples: float,
    beam: int | None = None,
    nbits: int = CASM_NBITS,
    seed: int | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, dict, dict]:
    """Inject an FRB into an existing filterbank file.

    Returns
    -------
    data : numpy.ndarray
        Injected data, shape ``(nsamples, nchans)``.
    metadata : dict
        Injection metadata.
    header : dict
        Header from the input file.
    """
    fb = FilterbankFile(input_path, beam=beam, verbose=verbose)

    pulse = GaussianPulse(
        fwhm_samples=fwhm_samples, dm=dm,
        fch1=fb.header["fch1"], foff=fb.header["foff"],
        nchans=fb.nchans, tsamp=fb.header["tsamp"],
        nsamples=fb.nsamples,
    )

    if pulse.sweep_samples >= fb.nsamples:
        raise ValueError(
            f"Dispersion sweep ({pulse.sweep_samples} samples) exceeds "
            f"nsamples ({fb.nsamples})."
        )

    if verbose:
        freqs = fb.freq_mhz
        print(f"Injecting FRB into {input_path}")
        print(f"  DM: {dm} pc/cm^3")
        print(f"  Target S/N: {target_snr}")
        print(f"  FWHM: {fwhm_samples} samples "
              f"({fwhm_samples * fb.header['tsamp'] * 1000:.3f} ms)")

    fb_mean, fb_std = compute_filterbank_stats(fb.data)
    if verbose:
        print(f"  Filterbank stats: mean={fb_mean:.2f}, std={fb_std:.2f}")

    injected, metadata = inject_frb(
        fb.data, fb_mean, fb_std, pulse, target_snr, fwhm_samples,
        nbits=nbits, seed=seed, verbose=verbose,
    )

    # Strip internal/sigpyproc-only keys from header for standalone writer
    skip_keys = {
        "az_start", "za_start", "rawdatafile", "source", "telescope",
        "backend", "filename", "basename", "data_type",
    }
    header = {
        k: v for k, v in fb.header.items()
        if not k.startswith("_") and k not in skip_keys
    }
    header.setdefault("data_type", 1)
    return injected, metadata, header


# ===================================================================
# CLI
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inject FRB into filterbank using probabilistic de-digitization",
    )
    parser.add_argument("--output", "-o", required=True,
                        help="Output filterbank file")
    parser.add_argument("--dm", type=float, required=True,
                        help="Dispersion measure in pc/cm^3")
    parser.add_argument("--fwhm", type=float, required=True,
                        help="Intrinsic pulse FWHM in samples")
    parser.add_argument("--snr", "-s", type=float, required=True,
                        help="Target matched-filter S/N")
    parser.add_argument("--input", "-i", default=None,
                        help="Input filterbank to inject into (omit for synthetic)")
    parser.add_argument("--beam", type=int, default=None,
                        help="Beam to read from input file (multi-beam)")
    parser.add_argument("--fch1", type=float, default=CASM_FCH1,
                        help=f"Top frequency MHz (default: {CASM_FCH1})")
    parser.add_argument("--foff", type=float, default=CASM_FOFF,
                        help=f"Channel width MHz (default: {CASM_FOFF})")
    parser.add_argument("--nchans", type=int, default=CASM_NCHANS,
                        help=f"Number of channels (default: {CASM_NCHANS})")
    parser.add_argument("--tsamp", type=float, default=CASM_TSAMP,
                        help=f"Sampling time seconds (default: {CASM_TSAMP})")
    parser.add_argument("--nsamples", type=int, default=CASM_NSAMPLES,
                        help=f"Time samples (default: {CASM_NSAMPLES})")
    parser.add_argument("--nbits", type=int, default=CASM_NBITS,
                        help=f"Bits per sample (default: {CASM_NBITS})")
    parser.add_argument("--nbeams", type=int, default=1,
                        help="Number of beams (synthetic mode, default: 1)")
    parser.add_argument("--ibeam", type=int, default=0,
                        help="Beam index for FRB (synthetic mode, default: 0)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress output")

    args = parser.parse_args()
    verbose = not args.quiet

    if args.input is not None:
        # Inject into existing file
        data, metadata, header = inject_into_file(
            input_path=args.input, dm=args.dm, target_snr=args.snr,
            fwhm_samples=args.fwhm, beam=args.beam, nbits=args.nbits,
            seed=args.seed, verbose=verbose,
        )
    else:
        # Synthetic mode
        data, metadata, header = inject_synthetic(
            dm=args.dm, target_snr=args.snr, fwhm_samples=args.fwhm,
            fch1=args.fch1, foff=args.foff, nchans=args.nchans,
            tsamp=args.tsamp, nsamples=args.nsamples, nbits=args.nbits,
            nbeams=args.nbeams, ibeam=args.ibeam,
            seed=args.seed, verbose=verbose,
        )

    write_filterbank(
        filepath=args.output, data=data, header=header, nbits=args.nbits,
    )
    if verbose:
        print(f"  Written to {args.output}")


if __name__ == "__main__":
    main()
