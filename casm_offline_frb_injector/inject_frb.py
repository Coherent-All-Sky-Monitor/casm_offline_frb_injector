#!/usr/bin/env python3
"""
Core single-injection tool: generate a filterbank file with a dispersed
Gaussian pulse in calibrated white noise.

Uses ``casm_io.filterbank.write_filterbank()`` for output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from casm_io.filterbank import write_filterbank

# ---------------------------------------------------------------------------
# CASM instrument defaults (all overridable via CLI)
# ---------------------------------------------------------------------------
CASM_FCH1: float = 468.75                # MHz, top of band
CASM_FOFF: float = -0.030517578125       # MHz, 125/4096, descending
CASM_NCHANS: int = 3072                  # full resolution
CASM_TSAMP: float = 0.001048576          # s, 32.768 µs × 32
CASM_NSAMPLES: int = 4096               # tobs ≈ 4.29 s
CASM_NBITS: int = 8
CASM_TELESCOPE_ID: int = 20             # OVRO

# Dispersion constant  (MHz² s pc⁻¹ cm³)
K_DM: float = 4.148808e3


# ===================================================================
# GaussianPulse
# ===================================================================
class GaussianPulse:
    """Generate and disperse a Gaussian pulse.

    Parameters
    ----------
    fwhm_samples : float
        Intrinsic full-width-at-half-maximum in time samples.
    dm : float
        Dispersion measure in pc cm⁻³.
    fch1 : float
        Frequency of the first (highest) channel in MHz.
    foff : float
        Channel bandwidth in MHz (negative for descending frequency).
    nchans : int
        Number of frequency channels.
    tsamp : float
        Sampling time in seconds.
    nsamples : int
        Total number of time samples.
    position : float
        Pulse position as a fraction of the *usable* window (0–1).
        The usable window starts after the dispersion sweep so the
        pulse is visible after dedispersion.
    """

    def __init__(
        self,
        fwhm_samples: float,
        dm: float,
        fch1: float = CASM_FCH1,
        foff: float = CASM_FOFF,
        nchans: int = CASM_NCHANS,
        tsamp: float = CASM_TSAMP,
        nsamples: int = CASM_NSAMPLES,
        position: float = 0.5,
    ) -> None:
        self._fwhm_samples = float(fwhm_samples)
        self._dm = float(dm)
        self._fch1 = float(fch1)
        self._foff = float(foff)
        self._nchans = int(nchans)
        self._tsamp = float(tsamp)
        self._nsamples = int(nsamples)
        self._position = float(position)

        self._freqs = self._fch1 + np.arange(self._nchans) * self._foff
        self._freq_hi = float(np.max(self._freqs))
        self._freq_lo = float(np.min(self._freqs))

        self._pulse_1d: np.ndarray | None = None
        self._dispersed: np.ndarray | None = None

    # -- properties ---------------------------------------------------------

    @property
    def sweep_samples(self) -> int:
        """Dispersion sweep across the full band in samples."""
        delay_s = K_DM * self._dm * (self._freq_lo ** -2 - self._freq_hi ** -2)
        return int(delay_s / self._tsamp)

    @property
    def pulse_center(self) -> int:
        """Sample index of the pulse centre at the highest frequency."""
        sweep = self.sweep_samples
        return int(sweep + (self._nsamples - sweep) * self._position)

    @property
    def pulse_1d(self) -> np.ndarray:
        """Normalised 1-D Gaussian profile (peak = 1)."""
        if self._pulse_1d is None:
            sigma = self._fwhm_samples / 2.355
            t = np.arange(self._nsamples)
            self._pulse_1d = np.exp(-0.5 * ((t - self.pulse_center) / sigma) ** 2)
        return self._pulse_1d

    @property
    def dispersed_data(self) -> np.ndarray:
        """Dispersed filterbank data, shape ``(nsamples, nchans)``."""
        if self._dispersed is None:
            self._dispersed = self._disperse()
        return self._dispersed

    # -- public methods -----------------------------------------------------

    def dedisperse(self) -> np.ndarray:
        """Dedisperse the (noiseless) signal and return the 1-D timeseries."""
        return _dedisperse(
            self.dispersed_data,
            self._dm,
            self._fch1,
            self._foff,
            self._tsamp,
        )

    # -- internals ----------------------------------------------------------

    def _disperse(self) -> np.ndarray:
        pulse = self.pulse_1d
        nsamples = len(pulse)
        delays = np.array([
            K_DM * self._dm * (f ** -2 - self._freq_hi ** -2) / self._tsamp
            for f in self._freqs
        ])
        t = np.arange(nsamples, dtype=np.float64)
        data = np.zeros((nsamples, self._nchans), dtype=np.float32)
        for i, delay in enumerate(delays):
            data[:, i] = np.interp(t - delay, t, pulse, left=0, right=0)
        return data


# ===================================================================
# SNRCalibrator
# ===================================================================
class SNRCalibrator:
    """Calibrate noise level to achieve a target matched-filter SNR.

    Parameters
    ----------
    signal_data : numpy.ndarray
        Noiseless dedispersed 1-D timeseries.
    fwhm_samples : float
        Intrinsic FWHM of the injected pulse in samples.
    nchans : int
        Number of frequency channels (used in noise scaling).
    """

    _FWHM_LIST = [1, 2, 4, 8, 16, 32, 64, 128]

    def __init__(
        self,
        signal_data: np.ndarray,
        fwhm_samples: float,
        nchans: int,
    ) -> None:
        self._signal = signal_data
        self._fwhm_samples = float(fwhm_samples)
        self._nchans = int(nchans)

        self._optimal_fwhm: float | None = None
        self._mf_peak: float | None = None
        self._compute_matched_filter()

    # -- properties ---------------------------------------------------------

    @property
    def optimal_fwhm(self) -> float:
        """FWHM (in samples) that maximises the matched-filter response."""
        return self._optimal_fwhm  # type: ignore[return-value]

    @property
    def matched_filter_peak(self) -> float:
        """Peak matched-filter output for unit-noise."""
        return self._mf_peak  # type: ignore[return-value]

    # -- public methods -----------------------------------------------------

    def noise_std_for_snr(self, target_snr: float) -> float:
        """Per-channel noise std that yields *target_snr* after dedispersion.

        Parameters
        ----------
        target_snr : float
            Desired matched-filter SNR.

        Returns
        -------
        float
            Standard deviation of Gaussian noise per sample per channel.
        """
        return self._mf_peak / (target_snr * np.sqrt(self._nchans))

    def measure_snr(self, noisy_timeseries: np.ndarray) -> float:
        """Measure matched-filter SNR of an already-dedispersed timeseries.

        Parameters
        ----------
        noisy_timeseries : numpy.ndarray
            Dedispersed timeseries containing signal + noise.

        Returns
        -------
        float
            Measured matched-filter SNR.
        """
        snr, _, _ = _matched_filter_snr(noisy_timeseries, self.optimal_fwhm)
        return snr

    # -- internals ----------------------------------------------------------

    def _compute_matched_filter(self) -> None:
        peaks: dict[float, float] = {}
        for fwhm in self._FWHM_LIST:
            _, peak, _ = _matched_filter_snr(self._signal, fwhm, noise_std=1.0)
            peaks[fwhm] = peak

        fwhms = np.array(self._FWHM_LIST, dtype=float)
        peak_arr = np.array([peaks[f] for f in fwhms])
        max_idx = int(np.argmax(peak_arr))
        best_fwhm = fwhms[max_idx]
        best_peak = peak_arr[max_idx]

        # parabolic interpolation in log-space
        if 0 < max_idx < len(fwhms) - 1:
            idx_range = [max_idx - 1, max_idx, max_idx + 1]
            log_f = np.log2(fwhms[idx_range])
            sub_peaks = peak_arr[idx_range]
            coeffs = np.polyfit(log_f, sub_peaks, 2)
            if coeffs[0] < 0:
                opt = -coeffs[1] / (2 * coeffs[0])
                if log_f[0] <= opt <= log_f[2]:
                    best_fwhm = 2 ** opt
                    best_peak = float(np.polyval(coeffs, opt))

        self._optimal_fwhm = float(best_fwhm)
        self._mf_peak = float(best_peak)


# ===================================================================
# FRBInjector
# ===================================================================
class FRBInjector:
    """Create a filterbank file with a dispersed Gaussian pulse in white noise.

    Parameters
    ----------
    dm : float
        Dispersion measure in pc cm⁻³.
    fwhm_samples : float
        Intrinsic pulse FWHM in time samples.
    target_snr : float
        Target matched-filter SNR after dedispersion.
    fch1 : float
        Frequency of first channel in MHz.
    foff : float
        Channel bandwidth in MHz (negative for descending).
    nchans : int
        Number of frequency channels.
    tsamp : float
        Sampling time in seconds.
    nsamples : int
        Number of time samples.
    nbits : int
        Bits per output sample (8, 16, or 32).
    telescope_id : int
        SIGPROC telescope identifier.
    source_name : str
        Source name written into the header.
    position : float
        Pulse position as fraction of usable window (0–1).
    seed : int | None
        Random seed for noise generation.
    verbose : bool
        Print diagnostic information.
    """

    def __init__(
        self,
        dm: float,
        fwhm_samples: float,
        target_snr: float,
        fch1: float = CASM_FCH1,
        foff: float = CASM_FOFF,
        nchans: int = CASM_NCHANS,
        tsamp: float = CASM_TSAMP,
        nsamples: int = CASM_NSAMPLES,
        nbits: int = CASM_NBITS,
        telescope_id: int = CASM_TELESCOPE_ID,
        source_name: str = "FRB_injection",
        position: float = 0.5,
        seed: int | None = None,
        verbose: bool = True,
    ) -> None:
        self._dm = float(dm)
        self._fwhm_samples = float(fwhm_samples)
        self._target_snr = float(target_snr)
        self._fch1 = float(fch1)
        self._foff = float(foff)
        self._nchans = int(nchans)
        self._tsamp = float(tsamp)
        self._nsamples = int(nsamples)
        self._nbits = int(nbits)
        self._telescope_id = int(telescope_id)
        self._source_name = str(source_name)
        self._position = float(position)
        self._seed = seed
        self._verbose = verbose

        self._result: dict | None = None

    # -- properties ---------------------------------------------------------

    @property
    def header(self) -> dict:
        """SIGPROC-style header dict for ``write_filterbank``."""
        return {
            "source_name": self._source_name,
            "telescope_id": self._telescope_id,
            "data_type": 1,
            "fch1": self._fch1,
            "foff": self._foff,
            "nchans": self._nchans,
            "nbits": self._nbits,
            "tsamp": self._tsamp,
            "tstart": 59000.0,
            "nifs": 1,
            "nsamples": self._nsamples,
        }

    @property
    def result(self) -> dict | None:
        """Injection metadata populated after :meth:`inject` is called."""
        return self._result

    # -- public methods -----------------------------------------------------

    def inject(self) -> dict:
        """Run the injection pipeline and return metadata.

        Returns
        -------
        dict
            Keys: dm, fwhm_samples, target_snr, measured_snr,
            optimal_fwhm, pulse_center, sweep_samples, noise_std,
            data (the final uint8 array).
        """
        rng = np.random.default_rng(self._seed)

        pulse = GaussianPulse(
            fwhm_samples=self._fwhm_samples,
            dm=self._dm,
            fch1=self._fch1,
            foff=self._foff,
            nchans=self._nchans,
            tsamp=self._tsamp,
            nsamples=self._nsamples,
            position=self._position,
        )

        if pulse.sweep_samples >= self._nsamples:
            raise ValueError(
                f"Dispersion sweep ({pulse.sweep_samples} samples) exceeds "
                f"nsamples ({self._nsamples}). Increase --nsamples to at least "
                f"{pulse.sweep_samples + 100} or lower the DM / bandwidth."
            )

        signal = pulse.dispersed_data
        dedispersed_signal = pulse.dedisperse()

        calibrator = SNRCalibrator(dedispersed_signal, self._fwhm_samples, self._nchans)
        noise_std = calibrator.noise_std_for_snr(self._target_snr)

        if self._verbose:
            freqs = self._fch1 + np.arange(self._nchans) * self._foff
            print(f"Creating Gaussian burst filterbank")
            print(f"  DM: {self._dm} pc/cm^3")
            print(f"  Intrinsic FWHM: {self._fwhm_samples} samples "
                  f"({self._fwhm_samples * self._tsamp * 1000:.3f} ms)")
            print(f"  Target S/N: {self._target_snr}")
            print(f"  Dispersion sweep: {pulse.sweep_samples} samples "
                  f"({pulse.sweep_samples * self._tsamp * 1000:.1f} ms)")
            print(f"  Frequency range: {np.min(freqs):.3f} - {np.max(freqs):.3f} MHz")
            print(f"  Channels: {self._nchans}, Samples: {self._nsamples}")
            print(f"  Pulse center (high freq): sample {pulse.pulse_center}")
            print(f"  Matched filter peak: {calibrator.matched_filter_peak:.2f} "
                  f"(optimal FWHM: {calibrator.optimal_fwhm:.1f} samples)")
            print(f"  Noise std per channel: {noise_std:.6f}")

        # Add noise
        noise = rng.normal(0, noise_std, signal.shape).astype(np.float32)
        noisy_data = signal + noise

        # Verify
        dedispersed_noisy = _dedisperse(
            noisy_data, self._dm, self._fch1, self._foff, self._tsamp,
        )
        measured_snr = calibrator.measure_snr(dedispersed_noisy)

        if self._verbose:
            print(f"  Measured matched-filter S/N: {measured_snr:.1f}")

        # Quantize to nbits
        data_mean = np.mean(noisy_data)
        data_std = np.std(noisy_data)
        if data_std == 0:
            data_std = 1.0
        if self._nbits == 8:
            scaled = (noisy_data - data_mean) / (4 * data_std) * 127 + 128
            scaled = np.clip(scaled, 0, 255).astype(np.uint8)
        elif self._nbits == 16:
            scaled = (noisy_data - data_mean) / (4 * data_std) * 32767 + 32768
            scaled = np.clip(scaled, 0, 65535).astype(np.uint16)
        else:
            scaled = noisy_data

        self._result = {
            "dm": self._dm,
            "fwhm_samples": self._fwhm_samples,
            "target_snr": self._target_snr,
            "measured_snr": float(measured_snr),
            "optimal_fwhm": calibrator.optimal_fwhm,
            "pulse_center": pulse.pulse_center,
            "sweep_samples": pulse.sweep_samples,
            "noise_std": float(noise_std),
            "data": scaled,
        }
        return self._result

    def write(self, output_path: str | Path) -> Path:
        """Run injection (if needed) and write to a filterbank file.

        Parameters
        ----------
        output_path : str or Path
            Destination ``.fil`` path.

        Returns
        -------
        Path
            The written file path.
        """
        if self._result is None:
            self.inject()

        output_path = Path(output_path)
        write_filterbank(
            filepath=str(output_path),
            data=self._result["data"],  # type: ignore[index]
            header=self.header,
            nbits=self._nbits,
        )

        if self._verbose:
            print(f"  Written to {output_path}")

        return output_path


# ===================================================================
# Module-level helpers (shared by classes)
# ===================================================================

def _dedisperse(
    data: np.ndarray,
    dm: float,
    fch1: float,
    foff: float,
    tsamp: float,
) -> np.ndarray:
    """Dedisperse filterbank data to a 1-D timeseries."""
    nsamples, nchans = data.shape
    freqs = fch1 + np.arange(nchans) * foff
    freq_hi = float(np.max(freqs))
    delays = np.array([
        int(K_DM * dm * (f ** -2 - freq_hi ** -2) / tsamp) for f in freqs
    ])
    result = np.zeros(nsamples)
    for i, delay in enumerate(delays):
        if delay < nsamples:
            result[:nsamples - delay] += data[delay:, i]
    return result


def _matched_filter_snr(
    timeseries: np.ndarray,
    fwhm_samples: float,
    noise_std: float | None = None,
) -> tuple[float, float, int]:
    """Matched-filter SNR using a Gaussian template.

    Returns ``(snr, peak_value, peak_idx)``.
    """
    sigma = fwhm_samples / 2.355
    half_width = max(int(4 * sigma), 1)
    x = np.arange(-half_width, half_width + 1)
    template = np.exp(-0.5 * (x / max(sigma, 0.1)) ** 2)
    template /= np.sqrt(np.sum(template ** 2))

    convolved = np.convolve(timeseries, template, mode="same")

    if noise_std is None:
        noise_std = 1.4826 * np.median(np.abs(convolved - np.median(convolved)))
        baseline = np.median(convolved)
    else:
        baseline = 0.0

    peak_idx = int(np.argmax(convolved))
    peak_value = float(convolved[peak_idx] - baseline)
    snr = peak_value / noise_std if noise_std > 0 else float("inf")
    return snr, peak_value, peak_idx


# ===================================================================
# CLI
# ===================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a filterbank file with a dispersed Gaussian pulse",
    )
    parser.add_argument("--output", "-o", required=True,
                        help="Output filterbank file")
    parser.add_argument("--dm", type=float, required=True,
                        help="Dispersion measure in pc/cm^3")
    parser.add_argument("--fwhm", type=float, required=True,
                        help="Intrinsic pulse FWHM in samples")
    parser.add_argument("--snr", "-s", type=float, required=True,
                        help="Target matched-filter S/N")
    parser.add_argument("--fch1", type=float, default=CASM_FCH1,
                        help=f"Top frequency in MHz (default: {CASM_FCH1})")
    parser.add_argument("--foff", type=float, default=CASM_FOFF,
                        help=f"Channel width in MHz (default: {CASM_FOFF})")
    parser.add_argument("--nchans", type=int, default=CASM_NCHANS,
                        help=f"Number of channels (default: {CASM_NCHANS})")
    parser.add_argument("--tsamp", type=float, default=CASM_TSAMP,
                        help=f"Sampling time in seconds (default: {CASM_TSAMP})")
    parser.add_argument("--nsamples", type=int, default=CASM_NSAMPLES,
                        help=f"Number of time samples (default: {CASM_NSAMPLES})")
    parser.add_argument("--nbits", type=int, default=CASM_NBITS,
                        help=f"Bits per sample (default: {CASM_NBITS})")
    parser.add_argument("--telescope_id", type=int, default=CASM_TELESCOPE_ID,
                        help=f"Telescope ID (default: {CASM_TELESCOPE_ID})")
    parser.add_argument("--source_name", default="FRB_injection",
                        help="Source name in header (default: FRB_injection)")
    parser.add_argument("--position", type=float, default=0.5,
                        help="Pulse position as fraction 0-1 (default: 0.5)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress diagnostic output")

    args = parser.parse_args()

    injector = FRBInjector(
        dm=args.dm,
        fwhm_samples=args.fwhm,
        target_snr=args.snr,
        fch1=args.fch1,
        foff=args.foff,
        nchans=args.nchans,
        tsamp=args.tsamp,
        nsamples=args.nsamples,
        nbits=args.nbits,
        telescope_id=args.telescope_id,
        source_name=args.source_name,
        position=args.position,
        seed=args.seed,
        verbose=not args.quiet,
    )
    injector.write(args.output)


if __name__ == "__main__":
    main()
