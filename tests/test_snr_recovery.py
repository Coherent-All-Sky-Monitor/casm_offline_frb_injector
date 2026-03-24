#!/usr/bin/env python3
"""
SNR recovery validation for the de-digitized FRB injector.

Tests that the injected SNR matches the target SNR on uint8 data
across a range of DMs, pulse widths, and SNR values. Also compares
against the old injector to quantify the improvement.

Can be run standalone or via pytest:
    python test_snr_recovery.py
    pytest test_snr_recovery.py -v
"""

from __future__ import annotations

import tempfile
import os

import numpy as np
import pytest

from casm_io.filterbank import FilterbankFile, write_filterbank
from casm_offline_frb_injector.inject_frb import (
    FRBInjector,
    _dedisperse,
    _matched_filter_snr,
)
from casm_offline_frb_injector.inject_frb_dedigitized import (
    inject_synthetic,
    inject_into_file,
)


# ===================================================================
# Helpers
# ===================================================================

def measure_uint8_snr(data, header, dm, optimal_fwhm):
    """Dedisperse uint8 data and measure matched-filter SNR."""
    ts = _dedisperse(
        data.astype(np.float32),
        dm, header["fch1"], header["foff"], header["tsamp"],
    )
    snr, _, _ = _matched_filter_snr(ts, optimal_fwhm)
    return snr


def write_and_readback_snr(data, header, dm, optimal_fwhm, nbits=8):
    """Write to .fil, read back, measure SNR — full round-trip."""
    with tempfile.NamedTemporaryFile(suffix=".fil", delete=False) as f:
        fpath = f.name
    try:
        write_filterbank(fpath, data, header, nbits=nbits)
        fb = FilterbankFile(fpath, verbose=False)
        return measure_uint8_snr(fb.data, header, dm, optimal_fwhm)
    finally:
        os.unlink(fpath)


# ===================================================================
# Pytest parametrised tests
# ===================================================================

SNR_TARGETS = [10, 20, 50, 100]
DM_VALUES = [30, 100, 300]
# nsamples must be large enough for the dispersion sweep + pulse window
DM_NSAMPLES = {30: 4096, 100: 8192, 300: 16384}
FWHM_VALUES = [2, 5, 16]


class TestDedigitizedSNRRecovery:
    """Verify that inject_synthetic achieves the target SNR on uint8 data."""

    @pytest.mark.parametrize("target_snr", SNR_TARGETS)
    def test_snr_sweep(self, target_snr):
        """SNR recovery within 15% for various target SNR values."""
        data, meta, hdr = inject_synthetic(
            dm=100, target_snr=target_snr, fwhm_samples=5, verbose=False,
        )
        snr = write_and_readback_snr(data, hdr, 100, meta["optimal_fwhm"])
        ratio = snr / target_snr
        assert 0.85 <= ratio <= 1.15, (
            f"target={target_snr}, measured={snr:.1f}, ratio={ratio:.3f}"
        )

    @pytest.mark.parametrize("dm", DM_VALUES)
    def test_dm_sweep(self, dm):
        """SNR recovery within 15% across DM range."""
        target_snr = 30
        data, meta, hdr = inject_synthetic(
            dm=dm, target_snr=target_snr, fwhm_samples=5,
            nsamples=DM_NSAMPLES[dm], verbose=False,
        )
        snr = write_and_readback_snr(data, hdr, dm, meta["optimal_fwhm"])
        ratio = snr / target_snr
        assert 0.85 <= ratio <= 1.15, (
            f"dm={dm}, measured={snr:.1f}, ratio={ratio:.3f}"
        )

    @pytest.mark.parametrize("fwhm", FWHM_VALUES)
    def test_fwhm_sweep(self, fwhm):
        """SNR recovery within 15% across pulse widths."""
        target_snr = 30
        data, meta, hdr = inject_synthetic(
            dm=100, target_snr=target_snr, fwhm_samples=fwhm, verbose=False,
        )
        snr = write_and_readback_snr(data, hdr, 100, meta["optimal_fwhm"])
        ratio = snr / target_snr
        assert 0.85 <= ratio <= 1.15, (
            f"fwhm={fwhm}, measured={snr:.1f}, ratio={ratio:.3f}"
        )

    def test_multibeam(self):
        """Multi-beam injection: beam 0 has signal, beam 1 is noise."""
        target_snr = 30
        data, meta, hdr = inject_synthetic(
            dm=100, target_snr=target_snr, fwhm_samples=5,
            nbeams=4, ibeam=0, verbose=False,
        )
        with tempfile.NamedTemporaryFile(suffix=".fil", delete=False) as f:
            fpath = f.name
        try:
            write_filterbank(fpath, data, hdr, nbits=8)
            # Beam 0 should have signal
            fb0 = FilterbankFile(fpath, beam=0, verbose=False)
            snr0 = measure_uint8_snr(fb0.data, hdr, 100, meta["optimal_fwhm"])
            assert snr0 / target_snr > 0.85, f"beam 0 snr={snr0:.1f}"

            # Beam 1 should be noise
            fb1 = FilterbankFile(fpath, beam=1, verbose=False)
            snr1 = measure_uint8_snr(fb1.data, hdr, 100, meta["optimal_fwhm"])
            assert snr1 < 7, f"beam 1 should be noise, got snr={snr1:.1f}"
        finally:
            os.unlink(fpath)

    def test_inject_into_existing(self):
        """Inject into an existing filterbank preserves target SNR."""
        # Create a noise-only filterbank
        target_snr = 25
        noise_data, _, noise_hdr = inject_synthetic(
            dm=100, target_snr=10, fwhm_samples=5, verbose=False,
        )
        with tempfile.NamedTemporaryFile(suffix=".fil", delete=False) as f:
            noise_path = f.name
        try:
            write_filterbank(noise_path, noise_data, noise_hdr, nbits=8)

            data, meta, hdr = inject_into_file(
                noise_path, dm=50, target_snr=target_snr,
                fwhm_samples=3, verbose=False,
            )
            snr = measure_uint8_snr(data, hdr, 50, meta["optimal_fwhm"])
            ratio = snr / target_snr
            assert 0.80 <= ratio <= 1.20, (
                f"inject_into_file: target={target_snr}, "
                f"measured={snr:.1f}, ratio={ratio:.3f}"
            )
        finally:
            os.unlink(noise_path)


class TestOldVsNewComparison:
    """Compare old injector vs de-digitized injector SNR recovery."""

    @pytest.mark.parametrize("target_snr", [20, 50, 100])
    def test_new_beats_old(self, target_snr):
        """De-digitized injector should be closer to target than old."""
        # Old injector
        inj = FRBInjector(
            dm=100, target_snr=target_snr, fwhm_samples=5, verbose=False,
        )
        inj.inject()
        old_snr = write_and_readback_snr(
            inj.result["data"], inj.header, 100, inj.result["optimal_fwhm"],
        )

        # New injector
        data, meta, hdr = inject_synthetic(
            dm=100, target_snr=target_snr, fwhm_samples=5, verbose=False,
        )
        new_snr = write_and_readback_snr(
            data, hdr, 100, meta["optimal_fwhm"],
        )

        old_err = abs(old_snr / target_snr - 1.0)
        new_err = abs(new_snr / target_snr - 1.0)
        assert new_err < old_err, (
            f"target={target_snr}: old_snr={old_snr:.1f} (err={old_err:.2f}), "
            f"new_snr={new_snr:.1f} (err={new_err:.2f})"
        )


# ===================================================================
# Standalone report (when run as script)
# ===================================================================

def run_full_report():
    """Print a comprehensive SNR recovery report."""
    print("=" * 72)
    print("SNR Recovery Validation: De-digitized FRB Injector")
    print("=" * 72)

    # --- SNR sweep ---
    print("\n--- SNR sweep (DM=100, FWHM=5) ---")
    print(f"{'target':>8s} {'old_uint8':>10s} {'new_uint8':>10s} "
          f"{'old_ratio':>10s} {'new_ratio':>10s}")
    for target in SNR_TARGETS:
        # Old
        inj = FRBInjector(
            dm=100, target_snr=target, fwhm_samples=5, verbose=False,
        )
        inj.inject()
        old_snr = write_and_readback_snr(
            inj.result["data"], inj.header, 100, inj.result["optimal_fwhm"],
        )

        # New
        data, meta, hdr = inject_synthetic(
            dm=100, target_snr=target, fwhm_samples=5, verbose=False,
        )
        new_snr = write_and_readback_snr(data, hdr, 100, meta["optimal_fwhm"])

        print(f"{target:8d} {old_snr:10.1f} {new_snr:10.1f} "
              f"{old_snr / target:10.3f} {new_snr / target:10.3f}")

    # --- DM sweep ---
    print("\n--- DM sweep (SNR=30, FWHM=5) ---")
    print(f"{'DM':>8s} {'nsamples':>10s} {'new_uint8':>10s} {'ratio':>10s}")
    for dm in DM_VALUES:
        nsamp = DM_NSAMPLES[dm]
        data, meta, hdr = inject_synthetic(
            dm=dm, target_snr=30, fwhm_samples=5,
            nsamples=nsamp, verbose=False,
        )
        snr = write_and_readback_snr(data, hdr, dm, meta["optimal_fwhm"])
        print(f"{dm:8d} {nsamp:10d} {snr:10.1f} {snr / 30:10.3f}")

    # --- FWHM sweep ---
    print("\n--- FWHM sweep (SNR=30, DM=100) ---")
    print(f"{'FWHM':>8s} {'new_uint8':>10s} {'ratio':>10s}")
    for fwhm in FWHM_VALUES:
        data, meta, hdr = inject_synthetic(
            dm=100, target_snr=30, fwhm_samples=fwhm, verbose=False,
        )
        snr = write_and_readback_snr(data, hdr, 100, meta["optimal_fwhm"])
        print(f"{fwhm:8d} {snr:10.1f} {snr / 30:10.3f}")

    # --- Multi-beam ---
    print("\n--- Multi-beam (SNR=30, DM=100, FWHM=5, 64 beams) ---")
    data, meta, hdr = inject_synthetic(
        dm=100, target_snr=30, fwhm_samples=5,
        nbeams=64, ibeam=0, verbose=False,
    )
    with tempfile.NamedTemporaryFile(suffix=".fil", delete=False) as f:
        fpath = f.name
    write_filterbank(fpath, data, hdr, nbits=8)
    for b in [0, 1, 32, 63]:
        fb = FilterbankFile(fpath, beam=b, verbose=False)
        snr = measure_uint8_snr(fb.data, hdr, 100, meta["optimal_fwhm"])
        label = "FRB" if b == 0 else "noise"
        print(f"  beam {b:2d} ({label:>5s}): SNR = {snr:.1f}")
    os.unlink(fpath)

    print("\n" + "=" * 72)
    print("DONE")


if __name__ == "__main__":
    run_full_report()
