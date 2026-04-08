#!/usr/bin/env python3
"""
Tests for casm_offline_frb_injector pipeline components.

Covers:
  - CandidateMatcher (run_hella.py)
  - ExpectedBoxcar (run_hella.py)
  - RecoveryAnalyzer (plot_recovery.py)
  - RecoveryPlotter (plot_recovery.py)
  - InjectionParameterSampler (batch_inject_frbs.py)
  - _run_one_injection (batch_inject_frbs.py)
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from casm_offline_frb_injector.run_hella import CandidateMatcher, ExpectedBoxcar
from casm_offline_frb_injector.plot_recovery import RecoveryAnalyzer, RecoveryPlotter
from casm_offline_frb_injector.batch_inject_frbs import (
    InjectionParameterSampler,
    _run_one_injection,
)
from casm_offline_frb_injector.inject_frb import CASM_TSAMP


# ===================================================================
# Fixtures
# ===================================================================

def _make_cand_df(**extra_cols):
    """Return a minimal candidate DataFrame with required columns."""
    base = {
        "snr": [7.0, 12.0, 5.5],
        "dm": [99.0, 101.5, 200.0],
        "sample_index": [512, 512, 512],
        "time_start": [0.537, 0.537, 0.537],
        "boxcar_width": [1, 2, 1],
        "dm_index": [10, 11, 20],
        "beam_index": [0, 0, 0],
    }
    base.update(extra_cols)
    return pd.DataFrame(base)


def _make_summary_csv(rows: list[dict], tmp_path: Path) -> Path:
    """Write rows to a temp CSV and return the path."""
    df = pd.DataFrame(rows)
    p = tmp_path / "summary.csv"
    df.to_csv(p, index=False)
    return p


def _minimal_summary_rows(n_detected: int = 2, n_missed: int = 1) -> list[dict]:
    """Build a minimal valid summary for RecoveryAnalyzer."""
    rows = []
    for i in range(n_detected):
        rows.append({
            "snr_injected": 20.0 + i,
            "snr_rec": 18.0 + i,
            "dm_true": 100.0 + i * 50,
            "dm_rec": 101.0 + i * 50,
            "detected": 1,
            "recovered_fraction": 0.9,
            "width_ms": 2.0,
            "dm_diff": 1.0,
            "expected_ibox": i,
            "boxcar_width_rec": i,
        })
    for j in range(n_missed):
        rows.append({
            "snr_injected": 8.0,
            "snr_rec": 0.0,
            "dm_true": 300.0,
            "dm_rec": float("nan"),
            "detected": 0,
            "recovered_fraction": float("nan"),
            "width_ms": 5.0,
            "dm_diff": float("nan"),
            "expected_ibox": float("nan"),
            "boxcar_width_rec": float("nan"),
        })
    return rows


# ===================================================================
# CandidateMatcher.effective_window
# ===================================================================

class TestCandidateMatcherEffectiveWindow:
    def setup_method(self):
        self.m = CandidateMatcher(dm_window=15.0, dm_window_frac=0.06)

    def test_low_dm_uses_floor(self):
        # 0.06 * 100 = 6.0, floor is 15.0
        assert self.m.effective_window(100.0) == 15.0

    def test_medium_dm_fractional_dominates(self):
        # 0.06 * 300 = 18.0 > 15.0
        assert self.m.effective_window(300.0) == 18.0

    def test_high_dm_fractional_dominates(self):
        # 0.06 * 1000 = 60.0
        assert self.m.effective_window(1000.0) == 60.0

    def test_zero_dm_returns_floor(self):
        assert self.m.effective_window(0.0) == 15.0

    def test_negative_dm_uses_abs(self):
        # abs(-300) * 0.06 = 18.0 > 15.0
        assert self.m.effective_window(-300.0) == 18.0

    def test_boundary_exactly_at_floor(self):
        # 0.06 * 250 = 15.0, should return exactly 15.0
        assert self.m.effective_window(250.0) == 15.0

    def test_custom_floor(self):
        m = CandidateMatcher(dm_window=5.0, dm_window_frac=0.01)
        # 0.01 * 100 = 1.0 < 5.0 → floor
        assert m.effective_window(100.0) == 5.0
        # 0.01 * 1000 = 10.0 > 5.0 → fractional
        assert m.effective_window(1000.0) == 10.0


# ===================================================================
# CandidateMatcher.match
# ===================================================================

class TestCandidateMatcherMatch:
    def setup_method(self):
        self.m = CandidateMatcher(dm_window=15.0, dm_window_frac=0.06)

    def test_empty_dataframe_returns_no_detection(self):
        result = self.m.match(pd.DataFrame(), dm_true=100.0)
        assert result["detected"] == 0
        assert result["n_matches"] == 0
        assert result["best"] is None

    def test_candidate_within_window_is_detected(self):
        df = _make_cand_df()
        result = self.m.match(df, dm_true=100.0)
        assert result["detected"] == 1

    def test_all_candidates_outside_window_is_missed(self):
        df = _make_cand_df(dm=[300.0, 350.0, 400.0])
        result = self.m.match(df, dm_true=100.0)
        assert result["detected"] == 0
        assert result["n_matches"] == 0
        assert result["best"] is None

    def test_best_candidate_is_highest_snr_in_window(self):
        # dm=99 (in window) with snr=7, dm=101.5 (in window) with snr=12
        # dm=200 (outside window of 15 from 100) with snr=5.5
        df = _make_cand_df()
        result = self.m.match(df, dm_true=100.0)
        assert result["best"]["snr"] == pytest.approx(12.0)
        assert result["n_matches"] == 2

    def test_n_matches_counts_all_within_window(self):
        df = pd.DataFrame({
            "snr": [7.0, 8.0, 9.0, 30.0],
            "dm": [90.0, 95.0, 105.0, 500.0],
            "sample_index": [100] * 4,
            "time_start": [0.1] * 4,
            "boxcar_width": [1] * 4,
            "dm_index": [0, 1, 2, 10],
            "beam_index": [0] * 4,
        })
        result = self.m.match(df, dm_true=100.0)
        assert result["n_matches"] == 3

    def test_single_candidate_at_exact_dm_is_detected(self):
        df = pd.DataFrame({
            "snr": [10.0],
            "dm": [100.0],
            "sample_index": [100],
            "time_start": [0.1],
            "boxcar_width": [1],
            "dm_index": [0],
            "beam_index": [0],
        })
        result = self.m.match(df, dm_true=100.0)
        assert result["detected"] == 1
        assert result["best"]["snr"] == pytest.approx(10.0)

    def test_candidate_at_window_boundary_is_included(self):
        df = pd.DataFrame({
            "snr": [10.0],
            "dm": [115.0],  # exactly at window edge (100 + 15)
            "sample_index": [100],
            "time_start": [0.1],
            "boxcar_width": [1],
            "dm_index": [0],
            "beam_index": [0],
        })
        result = self.m.match(df, dm_true=100.0)
        assert result["detected"] == 1

    def test_candidate_just_outside_window_is_missed(self):
        df = pd.DataFrame({
            "snr": [10.0],
            "dm": [115.001],  # just beyond window edge
            "sample_index": [100],
            "time_start": [0.1],
            "boxcar_width": [1],
            "dm_index": [0],
            "beam_index": [0],
        })
        result = self.m.match(df, dm_true=100.0)
        assert result["detected"] == 0


# ===================================================================
# ExpectedBoxcar.compute
# ===================================================================

class TestExpectedBoxcarCompute:
    def setup_method(self):
        self.eb = ExpectedBoxcar()

    def test_returns_required_keys(self):
        result = self.eb.compute(width_ms=1.0, dm=100.0)
        assert set(result.keys()) == {"ibox", "w_intr_samples", "tau_dm_samples", "w_eff_samples"}

    def test_narrow_pulse_low_dm_gives_ibox_zero(self):
        # w_intr ~ 0.95, tau_dm small → w_eff clamped to 1.0 → ibox=0
        result = self.eb.compute(width_ms=1.0, dm=50.0)
        assert result["ibox"] == 0

    def test_wide_pulse_gives_higher_ibox(self):
        # width=10ms at DM=100 → w_intr~9.5 → nearest pow2 is 8 → ibox=3
        result = self.eb.compute(width_ms=10.0, dm=100.0)
        assert result["ibox"] == 3

    def test_w_eff_always_at_least_one(self):
        # Even for extremely narrow pulse and low DM, w_eff is clamped to 1.0
        result = self.eb.compute(width_ms=0.001, dm=10.0)
        assert result["w_eff_samples"] >= 1.0

    def test_w_eff_at_least_one_across_range(self):
        cases = [
            (0.001, 0.1),
            (0.01, 1.0),
            (1.0, 50.0),
            (5.0, 500.0),
            (20.0, 2000.0),
        ]
        for w_ms, dm in cases:
            result = self.eb.compute(w_ms, dm)
            assert result["w_eff_samples"] >= 1.0, (
                f"w_eff < 1 for width_ms={w_ms}, dm={dm}: {result}"
            )

    def test_intrinsic_width_scales_with_width_ms(self):
        r1 = self.eb.compute(width_ms=2.0, dm=10.0)
        r2 = self.eb.compute(width_ms=4.0, dm=10.0)
        # Doubling width_ms should double w_intr_samples
        assert r2["w_intr_samples"] == pytest.approx(2 * r1["w_intr_samples"])

    def test_tau_dm_scales_with_dm(self):
        r1 = self.eb.compute(width_ms=1.0, dm=100.0)
        r2 = self.eb.compute(width_ms=1.0, dm=200.0)
        # tau_dm is proportional to DM
        assert r2["tau_dm_samples"] == pytest.approx(2 * r1["tau_dm_samples"])

    def test_high_dm_increases_ibox(self):
        r_low = self.eb.compute(width_ms=1.0, dm=50.0)
        r_high = self.eb.compute(width_ms=1.0, dm=500.0)
        assert r_high["ibox"] >= r_low["ibox"]

    def test_ibox_is_non_negative(self):
        result = self.eb.compute(width_ms=1.0, dm=100.0)
        assert result["ibox"] >= 0

    def test_custom_tsamp_changes_w_intr(self):
        tsamp_fast = 0.0005
        tsamp_slow = 0.002
        eb_fast = ExpectedBoxcar(tsamp=tsamp_fast)
        eb_slow = ExpectedBoxcar(tsamp=tsamp_slow)
        r_fast = eb_fast.compute(width_ms=1.0, dm=10.0)
        r_slow = eb_slow.compute(width_ms=1.0, dm=10.0)
        # Faster sampling → more samples for same physical width
        assert r_fast["w_intr_samples"] > r_slow["w_intr_samples"]


# ===================================================================
# RecoveryAnalyzer
# ===================================================================

class TestRecoveryAnalyzerLoad:
    def test_valid_csv_loads_without_error(self, tmp_path):
        p = _make_summary_csv(_minimal_summary_rows(), tmp_path)
        ana = RecoveryAnalyzer(p)
        assert ana.n_injections == 3

    def test_missing_columns_raises_value_error(self, tmp_path):
        rows = [{"snr_injected": 10, "dm_true": 100}]
        p = _make_summary_csv(rows, tmp_path)
        with pytest.raises(ValueError, match="missing columns"):
            RecoveryAnalyzer(p)

    def test_missing_single_required_column_raises(self, tmp_path):
        rows = _minimal_summary_rows()
        df = pd.DataFrame(rows).drop(columns=["recovered_fraction"])
        p = tmp_path / "s.csv"
        df.to_csv(p, index=False)
        with pytest.raises(ValueError):
            RecoveryAnalyzer(p)


class TestRecoveryAnalyzerProperties:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        rows = _minimal_summary_rows(n_detected=2, n_missed=1)
        p = Path(self._tmpdir) / "summary.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        self.ana = RecoveryAnalyzer(p)

    def teardown_method(self):
        shutil.rmtree(self._tmpdir)

    def test_detected_and_missed_partition_all_rows(self):
        total = self.ana.n_injections
        assert len(self.ana.detected) + len(self.ana.missed) == total

    def test_detected_all_have_detected_flag_one(self):
        assert (self.ana.detected["detected"] == 1).all()

    def test_missed_all_have_detected_flag_zero(self):
        assert (self.ana.missed["detected"] == 0).all()

    def test_detection_rate_matches_counts(self):
        expected = len(self.ana.detected) / self.ana.n_injections
        assert self.ana.detection_rate == pytest.approx(expected)

    def test_detection_rate_all_detected(self):
        tmpdir = tempfile.mkdtemp()
        try:
            rows = _minimal_summary_rows(n_detected=4, n_missed=0)
            p = Path(tmpdir) / "s.csv"
            pd.DataFrame(rows).to_csv(p, index=False)
            ana = RecoveryAnalyzer(p)
            assert ana.detection_rate == pytest.approx(1.0)
        finally:
            shutil.rmtree(tmpdir)

    def test_detection_rate_none_detected(self):
        tmpdir = tempfile.mkdtemp()
        try:
            rows = _minimal_summary_rows(n_detected=0, n_missed=3)
            p = Path(tmpdir) / "s.csv"
            pd.DataFrame(rows).to_csv(p, index=False)
            ana = RecoveryAnalyzer(p)
            assert ana.detection_rate == pytest.approx(0.0)
        finally:
            shutil.rmtree(tmpdir)

    def test_has_width_columns_true_when_present(self):
        assert self.ana.has_width_columns is True

    def test_has_width_columns_false_when_missing(self):
        tmpdir = tempfile.mkdtemp()
        try:
            rows = _minimal_summary_rows()
            df = pd.DataFrame(rows).drop(columns=["expected_ibox"])
            p = Path(tmpdir) / "s.csv"
            df.to_csv(p, index=False)
            ana = RecoveryAnalyzer(p)
            assert ana.has_width_columns is False
        finally:
            shutil.rmtree(tmpdir)

    def test_n_injections_counts_all_rows(self):
        assert self.ana.n_injections == 3


# ===================================================================
# RecoveryPlotter
# ===================================================================

class TestRecoveryPlotter:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        rows = _minimal_summary_rows(n_detected=3, n_missed=2)
        csv_path = Path(self._tmpdir) / "summary.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        self.ana = RecoveryAnalyzer(csv_path)
        self.outdir = Path(self._tmpdir) / "plots"
        self.plotter = RecoveryPlotter(self.ana, self.outdir)

    def teardown_method(self):
        shutil.rmtree(self._tmpdir)

    def test_plot_all_returns_paths_that_exist(self):
        paths = self.plotter.plot_all()
        assert len(paths) > 0
        for p in paths:
            assert p.exists(), f"Expected plot file missing: {p}"

    def test_plot_all_generates_known_filenames(self):
        paths = self.plotter.plot_all()
        names = {p.name for p in paths}
        expected = {
            "recall_vs_snr.png",
            "recall_vs_dm.png",
            "recall_vs_width.png",
            "snr_recovery.png",
            "dm_recovery.png",
            "recovery_dashboard.png",
        }
        assert expected.issubset(names), (
            f"Missing plots: {expected - names}"
        )

    def test_plot_recall_vs_snr_returns_path(self):
        p = self.plotter.plot_recall_vs_snr()
        assert isinstance(p, Path)
        assert p.exists()

    def test_plot_snr_recovery_returns_path(self):
        p = self.plotter.plot_snr_recovery()
        assert isinstance(p, Path)
        assert p.exists()

    def test_plot_dm_recovery_returns_path(self):
        p = self.plotter.plot_dm_recovery()
        assert isinstance(p, Path)
        assert p.exists()

    def test_plot_width_recovery_returns_path(self):
        p = self.plotter.plot_width_recovery()
        assert isinstance(p, Path)

    def test_plot_dashboard_returns_path(self):
        p = self.plotter.plot_dashboard()
        assert isinstance(p, Path)
        assert p.exists()

    def test_output_files_are_nonempty(self):
        paths = self.plotter.plot_all()
        for p in paths:
            assert p.stat().st_size > 0, f"Zero-byte plot: {p}"

    def test_all_missing_detections_does_not_raise(self):
        # Plotter must be robust when every injection was missed
        tmpdir = tempfile.mkdtemp()
        try:
            rows = _minimal_summary_rows(n_detected=0, n_missed=3)
            csv_p = Path(tmpdir) / "s.csv"
            pd.DataFrame(rows).to_csv(csv_p, index=False)
            ana = RecoveryAnalyzer(csv_p)
            plotter = RecoveryPlotter(ana, Path(tmpdir) / "plots")
            paths = plotter.plot_all()
            assert len(paths) > 0
        finally:
            shutil.rmtree(tmpdir)


# ===================================================================
# InjectionParameterSampler
# ===================================================================

class TestInjectionParameterSampler:
    def test_same_seed_produces_identical_parameters(self):
        s1 = InjectionParameterSampler(n=10, rng_seed=42)
        s2 = InjectionParameterSampler(n=10, rng_seed=42)
        for p1, p2 in zip(s1.parameters, s2.parameters):
            assert p1["dm"] == p2["dm"]
            assert p1["snr"] == p2["snr"]
            assert p1["seed"] == p2["seed"]

    def test_different_seeds_produce_different_parameters(self):
        s1 = InjectionParameterSampler(n=10, rng_seed=1)
        s2 = InjectionParameterSampler(n=10, rng_seed=2)
        dms1 = [p["dm"] for p in s1.parameters]
        dms2 = [p["dm"] for p in s2.parameters]
        assert dms1 != dms2

    def test_parameter_count_matches_n(self):
        n = 17
        s = InjectionParameterSampler(n=n, rng_seed=0)
        assert len(s.parameters) == n

    def test_snr_within_specified_range(self):
        lo, hi = 10.0, 25.0
        s = InjectionParameterSampler(n=100, rng_seed=7, snr_range=(lo, hi))
        snrs = [p["snr"] for p in s.parameters]
        assert all(lo <= v <= hi for v in snrs), (
            f"SNR out of range: min={min(snrs):.3f}, max={max(snrs):.3f}"
        )

    def test_dm_within_specified_range(self):
        lo, hi = 50.0, 150.0
        s = InjectionParameterSampler(n=100, rng_seed=7, dm_range=(lo, hi))
        dms = [p["dm"] for p in s.parameters]
        assert all(lo <= v <= hi for v in dms)

    def test_width_ms_within_specified_range(self):
        lo, hi = 1.0, 5.0
        s = InjectionParameterSampler(n=100, rng_seed=7, width_ms_range=(lo, hi))
        widths = [p["width_ms"] for p in s.parameters]
        assert all(lo <= v <= hi for v in widths)

    def test_position_within_specified_range(self):
        lo, hi = 0.2, 0.8
        s = InjectionParameterSampler(n=100, rng_seed=7, pos_range=(lo, hi))
        positions = [p["position"] for p in s.parameters]
        assert all(lo <= v <= hi for v in positions)

    def test_required_keys_present(self):
        s = InjectionParameterSampler(n=3, rng_seed=42)
        for p in s.parameters:
            assert set(p.keys()) == {"dm", "snr", "width_ms", "fwhm_samples", "position", "seed"}

    def test_fwhm_samples_positive(self):
        s = InjectionParameterSampler(n=50, rng_seed=99)
        assert all(p["fwhm_samples"] > 0 for p in s.parameters)

    def test_fwhm_samples_consistent_with_width_and_tsamp(self):
        tsamp = 0.002
        s = InjectionParameterSampler(n=20, rng_seed=5, tsamp=tsamp)
        for p in s.parameters:
            expected_fwhm = (p["width_ms"] * 1e-3) / tsamp
            assert p["fwhm_samples"] == pytest.approx(expected_fwhm)

    def test_unknown_snr_distribution_raises_value_error(self):
        s = InjectionParameterSampler(n=5, dist_snr="gaussian")
        with pytest.raises(ValueError, match="Unknown distribution"):
            _ = s.parameters

    def test_unknown_dm_distribution_raises_value_error(self):
        s = InjectionParameterSampler(n=5, dist_dm="powerlaw")
        with pytest.raises(ValueError, match="Unknown distribution"):
            _ = s.parameters

    def test_unknown_width_distribution_raises_value_error(self):
        s = InjectionParameterSampler(n=5, dist_width="exponential")
        with pytest.raises(ValueError, match="Unknown distribution"):
            _ = s.parameters

    def test_loguniform_covers_range_logarithmically(self):
        # With loguniform, geometric mean should be close to sqrt(lo*hi)
        lo, hi = 1.0, 100.0
        s = InjectionParameterSampler(
            n=5000, rng_seed=0,
            width_ms_range=(lo, hi),
            dist_width="loguniform",
        )
        widths = np.array([p["width_ms"] for p in s.parameters])
        log_mean = np.exp(np.mean(np.log(widths)))
        geometric_center = np.sqrt(lo * hi)
        assert abs(log_mean / geometric_center - 1.0) < 0.05

    def test_parameters_property_cached(self):
        # Calling parameters twice returns the same object
        s = InjectionParameterSampler(n=5, rng_seed=1)
        p1 = s.parameters
        p2 = s.parameters
        assert p1 is p2


# ===================================================================
# _run_one_injection
# ===================================================================

class TestRunOneInjection:
    def test_returns_four_values(self):
        result = _run_one_injection(["true"], "/nonexistent/path.fil")
        assert len(result) == 4

    def test_return_types_are_correct(self):
        rc, success, stdout, stderr = _run_one_injection(["true"], "/nonexistent.fil")
        assert isinstance(rc, int)
        assert isinstance(success, bool)
        assert isinstance(stdout, str)
        assert isinstance(stderr, str)

    def test_failed_command_returns_nonzero_rc(self):
        rc, success, stdout, stderr = _run_one_injection(
            ["false"], "/nonexistent.fil"
        )
        assert rc != 0
        assert success is False

    def test_success_requires_file_existence(self, tmp_path):
        # Even with rc=0, if the output file doesn't exist, success=False
        rc, success, stdout, stderr = _run_one_injection(
            ["true"], str(tmp_path / "missing.fil")
        )
        assert rc == 0
        assert success is False

    def test_success_requires_nonempty_file(self, tmp_path):
        empty_fil = tmp_path / "empty.fil"
        empty_fil.touch()
        # Command that succeeds: write nothing to the file (it stays empty)
        rc, success, stdout, stderr = _run_one_injection(["true"], str(empty_fil))
        assert rc == 0
        assert success is False

    def test_success_true_when_file_nonempty(self, tmp_path):
        out_fil = tmp_path / "output.fil"
        # Use shell to create file with content
        cmd = ["bash", "-c", f"echo dummy > {out_fil}"]
        rc, success, stdout, stderr = _run_one_injection(cmd, str(out_fil))
        assert rc == 0
        assert success is True

    def test_invalid_executable_returns_failure(self):
        # _run_one_injection catches OSError for missing binaries and returns
        # a failure tuple instead of raising.
        rc, success, stdout, stderr = _run_one_injection(
            ["/nonexistent_binary_xyz"], "/tmp/out.fil"
        )
        assert rc == -1
        assert success is False
        assert "Failed to execute" in stderr
