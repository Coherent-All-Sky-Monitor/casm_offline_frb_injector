#!/usr/bin/env python3
"""
Injection recovery analysis plots.

Classes
-------
RecoveryAnalyzer
    Load and query Hella injection recovery summary CSVs.
RecoveryPlotter
    Publication-quality plotter for recall and recovery diagnostics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# -- Style defaults --------------------------------------------------------
_SCATTER_KW = dict(edgecolors="0.3", linewidths=0.4, s=28, zorder=3)
_CMAP = "cividis"
_STATS_BOX = dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="0.65", alpha=0.9)
_FIG_KW = dict(dpi=150, bbox_inches="tight")


def _apply_style(ax: plt.Axes, grid: bool = False) -> None:
    """Minimal axis styling with despined axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(direction="out", top=False, right=False)
    if grid:
        ax.grid(True, alpha=0.12, lw=0.5)
    else:
        ax.grid(False)


def _add_text(ax: plt.Axes, text: str, loc: str = "upper left") -> None:
    """Add a stats annotation without garish background."""
    x, y, va, ha = 0.03, 0.97, "top", "left"
    if "lower" in loc:
        y = 0.03; va = "bottom"
    if "right" in loc:
        x = 0.97; ha = "right"
    ax.text(x, y, text, transform=ax.transAxes, fontsize=8.5,
            va=va, ha=ha, bbox=_STATS_BOX, family="monospace")


def _slim_colorbar(fig, sc, ax, label: str) -> None:
    """Thin colorbar that doesn't distort the axes."""
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.045, aspect=30)
    cbar.set_label(label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)


# ===================================================================
# RecoveryAnalyzer
# ===================================================================
class RecoveryAnalyzer:
    """Load and analyze Hella injection recovery summary CSVs.

    Parameters
    ----------
    summary_csv : str | Path
        Path to a summary CSV produced by ``run_hella.py``.
    """

    _REQUIRED_COLS = {
        "snr_injected", "snr_rec", "dm_true", "dm_rec",
        "detected", "recovered_fraction", "width_ms",
    }

    def __init__(self, summary_csv: str | Path) -> None:
        self._path = Path(summary_csv)
        self.df = pd.read_csv(self._path)
        missing = self._REQUIRED_COLS - set(self.df.columns)
        if missing:
            raise ValueError(
                f"Summary CSV missing columns: {missing}. "
                f"Available: {list(self.df.columns)}"
            )

    @classmethod
    def from_multiple(cls, *csvs: str | Path) -> RecoveryAnalyzer:
        """Merge multiple summary CSVs."""
        frames = [pd.read_csv(p) for p in csvs]
        merged = pd.concat(frames, ignore_index=True)
        obj = cls.__new__(cls)
        obj._path = None
        obj.df = merged
        missing = cls._REQUIRED_COLS - set(merged.columns)
        if missing:
            raise ValueError(
                f"Summary CSV missing columns: {missing}. "
                f"Available: {list(merged.columns)}"
            )
        return obj

    @property
    def detected(self) -> pd.DataFrame:
        return self.df[self.df["detected"] == 1].copy()

    @property
    def missed(self) -> pd.DataFrame:
        return self.df[self.df["detected"] == 0].copy()

    @property
    def detection_rate(self) -> float:
        if len(self.df) == 0:
            return 0.0
        return float(self.df["detected"].mean())

    @property
    def recovery_fraction(self) -> pd.Series:
        return self.detected["recovered_fraction"]

    @property
    def n_injections(self) -> int:
        return len(self.df)

    @property
    def has_width_columns(self) -> bool:
        return {"expected_ibox", "boxcar_width_rec"}.issubset(set(self.df.columns))


# ===================================================================
# RecoveryPlotter
# ===================================================================
class RecoveryPlotter:
    """Publication-quality plotter for injection recovery diagnostics.

    Parameters
    ----------
    analyzer : RecoveryAnalyzer
        Analyzer loaded with summary data.
    output_dir : str | Path
        Directory for output plots.
    width_min : int
        Hella WIDTH_MIN config value (default: 1).
    tsamp_ms : float
        Sampling time in ms for width axis labels.
    """

    def __init__(
        self,
        analyzer: RecoveryAnalyzer,
        output_dir: str | Path,
        width_min: int = 1,
        tsamp_ms: float = 1.048576,
    ) -> None:
        self.analyzer = analyzer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._width_min = width_min
        self._tsamp_ms = tsamp_ms

    def _boxcar_to_samples(self, idx):
        return self._width_min * (2.0 ** np.asarray(idx, dtype=float))

    def _save(self, fig, filename):
        outpath = self.output_dir / filename
        fig.savefig(outpath, **_FIG_KW)
        plt.close(fig)
        print(f"Saved: {outpath}")
        return outpath

    # -- Recall plots (binned detection fraction) -----------------------

    def _plot_recall_binned(self, col, bins, xlabel, filename,
                            figsize=(5.5, 4)):
        """Generic binned recall (completeness) curve."""
        ana = self.analyzer
        fig, ax = plt.subplots(figsize=figsize)

        centers, rates, lo_err, hi_err = [], [], [], []
        for i in range(len(bins) - 1):
            mask = (ana.df[col] >= bins[i]) & (ana.df[col] < bins[i + 1])
            sub = ana.df[mask]
            n = len(sub)
            if n == 0:
                continue
            k = int(sub["detected"].sum())
            rate = k / n
            # Wilson score interval
            z = 1.96
            denom = 1 + z**2 / n
            centre = (rate + z**2 / (2 * n)) / denom
            margin = z * np.sqrt((rate * (1 - rate) + z**2 / (4 * n)) / n) / denom
            centers.append(0.5 * (bins[i] + bins[i + 1]))
            rates.append(rate)
            lo_err.append(max(rate - (centre - margin), 0))
            hi_err.append(max((centre + margin) - rate, 0))

        ax.errorbar(centers, rates, yerr=[lo_err, hi_err],
                     fmt="o-", color="0.2", markersize=5, capsize=3,
                     elinewidth=0.8, linewidth=1.0)
        ax.set_ylim(-0.05, 1.08)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Recall (detection fraction)", fontsize=10)
        ax.axhline(1.0, ls=":", color="0.5", lw=0.6)
        _apply_style(ax, grid=True)

        _add_text(ax, f"N = {ana.n_injections}\n"
                      f"Detected: {len(ana.detected)}/{ana.n_injections} "
                      f"({ana.detection_rate:.0%})",
                  loc="lower right")
        fig.tight_layout()
        return self._save(fig, filename)

    def plot_recall_vs_snr(self, bins=None, **kw):
        """Recall (detection fraction) vs injected SNR."""
        if bins is None:
            bins = [8, 10, 12, 15, 18, 22, 27, 33, 40, 50]
        return self._plot_recall_binned(
            "snr_injected", bins, "Injected SNR", "recall_vs_snr.png", **kw)

    def plot_recall_vs_dm(self, bins=None, **kw):
        """Recall (detection fraction) vs DM."""
        if bins is None:
            bins = np.arange(0, 1100, 100)
        return self._plot_recall_binned(
            "dm_true", bins, r"Injected DM (pc cm$^{-3}$)",
            "recall_vs_dm.png", **kw)

    def plot_recall_vs_width(self, bins=None, **kw):
        """Recall (detection fraction) vs pulse width."""
        if bins is None:
            bins = [1, 2, 4, 8, 16, 35, 68]
        return self._plot_recall_binned(
            "width_ms", bins, "Injected width (ms)",
            "recall_vs_width.png", **kw)

    # -- Recovery plots (detected only, scatter) ------------------------

    def plot_snr_recovery(self, figsize=(5.5, 5), filename="snr_recovery.png"):
        """Injected vs recovered SNR."""
        det = self.analyzer.detected
        mis = self.analyzer.missed
        fig, ax = plt.subplots(figsize=figsize)

        snr_all = self.analyzer.df["snr_injected"]
        lo, hi = snr_all.min() * 0.85, snr_all.max() * 1.1
        ax.plot([lo, hi], [lo, hi], "-", color="0.6", lw=0.8)

        if len(det) > 0:
            sc = ax.scatter(det["snr_injected"], det["snr_rec"],
                            c=det["dm_true"], cmap=_CMAP, **_SCATTER_KW)
            _slim_colorbar(fig, sc, ax, r"DM (pc cm$^{-3}$)")

        if len(mis) > 0:
            ax.scatter(mis["snr_injected"], [0] * len(mis),
                       marker="x", c="0.45", s=20, linewidths=0.8,
                       zorder=2, label=f"Missed ({len(mis)})")
            ax.legend(fontsize=8, frameon=False, loc="lower right")

        ax.set_xlabel("Injected SNR", fontsize=10)
        ax.set_ylabel("Recovered SNR", fontsize=10)
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, hi)
        _apply_style(ax)

        if len(det) > 0:
            _add_text(ax, f"N detected: {len(det)}/{self.analyzer.n_injections} "
                          f"({len(det)/self.analyzer.n_injections:.0%})",
                      loc="upper left")
        fig.tight_layout()
        return self._save(fig, filename)

    def plot_dm_recovery(self, figsize=(5.5, 5), filename="dm_recovery.png"):
        """Injected vs recovered DM."""
        det = self.analyzer.detected
        fig, ax = plt.subplots(figsize=figsize)

        dm_all = self.analyzer.df["dm_true"]
        lo, hi = dm_all.min() * 0.85, dm_all.max() * 1.1
        ax.plot([lo, hi], [lo, hi], "-", color="0.6", lw=0.8)

        if len(det) > 0:
            sc = ax.scatter(det["dm_true"], det["dm_rec"],
                            c=det["snr_injected"], cmap="magma",
                            **_SCATTER_KW)
            _slim_colorbar(fig, sc, ax, "Injected SNR")

        ax.set_xlabel(r"Injected DM (pc cm$^{-3}$)", fontsize=10)
        ax.set_ylabel(r"Recovered DM (pc cm$^{-3}$)", fontsize=10)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        _apply_style(ax)

        if len(det) > 0 and "dm_diff" in det.columns:
            med = det["dm_diff"].median()
            _add_text(ax, f"Median offset: {med:+.1f} pc/cm³")
        fig.tight_layout()
        return self._save(fig, filename)

    def plot_width_recovery(self, figsize=(5.5, 5),
                            filename="width_recovery.png"):
        """Confusion matrix: expected vs recovered boxcar index."""
        ana = self.analyzer
        if not ana.has_width_columns:
            print("Skipping width_recovery: columns missing")
            return self.output_dir / filename

        det = ana.detected.dropna(subset=["expected_ibox", "boxcar_width_rec"])
        fig, ax = plt.subplots(figsize=figsize)

        tsamp_ms = self._tsamp_ms
        n_box = 7
        labels = [f"{self._width_min * 2**i}\n({self._width_min * 2**i * tsamp_ms:.1f} ms)"
                  for i in range(n_box)]

        if len(det) > 0:
            # Build confusion matrix
            matrix = np.zeros((n_box, n_box), dtype=int)
            for _, row in det.iterrows():
                e = int(row["expected_ibox"])
                r = int(row["boxcar_width_rec"])
                if 0 <= e < n_box and 0 <= r < n_box:
                    matrix[e, r] += 1

            im = ax.imshow(matrix.T, cmap="Greys", origin="lower",
                           aspect="equal", interpolation="nearest")
            _slim_colorbar(fig, im, ax, "Count")

            # Annotate cells (x=expected, y=recovered)
            for e in range(n_box):
                for r in range(n_box):
                    v = matrix[e, r]
                    if v > 0:
                        color = "white" if v > matrix.max() * 0.6 else "0.2"
                        ax.text(e, r, str(v), ha="center", va="center",
                                fontsize=7.5, color=color)

            # Diagonal reference
            ax.plot([-0.5, n_box - 0.5], [-0.5, n_box - 0.5],
                    "-", color="0.5", lw=0.6, alpha=0.5)

            n_match = int(np.trace(matrix))
            n_total = int(matrix.sum())
            pct = f"{n_match/n_total:.0%}" if n_total > 0 else "N/A"
            _add_text(ax, f"Exact: {n_match}/{n_total} ({pct})",
                      loc="upper left")

        ax.set_xticks(range(n_box))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_yticks(range(n_box))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Expected width\n"r"$\mathrm{sm} = \mathrm{pow2}(w_{\mathrm{eff}}\, /\, 0.677)$,  "r"$w_{\mathrm{eff}} = \sqrt{w_{\mathrm{intr}}^2 + \tau_{\mathrm{DM}}^2}$", fontsize=8)
        ax.set_ylabel("Recovered width", fontsize=10)
        # Don't despine imshow plots
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.4)
        ax.tick_params(direction="out", top=False, right=False)
        fig.tight_layout()
        return self._save(fig, filename)

    def plot_recovery_fraction_vs_dm(self, figsize=(5.5, 4),
                                     filename="recovery_fraction_vs_dm.png"):
        """SNR recovery fraction vs DM (detected only)."""
        det = self.analyzer.detected
        fig, ax = plt.subplots(figsize=figsize)
        if len(det) > 0:
            sc = ax.scatter(det["dm_true"], det["recovered_fraction"],
                            c=det["width_ms"], cmap="RdYlBu_r",
                            **_SCATTER_KW)
            _slim_colorbar(fig, sc, ax, "Width (ms)")
        ax.axhline(1.0, ls=":", color="0.5", lw=0.6)
        ax.set_xlabel(r"Injected DM (pc cm$^{-3}$)", fontsize=10)
        ax.set_ylabel("SNR recovery fraction", fontsize=10)
        ax.set_ylim(0, 1.05)
        _apply_style(ax)
        fig.tight_layout()
        return self._save(fig, filename)

    def plot_recovery_fraction_vs_width(self, figsize=(5.5, 4),
                                        filename="recovery_fraction_vs_width.png"):
        """SNR recovery fraction vs pulse width (detected only)."""
        det = self.analyzer.detected
        fig, ax = plt.subplots(figsize=figsize)
        if len(det) > 0:
            sc = ax.scatter(det["width_ms"], det["recovered_fraction"],
                            c=det["dm_true"], cmap=_CMAP, **_SCATTER_KW)
            _slim_colorbar(fig, sc, ax, r"DM (pc cm$^{-3}$)")
        ax.axhline(1.0, ls=":", color="0.5", lw=0.6)
        ax.set_xlabel("Injected width (ms)", fontsize=10)
        ax.set_ylabel("SNR recovery fraction", fontsize=10)
        ax.set_ylim(0, 1.05)
        _apply_style(ax)
        fig.tight_layout()
        return self._save(fig, filename)

    def plot_dm_offset_vs_dm(self, figsize=(5.5, 4),
                             filename="dm_offset_vs_dm.png"):
        """DM bias trend."""
        det = self.analyzer.detected
        if "dm_diff" not in det.columns:
            return self.output_dir / filename

        fig, ax = plt.subplots(figsize=figsize)
        if len(det) > 0:
            sc = ax.scatter(det["dm_true"], det["dm_diff"],
                            c=det["snr_injected"], cmap="magma",
                            **_SCATTER_KW)
            _slim_colorbar(fig, sc, ax, "Injected SNR")

            # Predicted offset from TIME_RESOLUTION bug
            dm_range = np.linspace(det["dm_true"].min(), det["dm_true"].max(), 100)
            predicted = dm_range * (1.0 / 1.048576 - 1.0)
            ax.plot(dm_range, predicted, "--", color="0.4", lw=1.0,
                    label=r"Predicted ($\Delta t$ bug)")
            ax.legend(fontsize=8, frameon=False)

            med = det["dm_diff"].median()
            _add_text(ax, f"Median: {med:+.1f}", loc="lower left")

        ax.axhline(0, ls=":", color="0.5", lw=0.6)
        ax.set_xlabel(r"Injected DM (pc cm$^{-3}$)", fontsize=10)
        ax.set_ylabel(r"DM offset (rec $-$ true) (pc cm$^{-3}$)", fontsize=10)
        _apply_style(ax)
        fig.tight_layout()
        return self._save(fig, filename)

    def plot_recovery_fraction_vs_snr(self, figsize=(5.5, 4),
                                      filename="recovery_fraction_vs_snr.png"):
        """SNR recovery fraction vs injected SNR (detected only)."""
        det = self.analyzer.detected
        fig, ax = plt.subplots(figsize=figsize)
        if len(det) > 0:
            sc = ax.scatter(det["snr_injected"], det["recovered_fraction"],
                            c=det["dm_true"], cmap=_CMAP, **_SCATTER_KW)
            _slim_colorbar(fig, sc, ax, r"DM (pc cm$^{-3}$)")
        ax.axhline(1.0, ls=":", color="0.5", lw=0.6)
        ax.set_xlabel("Injected SNR", fontsize=10)
        ax.set_ylabel("SNR recovery fraction", fontsize=10)
        ax.set_ylim(0, 1.05)
        _apply_style(ax)
        fig.tight_layout()
        return self._save(fig, filename)

    # -- Dashboard ------------------------------------------------------

    def plot_dashboard(self, figsize=(16, 10),
                       filename="recovery_dashboard.png"):
        """3x3 dashboard: recall (top), recovery (middle), diagnostics (bottom)."""
        ana = self.analyzer
        det = ana.detected
        mis = ana.missed

        fig, axes = plt.subplots(3, 3, figsize=figsize)
        for ax in axes.flat:
            _apply_style(ax)

        small_scatter = dict(edgecolors="0.3", linewidths=0.25, s=14, zorder=3)
        tsamp_ms = self._tsamp_ms

        # -- Row 0: Recall curves --
        for ax, col, bins, xlabel in [
            (axes[0, 0], "snr_injected",
             [8, 10, 12, 15, 18, 22, 27, 33, 40, 50], "Injected SNR"),
            (axes[0, 1], "dm_true",
             list(np.arange(0, 1100, 100)), r"DM (pc cm$^{-3}$)"),
            (axes[0, 2], "width_ms",
             [1, 2, 4, 8, 16, 35, 68], "Width (ms)"),
        ]:
            centers, rates = [], []
            for i in range(len(bins) - 1):
                mask = (ana.df[col] >= bins[i]) & (ana.df[col] < bins[i + 1])
                sub = ana.df[mask]
                if len(sub) > 0:
                    centers.append(0.5 * (bins[i] + bins[i + 1]))
                    rates.append(sub["detected"].mean())
            ax.plot(centers, rates, "o-", color="0.2", markersize=3.5, lw=0.9)
            ax.set_ylim(-0.05, 1.08)
            ax.axhline(1.0, ls=":", color="0.5", lw=0.4)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel("Recall", fontsize=8)

        # -- Row 1 col 0: SNR recovery --
        ax = axes[1, 0]
        if len(det) > 0:
            snr_all = ana.df["snr_injected"]
            lo, hi = snr_all.min() * 0.85, snr_all.max() * 1.1
            ax.plot([lo, hi], [lo, hi], "-", color="0.6", lw=0.6)
            sc = ax.scatter(det["snr_injected"], det["snr_rec"],
                            c=det["dm_true"], cmap=_CMAP, **small_scatter)
            _slim_colorbar(fig, sc, ax, "DM")
            ax.set_xlim(lo, hi); ax.set_ylim(0, hi)
        if len(mis) > 0:
            ax.scatter(mis["snr_injected"], [0]*len(mis),
                       marker="x", c="0.5", s=8, linewidths=0.5, zorder=2)
        ax.set_xlabel("Injected SNR", fontsize=8)
        ax.set_ylabel("Recovered SNR", fontsize=8)

        # -- Row 1 col 1: DM recovery --
        ax = axes[1, 1]
        if len(det) > 0:
            dm_all = ana.df["dm_true"]
            lo, hi = dm_all.min() * 0.85, dm_all.max() * 1.1
            ax.plot([lo, hi], [lo, hi], "-", color="0.6", lw=0.6)
            sc = ax.scatter(det["dm_true"], det["dm_rec"],
                            c=det["snr_injected"], cmap="magma", **small_scatter)
            _slim_colorbar(fig, sc, ax, "SNR")
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Injected DM", fontsize=8)
        ax.set_ylabel("Recovered DM", fontsize=8)

        # -- Row 1 col 2: Width confusion matrix --
        ax = axes[1, 2]
        n_box = 7
        if ana.has_width_columns and len(det) > 0:
            dw = det.dropna(subset=["expected_ibox", "boxcar_width_rec"])
            matrix = np.zeros((n_box, n_box), dtype=int)
            for _, row in dw.iterrows():
                e, r = int(row["expected_ibox"]), int(row["boxcar_width_rec"])
                if 0 <= e < n_box and 0 <= r < n_box:
                    matrix[e, r] += 1
            im = ax.imshow(matrix.T, cmap="Greys", origin="lower",
                           aspect="equal", interpolation="nearest")
            for e in range(n_box):
                for r in range(n_box):
                    v = matrix[e, r]
                    if v > 0:
                        clr = "white" if v > matrix.max() * 0.6 else "0.2"
                        ax.text(e, r, str(v), ha="center", va="center",
                                fontsize=6, color=clr)
            ax.plot([-0.5, n_box-0.5], [-0.5, n_box-0.5], "-", color="0.5", lw=0.4, alpha=0.5)
            # Re-enable all spines for imshow
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.4)
        tick_labels = [str(self._width_min * 2**i) for i in range(n_box)]
        ax.set_xticks(range(n_box)); ax.set_xticklabels(tick_labels, fontsize=6)
        ax.set_yticks(range(n_box)); ax.set_yticklabels(tick_labels, fontsize=6)
        ax.set_xlabel("Expected width (samp)", fontsize=8)
        ax.set_ylabel("Recovered width (samp)", fontsize=8)

        # -- Row 2 col 0: Recovery fraction vs DM --
        ax = axes[2, 0]
        if len(det) > 0:
            sc = ax.scatter(det["dm_true"], det["recovered_fraction"],
                            c=det["width_ms"], cmap="RdYlBu_r", **small_scatter)
            _slim_colorbar(fig, sc, ax, "Width (ms)")
        ax.axhline(1.0, ls=":", color="0.5", lw=0.4)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(r"DM (pc cm$^{-3}$)", fontsize=8)
        ax.set_ylabel("Recovery fraction", fontsize=8)

        # -- Row 2 col 1: Recovery fraction vs width --
        ax = axes[2, 1]
        if len(det) > 0:
            sc = ax.scatter(det["width_ms"], det["recovered_fraction"],
                            c=det["dm_true"], cmap=_CMAP, **small_scatter)
            _slim_colorbar(fig, sc, ax, "DM")
        ax.axhline(1.0, ls=":", color="0.5", lw=0.4)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Width (ms)", fontsize=8)
        ax.set_ylabel("Recovery fraction", fontsize=8)

        # -- Row 2 col 2: DM offset vs DM --
        ax = axes[2, 2]
        if len(det) > 0 and "dm_diff" in det.columns:
            sc = ax.scatter(det["dm_true"], det["dm_diff"],
                            c=det["snr_injected"], cmap="magma", **small_scatter)
            _slim_colorbar(fig, sc, ax, "SNR")
            dm_r = np.linspace(det["dm_true"].min(), det["dm_true"].max(), 100)
            ax.plot(dm_r, dm_r * (1.0/1.048576 - 1.0), "--", color="0.4", lw=0.7)
        ax.axhline(0, ls=":", color="0.5", lw=0.4)
        ax.set_xlabel(r"DM (pc cm$^{-3}$)", fontsize=8)
        ax.set_ylabel("DM offset", fontsize=8)

        med_rec = det["recovered_fraction"].median() if len(det) > 0 else 0
        fig.suptitle(
            f"N={ana.n_injections}  |  "
            f"Detected: {len(det)}/{ana.n_injections} ({ana.detection_rate:.0%})  |  "
            f"Median recovery: {med_rec:.0%}",
            fontsize=10, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        return self._save(fig, filename)


    # -- Generate all ---------------------------------------------------

    def plot_all(self) -> list[Path]:
        """Generate all diagnostic plots."""
        paths = [
            self.plot_recall_vs_snr(),
            self.plot_recall_vs_dm(),
            self.plot_recall_vs_width(),
            self.plot_snr_recovery(),
            self.plot_dm_recovery(),
            self.plot_width_recovery(),
            self.plot_recovery_fraction_vs_dm(),
            self.plot_recovery_fraction_vs_width(),
            self.plot_recovery_fraction_vs_snr(),
            self.plot_dm_offset_vs_dm(),
            self.plot_dashboard(),
        ]
        return [p for p in paths if p.exists()]


# ===================================================================
# CLI
# ===================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot injection recovery diagnostics.",
    )
    ap.add_argument("--summary", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--plots", default="all",
                    help="all, or comma-separated: recall_snr, recall_dm, "
                         "recall_width, snr, dm, width, frac_dm, frac_width, "
                         "frac_snr, dm_offset, dashboard")

    args = ap.parse_args()
    analyzer = RecoveryAnalyzer(args.summary)
    plotter = RecoveryPlotter(analyzer, args.outdir)

    plot_map = {
        "recall_snr": plotter.plot_recall_vs_snr,
        "recall_dm": plotter.plot_recall_vs_dm,
        "recall_width": plotter.plot_recall_vs_width,
        "snr": plotter.plot_snr_recovery,
        "dm": plotter.plot_dm_recovery,
        "width": plotter.plot_width_recovery,
        "frac_dm": plotter.plot_recovery_fraction_vs_dm,
        "frac_width": plotter.plot_recovery_fraction_vs_width,
        "frac_snr": plotter.plot_recovery_fraction_vs_snr,
        "dm_offset": plotter.plot_dm_offset_vs_dm,
        "dashboard": plotter.plot_dashboard,
    }

    requested = [p.strip() for p in args.plots.split(",")]
    if "all" in requested:
        plotter.plot_all()
    else:
        for name in requested:
            if name in plot_map:
                plot_map[name]()
            else:
                print(f"Unknown plot: {name!r}. Options: {list(plot_map.keys())}")


if __name__ == "__main__":
    main()
