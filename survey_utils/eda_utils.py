"""Reusable EDA plotting helpers with explicit metadata and output inputs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis_utils import eda_weighting_label, weighted_quantile
from .kobo_metadata import display_category, ordered_response_codes


def safe_filename(text: object) -> str:
    """Return a filesystem-safe filename component."""
    cleaned = re.sub(r"[^\w\s-]", "", str(text))
    return re.sub(r"\s+", "_", cleaned.strip())


def plot_categorical(
    data: pd.DataFrame,
    variable: str,
    list_name_by_var: Mapping[str, str | None],
    choice_maps: Mapping[str, Mapping[str, str]],
    *,
    title: str | None = None,
    output_dir: Path,
    weight_col: str | None = None,
    dpi: int = 300,
) -> Path | None:
    """Save categorical EDA as counts or weighted percentages.

    Returns ``None`` when the requested variable is unavailable or has no
    observable values. Metadata controls category order and display labels.
    """
    if variable not in data.columns:
        return None
    series = data[variable]
    order = ordered_response_codes(variable, series.dropna().unique(), list_name_by_var, choice_maps)
    if not order:
        return None
    counts = series.value_counts(dropna=False).reindex(order, fill_value=0)
    labels = [display_category(value, variable, list_name_by_var, choice_maps) for value in counts.index]
    weighted = weight_col is not None and weight_col in data.columns
    if weighted:
        valid = data.loc[series.notna(), [variable, weight_col]].dropna()
        values = valid.groupby(variable)[weight_col].sum().reindex(order, fill_value=0)
        values = values / values.sum() * 100 if values.sum() else values
        ylabel = "Weighted percentage"
        annotation = lambda n, value: f"n={n}\n({value:.1f}%)"
    else:
        values = counts
        total = counts.sum()
        ylabel = "Frequency"
        annotation = lambda n, value: f"{n}\n({(n / total * 100 if total else 0):.1f}%)"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([str(label) for label in labels], values.values)
    ax.set_title(f"{title if title else variable} — {eda_weighting_label(weight_col)}")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    plt.xticks(rotation=45, ha="right")
    ymax = values.max() if len(values) else 0
    for bar, n, value in zip(bars, counts.values, values.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * 0.01, annotation(n, value), ha="center", fontsize=8)
    plt.tight_layout()
    path = output_dir / f"{safe_filename(variable)}_{safe_filename(title)[:80]}.png"
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return path


def plot_continuous(
    data: pd.DataFrame,
    variable: str,
    *,
    title: str,
    output_dir: Path,
    weight_col: str | None = None,
    bins: int = 20,
    dpi: int = 300,
) -> Path | None:
    """Save a continuous-variable histogram and boxplot.

    Returns ``None`` when no usable numeric observations are available.
    """
    if variable not in data.columns:
        return None
    values = pd.to_numeric(data[variable], errors="coerce")
    weighted = weight_col is not None and weight_col in data.columns
    weights = pd.to_numeric(data[weight_col], errors="coerce") if weighted else None
    valid = values.notna() & (weights.notna() if weighted else True)
    values = values.loc[valid]
    if values.empty:
        return None
    if weighted:
        weights = weights.loc[valid]
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(values, bins=bins, weights=weights if weighted else None)
    axes[0].set_title("Histogram")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Weighted count" if weighted else "Frequency")
    if weighted:
        q1, median, q3 = weighted_quantile(values, [0.25, 0.5, 0.75], weights)
        mean = np.average(values, weights=weights) if weights.sum() else np.nan
        variance = np.average((values - mean) ** 2, weights=weights) if weights.sum() else np.nan
        axes[1].bxp([{"med": median, "q1": q1, "q3": q3, "whislo": values.min(), "whishi": values.max(), "fliers": []}], showfliers=False)
        stats_text = f"n = {len(values)}\nWeighted mean = {mean:.2f}\nWeighted median = {median:.2f}\nWeighted SD = {np.sqrt(variance):.2f}"
    else:
        values.plot.box(ax=axes[1])
        stats_text = f"N = {len(values)}\nMean = {values.mean():.2f}\nMedian = {values.median():.2f}\nSD = {values.std():.2f}"
    axes[0].text(0.98, 0.98, stats_text, transform=axes[0].transAxes, ha="right", va="top", fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    axes[1].set_title("Boxplot")
    axes[1].set_xlabel("")
    plt.suptitle(f"{title} — {eda_weighting_label(weight_col)}")
    plt.tight_layout()
    path = output_dir / f"{safe_filename(variable)}_{safe_filename(title)[:80]}.png"
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return path


def plot_multiple_response(
    frequency_table: pd.DataFrame,
    variable: str,
    *,
    title: str,
    output_dir: Path,
    weight_col: str | None = None,
    dpi: int = 300,
) -> Path | None:
    """Save a multiple-response EDA chart or return ``None`` for empty input."""
    required = {"Option", "Frequency", "Respondent_%"}
    if frequency_table.empty or not required.issubset(frequency_table.columns):
        return None
    weighted = weight_col is not None and "Weighted_Respondent_%" in frequency_table.columns
    values = frequency_table["Weighted_Respondent_%"] if weighted else frequency_table["Frequency"]
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, max(4, len(frequency_table) * 0.45)))
    bars = ax.barh(frequency_table["Option"], values)
    ax.invert_yaxis()
    ax.set_title(f"{title} — {eda_weighting_label(weight_col)}")
    ax.set_xlabel("Weighted respondent percentage" if weighted else "Frequency")
    ax.set_ylabel("")
    xmax = values.max() if len(values) else 0
    for bar, freq, value, pct in zip(bars, frequency_table["Frequency"], values, frequency_table["Respondent_%"]):
        label = f"n={freq} ({value:.1f}%)" if weighted else f"{freq} ({pct:.1f}%)"
        ax.text(value + xmax * 0.01, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=9)
    plt.tight_layout()
    path = output_dir / f"{safe_filename(variable)}_{safe_filename(title)[:80]}.png"
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return path
