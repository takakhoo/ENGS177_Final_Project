"""Shared plot styling used across the report figures.

Calling `apply_style()` once at the top of an experiment script makes every
subsequent matplotlib figure use:
  - serif fonts compatible with the report's LaTeX body type
  - 11pt default axis-tick font, 12pt axis labels, 13pt title
  - light grid by default (alpha=0.25, behind data)
  - sensible figure size for letter-page reports (11 inch wide)
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.framealpha": 0.92,
        "legend.fancybox": False,
        "legend.edgecolor": "0.4",
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linestyle": ":",
        "grid.color": "0.5",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "figure.dpi": 100,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "lines.linewidth": 1.7,
    })


# Consistent colour palette across all plots (matches metropolis-beamer)
PALETTE = {
    "static":  "#666666",
    "qmdp":    "#1f77b4",
    "myopic":  "#2ca02c",
    "bear":    "#d62728",
    "bull":    "#1f77b4",
    "neutral": "#ff9f00",
    "nber":    "#cfcfcf",
    "highlight": "#d62728",
    "ann":     "#222222",
}


def annotate_event(ax, x, y, text, dx=20, dy=20, arrow=True):
    """Place a small annotation at (x, y) with a leader line."""
    arrowprops = dict(arrowstyle="->", color=PALETTE["ann"], lw=0.6) if arrow else None
    ax.annotate(
        text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
        fontsize=9, color=PALETTE["ann"], arrowprops=arrowprops,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.4", lw=0.4),
    )
