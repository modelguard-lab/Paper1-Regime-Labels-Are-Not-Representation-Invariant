"""
Public re-export surface for the visualization package.

The plotting code is split across three sibling modules; this file
preserves the historical ``from src.visualization.plots import ...``
import path so callers do not need to change.

Source modules:
  representation_failure  representation-failure-matrix figure (signature plot)
  disagreement            cross-resolution disagreement timeseries + heatmaps
  summary                 box / bar / line summaries; ARI-vs-step; ordering;
                          ARI-gap distributions; model-split grouped bars
"""

from __future__ import annotations

from .representation_failure import (
    _color_for_level,
    _compute_state_risk_ranks,
    _draw_state_band_no_legend,
    plot_representation_failure_matrix,
)
from .disagreement import (
    plot_disagreement_timeseries,
    plot_pairwise_matrix_heatmap,
    plot_stability_heatmap,
)
from .summary import (
    plot_ari_gap_distribution_from_key_results,
    plot_ari_vs_step,
    plot_box_by_group,
    plot_cross_rep_box_by_rep,
    plot_line_by_group,
    plot_model_split_grouped_bar_from_key_results,
    plot_ordering_consistency_summary,
    plot_synth_ari_vs_step_by_model,
)

__all__ = [
    "_color_for_level",
    "_compute_state_risk_ranks",
    "_draw_state_band_no_legend",
    "plot_ari_gap_distribution_from_key_results",
    "plot_ari_vs_step",
    "plot_box_by_group",
    "plot_cross_rep_box_by_rep",
    "plot_disagreement_timeseries",
    "plot_line_by_group",
    "plot_model_split_grouped_bar_from_key_results",
    "plot_ordering_consistency_summary",
    "plot_pairwise_matrix_heatmap",
    "plot_representation_failure_matrix",
    "plot_stability_heatmap",
    "plot_synth_ari_vs_step_by_model",
]
