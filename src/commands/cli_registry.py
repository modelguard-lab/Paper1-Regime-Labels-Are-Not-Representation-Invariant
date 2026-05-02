"""Central command registry for the Paper 1 CLI."""

from __future__ import annotations

import importlib

COMMANDS: dict[str, str] = {
    "pipeline": "src.workflows.pipeline",
    "posthoc_figs": "src.experiments.posthoc_figs",
    "posthoc_ami": "src.experiments.posthoc_ami_vi_perm",
    "posthoc_synthetic": "src.experiments.posthoc_synthetic_groundtruth",
    "posthoc_synthetic_psweep": "src.experiments.posthoc_synthetic_psweep",
    "posthoc_var_spread": "src.experiments.posthoc_var_spread",
    "posthoc_rank_aligned_ordering": "src.experiments.posthoc_rank_aligned_ordering",
    "posthoc_convergence_audit": "src.experiments.posthoc_convergence_audit",
    "posthoc_kmeans_robustness": "src.experiments.posthoc_kmeans_robustness",
    "posthoc_repr_decomp": "src.experiments.posthoc_repr_decomp",
    "posthoc_stationarity_null": "src.experiments.posthoc_stationarity_null",
    "paper_autofill": "src.workflows.paper_autofill",
    "aggregate": "src.workflows.aggregate",
}


def run_module_command(name: str) -> None:
    mod = importlib.import_module(COMMANDS[name])
    mod.main()
