"""Central command registry for the Paper 1 CLI."""

from __future__ import annotations

import importlib

COMMANDS: dict[str, str] = {
    "pipeline": "src.runner",
    "posthoc_figs": "src.posthoc_figs",
    "posthoc_ami": "src.posthoc_ami_vi_perm",
    "posthoc_synthetic": "src.posthoc_synthetic_groundtruth",
    "posthoc_var_spread": "src.posthoc_var_spread",
    "posthoc_rank_aligned_ordering": "src.posthoc_rank_aligned_ordering",
    "posthoc_convergence_audit": "src.posthoc_convergence_audit",
    "paper_autofill": "src.paper_autofill",
    "aggregate": "src.aggregate",
}


def run_module_command(name: str) -> None:
    mod = importlib.import_module(COMMANDS[name])
    mod.main()
