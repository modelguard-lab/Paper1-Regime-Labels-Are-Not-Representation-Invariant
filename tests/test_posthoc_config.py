import pytest

from posthoc_ami_vi_perm import _resolve_run_defs


class TestResolveRunDefs:
    def test_rejects_non_dict(self):
        with pytest.raises(TypeError):
            _resolve_run_defs(None)

    def test_rejects_empty_representations(self):
        cfg = {
            "grid": {"step": 21},
            "representations": {},
            "models": {"hmm": {"enabled": True}},
        }
        with pytest.raises(KeyError, match="representations"):
            _resolve_run_defs(cfg)

    def test_rejects_missing_step(self):
        cfg = {
            "grid": {},
            "representations": {"rep_a": {}},
            "models": {"hmm": {"enabled": True}},
        }
        with pytest.raises(KeyError, match="step"):
            _resolve_run_defs(cfg)

    def test_passes_through_all_reps_from_cfg(self):
        # Regression test for the silent-fallback bug: rep_e used to be dropped
        # whenever a caller forgot to pass cfg, because module-level REPS lacked it.
        cfg = {
            "grid": {"step_sweep": [21, 63]},
            "representations": {
                "rep_a": {}, "rep_a_unscaled": {}, "rep_b": {},
                "rep_c1": {}, "rep_c2": {}, "rep_c3": {},
                "rep_d": {}, "rep_e": {},
            },
            "models": {"gmm": {"enabled": True}, "hmm": {"enabled": True}},
        }
        steps, reps, models = _resolve_run_defs(cfg)
        assert steps == [21, 63]
        assert reps == [
            "rep_a", "rep_a_unscaled", "rep_b",
            "rep_c1", "rep_c2", "rep_c3",
            "rep_d", "rep_e",
        ]
        assert set(models) == {"gmm", "hmm"}

    def test_respects_disabled_models(self):
        cfg = {
            "grid": {"step": 21},
            "representations": {"rep_a": {}},
            "models": {"gmm": {"enabled": False}, "hmm": {"enabled": True}},
        }
        _, _, models = _resolve_run_defs(cfg)
        assert models == ["hmm"]
