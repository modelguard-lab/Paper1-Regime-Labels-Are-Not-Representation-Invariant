import pytest

from src.core.utils import assets_from_cfg, enabled_models_from_cfg, reps_from_cfg


class TestRepsFromCfg:
    def test_returns_all_keys_including_rep_e(self):
        # Regression for the bug this helper exists to prevent: rep_e being
        # silently dropped when callers rely on hardcoded module-level lists.
        cfg = {
            "representations": {
                "rep_a": {}, "rep_a_unscaled": {}, "rep_b": {},
                "rep_c1": {}, "rep_c2": {}, "rep_c3": {},
                "rep_d": {}, "rep_e": {"asset_filter": ["^GSPC"]},
            },
        }
        assert reps_from_cfg(cfg) == [
            "rep_a", "rep_a_unscaled", "rep_b",
            "rep_c1", "rep_c2", "rep_c3",
            "rep_d", "rep_e",
        ]

    def test_rejects_non_dict(self):
        with pytest.raises(TypeError):
            reps_from_cfg(None)
        with pytest.raises(TypeError):
            reps_from_cfg(["rep_a"])

    def test_rejects_empty(self):
        with pytest.raises(KeyError, match="representations"):
            reps_from_cfg({"representations": {}})
        with pytest.raises(KeyError, match="representations"):
            reps_from_cfg({})


class TestEnabledModelsFromCfg:
    def test_both_enabled(self):
        cfg = {"models": {"gmm": {"enabled": True}, "hmm": {"enabled": True}}}
        assert set(enabled_models_from_cfg(cfg)) == {"gmm", "hmm"}

    def test_disabled_filtered_out(self):
        cfg = {"models": {"gmm": {"enabled": False}, "hmm": {"enabled": True}}}
        assert enabled_models_from_cfg(cfg) == ["hmm"]

    def test_default_enabled_when_key_missing(self):
        cfg = {"models": {"gmm": {}, "hmm": {"enabled": True}}}  # gmm.enabled defaults True
        assert set(enabled_models_from_cfg(cfg)) == {"gmm", "hmm"}

    def test_raises_when_all_disabled(self):
        cfg = {"models": {"gmm": {"enabled": False}, "hmm": {"enabled": False}}}
        with pytest.raises(ValueError, match="no enabled"):
            enabled_models_from_cfg(cfg)

    def test_rejects_empty_dict(self):
        with pytest.raises(KeyError, match="models"):
            enabled_models_from_cfg({"models": {}})


class TestAssetsFromCfg:
    def test_returns_list_preserving_order(self):
        cfg = {"assets": ["IEF", "^GSPC", "GLD", "BTC-USD"]}
        assert assets_from_cfg(cfg) == ["IEF", "^GSPC", "GLD", "BTC-USD"]

    def test_rejects_empty_list(self):
        with pytest.raises(KeyError, match="assets"):
            assets_from_cfg({"assets": []})

    def test_rejects_non_list(self):
        with pytest.raises(KeyError, match="assets"):
            assets_from_cfg({"assets": "^GSPC"})
