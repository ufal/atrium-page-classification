"""
tests/test_parallel_best.py
===========================
Unit tests for parallel_best.py (issue #3 — parallel --best engine).

Scope
-----
Only the pure, torch-free machinery is exercised here so the suite stays
hermetic (no GPU, no torch, no model download, no network):

* pack_models       – greedy memory-aware grouping, max_group cap, oversized
                      models, determinism, empty/zero budget
* registry_is_fresh – hardware + batch + coverage invalidation rules
* merge_best        – byte-identical wide combined frame vs the original inline
                      merge that averaging.py's _WIDE_MODEL_RE parses

The GPU paths (profile_best_models, run_best_parallel, _run_group_single_pass)
load model checkpoints and require a CUDA device, so they belong in an
integration environment and are intentionally NOT covered here.

parallel_best imports torch/classifier lazily, so importing it (and these
helpers) needs only pandas.
"""
import pandas as pd

# parallel_best lives in the project root (on sys.path via conftest.py)
from parallel_best import merge_best, pack_models, registry_is_fresh


# ════════════════════════════════════════════════════════════════════════════
# pack_models
# ════════════════════════════════════════════════════════════════════════════
class TestPackModels:
    """Greedy first-fit-decreasing packing into VRAM-budget-bounded groups."""

    def test_all_models_fit_in_single_group(self):
        sizes = {"v1.3": 1, "v2.3": 1, "v3.3": 1}
        groups = pack_models(sizes, budget_bytes=10)
        assert len(groups) == 1
        assert sorted(groups[0]) == ["v1.3", "v2.3", "v3.3"]

    def test_every_model_covered_exactly_once(self):
        sizes = {"v1.3": 4, "v2.3": 4, "v3.3": 4, "v4.3": 4, "v5.3": 4}
        groups = pack_models(sizes, budget_bytes=9)  # 2 per group
        flat = [rev for g in groups for rev in g]
        assert sorted(flat) == sorted(sizes)
        assert len(flat) == len(set(flat))  # no duplicates

    def test_groups_respect_budget(self):
        sizes = {"a": 3, "b": 3, "c": 3, "d": 3}
        budget = 7
        groups = pack_models(sizes, budget_bytes=budget)
        for g in groups:
            assert sum(sizes[r] for r in g) <= budget

    def test_max_group_caps_group_size(self):
        sizes = {"a": 1, "b": 1, "c": 1, "d": 1}
        groups = pack_models(sizes, budget_bytes=1000, max_group=2)
        assert all(len(g) <= 2 for g in groups)
        assert len(groups) == 2  # 4 models, 2 per group

    def test_max_group_two_reproduces_two_at_a_time(self):
        sizes = {"v1.3": 1, "v2.3": 1, "v3.3": 1, "v4.3": 1, "v5.3": 1}
        groups = pack_models(sizes, budget_bytes=1000, max_group=2)
        # 5 models, 2 per group → [2, 2, 1]
        assert sorted((len(g) for g in groups), reverse=True) == [2, 2, 1]

    def test_oversized_model_gets_its_own_group(self):
        sizes = {"small": 1, "huge": 100}
        groups = pack_models(sizes, budget_bytes=10)
        # 'huge' exceeds the budget alone → solo group; 'small' separate
        huge_group = [g for g in groups if "huge" in g][0]
        assert huge_group == ["huge"]

    def test_zero_budget_degrades_to_one_per_group(self):
        sizes = {"a": 5, "b": 5, "c": 5}
        groups = pack_models(sizes, budget_bytes=0)
        assert all(len(g) == 1 for g in groups)
        assert len(groups) == 3

    def test_negative_budget_degrades_to_one_per_group(self):
        sizes = {"a": 5, "b": 5}
        groups = pack_models(sizes, budget_bytes=-100)
        assert all(len(g) == 1 for g in groups)

    def test_packing_is_deterministic(self):
        sizes = {"v1.3": 2, "v2.3": 3, "v3.3": 2, "v4.3": 5, "v5.3": 1}
        g1 = pack_models(sizes, budget_bytes=6)
        g2 = pack_models(sizes, budget_bytes=6)
        assert g1 == g2

    def test_largest_model_packed_first(self):
        """First-fit-decreasing: the biggest model anchors the first group."""
        sizes = {"small": 1, "big": 9, "mid": 5}
        groups = pack_models(sizes, budget_bytes=10)
        assert groups[0][0] == "big"

    def test_single_model_single_group(self):
        groups = pack_models({"only": 3}, budget_bytes=10)
        assert groups == [["only"]]


# ════════════════════════════════════════════════════════════════════════════
# registry_is_fresh
# ════════════════════════════════════════════════════════════════════════════
class TestRegistryIsFresh:
    """The recorded profile is only reusable on the same GPU + batch and when
    it covers every required model revision."""

    @staticmethod
    def _profile(name="NVIDIA A40", vram=48 * 1024 ** 3, batch=16, revs=("v1.3", "v2.3")):
        return {
            "gpu": {"name": name, "total_vram_bytes": vram},
            "batch": batch,
            "models": {r: {"peak_bytes": 1} for r in revs},
        }

    def test_fresh_when_everything_matches(self):
        p = self._profile()
        assert registry_is_fresh(p, "NVIDIA A40", 48 * 1024 ** 3, 16, ["v1.3", "v2.3"])

    def test_none_profile_is_stale(self):
        assert not registry_is_fresh(None, "NVIDIA A40", 1, 16, ["v1.3"])

    def test_empty_profile_is_stale(self):
        assert not registry_is_fresh({}, "NVIDIA A40", 1, 16, ["v1.3"])

    def test_gpu_name_mismatch_is_stale(self):
        p = self._profile(name="NVIDIA A40")
        assert not registry_is_fresh(p, "NVIDIA A100", 48 * 1024 ** 3, 16, ["v1.3", "v2.3"])

    def test_total_vram_mismatch_is_stale(self):
        p = self._profile(vram=48 * 1024 ** 3)
        assert not registry_is_fresh(p, "NVIDIA A40", 40 * 1024 ** 3, 16, ["v1.3", "v2.3"])

    def test_batch_mismatch_is_stale(self):
        p = self._profile(batch=16)
        assert not registry_is_fresh(p, "NVIDIA A40", 48 * 1024 ** 3, 32, ["v1.3", "v2.3"])

    def test_missing_required_revision_is_stale(self):
        p = self._profile(revs=("v1.3", "v2.3"))
        assert not registry_is_fresh(p, "NVIDIA A40", 48 * 1024 ** 3, 16, ["v1.3", "v2.3", "v5.3"])

    def test_extra_recorded_revision_is_still_fresh(self):
        """A profile that covers MORE models than needed is fine."""
        p = self._profile(revs=("v1.3", "v2.3", "v3.3"))
        assert registry_is_fresh(p, "NVIDIA A40", 48 * 1024 ** 3, 16, ["v1.3", "v2.3"])


# ════════════════════════════════════════════════════════════════════════════
# merge_best — byte-identical to the original inline merge
# ════════════════════════════════════════════════════════════════════════════
def _original_inline_merge(revision_best_models, rdf_by_rev):
    """Verbatim reproduction of run.py's pre-refactor merge loop, used as the
    golden reference for byte-identity."""
    combined_df = pd.DataFrame()
    for rev, rdf in rdf_by_rev.items():        # original iterated insertion order
        renamed_columns = {col: f"{col}-{rev}" for col in rdf.columns if col not in ["FILE", "PAGE"]}
        rdf_renamed = rdf.rename(columns=renamed_columns)
        if combined_df.empty:
            combined_df = rdf_renamed
        else:
            combined_df = pd.merge(combined_df, rdf_renamed, on=["FILE", "PAGE"], how="outer")
    return combined_df


class TestMergeBest:
    """The combined wide frame must match the original inline merge exactly so
    averaging.py's ^CLASS-(\\d+)-(.+)$ parsing keeps working."""

    @staticmethod
    def _rdf(class_map):
        """Build a single-model frame: [FILE, PAGE, CLASS-1]."""
        rows = [{"FILE": f, "PAGE": p, "CLASS-1": c} for (f, p), c in class_map.items()]
        return pd.DataFrame(rows).sort_values(["FILE", "PAGE"]).reset_index(drop=True)

    def _sample(self):
        revs = {"v1.3": "m1", "v2.3": "m2", "v3.3": "m3"}
        rdf_by_rev = {
            "v1.3": self._rdf({("atrium", 1): "TEXT_P", ("atrium", 2): "LINE_P"}),
            "v2.3": self._rdf({("atrium", 1): "TEXT",   ("atrium", 2): "LINE_P"}),
            "v3.3": self._rdf({("atrium", 1): "LINE_P", ("atrium", 2): "LINE_P"}),
        }
        return revs, rdf_by_rev

    def test_column_order_is_canonical(self):
        revs, rdf_by_rev = self._sample()
        out = merge_best(revs, rdf_by_rev)
        assert list(out.columns) == [
            "FILE", "PAGE", "CLASS-1-v1.3", "CLASS-1-v2.3", "CLASS-1-v3.3"
        ]

    def test_values_preserved_per_model(self):
        revs, rdf_by_rev = self._sample()
        out = merge_best(revs, rdf_by_rev).set_index(["FILE", "PAGE"])
        assert out.loc[("atrium", 1), "CLASS-1-v1.3"] == "TEXT_P"
        assert out.loc[("atrium", 1), "CLASS-1-v2.3"] == "TEXT"
        assert out.loc[("atrium", 1), "CLASS-1-v3.3"] == "LINE_P"

    def test_byte_identical_to_original_inline_merge(self):
        revs, rdf_by_rev = self._sample()
        new_csv = merge_best(revs, rdf_by_rev).to_csv(index=False)
        old_csv = _original_inline_merge(revs, rdf_by_rev).to_csv(index=False)
        assert new_csv == old_csv

    def test_canonical_order_independent_of_dict_insertion_order(self):
        """merge_best must follow revision_best_models order even if rdf_by_rev
        was populated out of order (as happens when groups execute v4.3 first)."""
        revs = {"v1.3": "m1", "v2.3": "m2", "v3.3": "m3"}
        _, rdf_by_rev = self._sample()
        shuffled = {"v3.3": rdf_by_rev["v3.3"], "v1.3": rdf_by_rev["v1.3"], "v2.3": rdf_by_rev["v2.3"]}
        out = merge_best(revs, shuffled)
        # Column order follows revs (canonical), not shuffled insertion order
        assert list(out.columns) == [
            "FILE", "PAGE", "CLASS-1-v1.3", "CLASS-1-v2.3", "CLASS-1-v3.3"
        ]

    def test_outer_merge_unions_disjoint_pages(self):
        revs = {"v1.3": "m1", "v2.3": "m2"}
        rdf_by_rev = {
            "v1.3": self._rdf({("docA", 1): "TEXT"}),
            "v2.3": self._rdf({("docB", 1): "DRAW"}),
        }
        out = merge_best(revs, rdf_by_rev)
        assert len(out) == 2  # outer join keeps both pages
