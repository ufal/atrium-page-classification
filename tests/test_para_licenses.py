"""
tests/test_para_licenses.py
===========================
Unit tests for para_licenses.py (license normalisation, resolution, and the
#12 cross-stage merge deduplication).

Design notes
------------
* No ML models, no network, no GPU required — pure stdlib + pytest.
* This file is **intentionally identical** across all four ATRIUM
  repositories; para_licenses.py itself is repo-agnostic, so unlike
  test_paradata.py there is no per-repo constant to change.
* The canonical copy lives in atrium-project
  `docs/templates/shared/test_para_licenses.py`, next to the canonical
  para_licenses.py enforced by the para-drift workflow.
"""

from para_licenses import (
    LICENSE_RANK,
    LICENSE_URL,
    merge_effective_licenses,
    normalise_license,
    resolve_effective_license,
)

# ════════════════════════════════════════════════════════════════════════════
# Catalogue invariants
# ════════════════════════════════════════════════════════════════════════════


class TestCatalogueInvariants:
    """The static tables must stay mutually consistent."""

    def test_every_ranked_license_has_a_url(self):
        missing = [lic for lic in LICENSE_RANK if lic not in LICENSE_URL]
        assert missing == [], f"LICENSE_URL is missing entries for: {missing}"

    def test_every_url_entry_is_ranked(self):
        missing = [lic for lic in LICENSE_URL if lic not in LICENSE_RANK]
        assert missing == [], f"LICENSE_RANK is missing entries for: {missing}"

    def test_rank_ordering_sanity(self):
        """Spot-check the restrictiveness ordering the resolver relies on."""
        assert LICENSE_RANK["Public Domain"] < LICENSE_RANK["MIT"]
        assert LICENSE_RANK["MIT"] < LICENSE_RANK["CC BY-SA 4.0"]
        assert LICENSE_RANK["CC BY-SA 4.0"] < LICENSE_RANK["CC BY-NC 4.0"]
        assert LICENSE_RANK["CC BY-NC 4.0"] < LICENSE_RANK["CC BY-NC-SA 4.0"]
        assert max(LICENSE_RANK.values()) == LICENSE_RANK["CC BY-NC-SA 4.0"]

    def test_agpl_is_catalogued(self):
        """The AGPL-3.0 entries were once missing from one repo's copy —
        their presence is exactly what the canonical superset guarantees."""
        assert "AGPL-3.0" in LICENSE_RANK
        assert LICENSE_RANK["AGPL-3.0"] == LICENSE_RANK["GPL-3.0"]
        assert "agpl.org" not in LICENSE_URL["AGPL-3.0"]  # gnu.org, not a squat
        assert LICENSE_URL["AGPL-3.0"].startswith("https://www.gnu.org/")


# ════════════════════════════════════════════════════════════════════════════
# normalise_license
# ════════════════════════════════════════════════════════════════════════════


class TestNormaliseLicense:
    def test_empty_string_returns_empty(self):
        assert normalise_license("") == ""

    def test_alias_mapping_is_case_insensitive(self):
        assert normalise_license("MIT LICENSE") == "MIT"
        assert normalise_license("Apache License 2.0") == "Apache-2.0"
        assert normalise_license("AGPLv3") == "AGPL-3.0"

    def test_surrounding_whitespace_is_stripped(self):
        assert normalise_license("  mit  ") == "MIT"

    def test_cc_hyphenated_spdx_style(self):
        assert normalise_license("cc-by-nc-sa-4.0") == "CC BY-NC-SA 4.0"
        assert normalise_license("cc-by-4.0") == "CC BY 4.0"

    def test_long_form_cc_names(self):
        long_nc = "Attribution-NonCommercial 4.0 International"
        assert normalise_license(long_nc) == "CC BY-NC 4.0"
        long_nc_sa = "Attribution-NonCommercial-ShareAlike 4.0 International"
        assert normalise_license(long_nc_sa) == "CC BY-NC-SA 4.0"

    def test_glm_variants(self):
        assert normalise_license("glm4") == "glm-4"
        assert normalise_license("GLM-4 License") == "glm-4"

    def test_unknown_license_passes_through_stripped(self):
        assert normalise_license("  Proprietary-EULA ") == "Proprietary-EULA"

    def test_canonical_key_not_in_alias_table_survives_exact_case(self):
        """Canonical keys like BSD-3-Clause have no alias entry; the exact
        string must survive so it still matches LICENSE_RANK downstream."""
        assert normalise_license("BSD-3-Clause") == "BSD-3-Clause"
        assert normalise_license("BSD-3-Clause") in LICENSE_RANK


# ════════════════════════════════════════════════════════════════════════════
# resolve_effective_license
# ════════════════════════════════════════════════════════════════════════════


class TestResolveEffectiveLicense:
    def test_no_components_defaults_to_mit(self):
        result = resolve_effective_license([])
        assert result["effective_license"] == "MIT"
        assert result["components"] == []
        assert result["determined_by"] == []
        assert result["unknown_licenses"] == []
        assert result["is_non_commercial"] is False
        assert result["is_share_alike"] is False

    def test_single_permissive_component(self):
        result = resolve_effective_license([("mytool", "MIT")])
        assert result["effective_license"] == "MIT"
        assert result["effective_license_url"] == LICENSE_URL["MIT"]
        assert result["determined_by"] == ["mytool"]
        assert len(result["components"]) == 1

    def test_most_restrictive_component_wins(self):
        result = resolve_effective_license([("code", "MIT"), ("cubbitt", "CC BY-NC-SA 4.0"), ("lib", "Apache-2.0")])
        assert result["effective_license"] == "CC BY-NC-SA 4.0"
        assert result["determined_by"] == ["cubbitt"]

    def test_non_commercial_flag_and_note(self):
        result = resolve_effective_license([("model", "CC BY-NC 4.0")])
        assert result["is_non_commercial"] is True
        assert result["is_share_alike"] is False
        assert "NON-COMMERCIAL" in result["notes"]

    def test_share_alike_flag_and_note(self):
        result = resolve_effective_license([("data", "CC BY-SA 4.0")])
        assert result["is_share_alike"] is True
        assert result["is_non_commercial"] is False
        assert "SHARE-ALIKE" in result["notes"]

    def test_nc_sa_sets_both_flags(self):
        result = resolve_effective_license([("cubbitt", "CC BY-NC-SA 4.0")])
        assert result["is_non_commercial"] is True
        assert result["is_share_alike"] is True

    def test_glm4_is_non_commercial(self):
        result = resolve_effective_license([("glm-4v-9b", "glm-4")])
        assert result["effective_license"] == "glm-4"
        assert result["is_non_commercial"] is True

    def test_copyleft_code_licenses_are_share_alike(self):
        for lic in ("GPL-3.0", "AGPL-3.0", "LGPL-3.0", "MPL 2.0"):
            result = resolve_effective_license([("lib", lic)])
            assert result["is_share_alike"] is True, lic

    def test_equal_rank_tie_keeps_first_but_credits_all(self):
        """Strict > comparison: first component of the winning rank names the
        license; determined_by lists every component at that rank."""
        result = resolve_effective_license([("a", "MIT"), ("b", "Apache-2.0")])
        assert result["effective_license"] == "MIT"
        assert result["determined_by"] == ["a", "b"]

    def test_raw_strings_are_normalised_before_ranking(self):
        result = resolve_effective_license([("x", "agplv3"), ("y", "mit license")])
        assert result["effective_license"] == "AGPL-3.0"
        licenses = {c["license"] for c in result["components"]}
        assert licenses == {"AGPL-3.0", "MIT"}

    def test_unknown_license_treated_as_maximally_restrictive(self):
        result = resolve_effective_license([("blob", "Mystery-1.0"), ("code", "MIT")])
        assert result["effective_license"] == "Mystery-1.0"
        assert result["unknown_licenses"] == ["Mystery-1.0"]
        assert "WARNING" in result["notes"]
        assert "blob" in result["determined_by"]

    def test_unknown_does_not_displace_known_max_seen_first(self):
        """A known max-rank license seen first keeps the title; the unknown
        still ranks equal-max, so both components share determined_by."""
        result = resolve_effective_license([("cubbitt", "CC BY-NC-SA 4.0"), ("blob", "Mystery-1.0")])
        assert result["effective_license"] == "CC BY-NC-SA 4.0"
        assert result["unknown_licenses"] == ["Mystery-1.0"]
        assert set(result["determined_by"]) == {"cubbitt", "blob"}

    def test_unknown_catalogue_entry_shape(self):
        result = resolve_effective_license([("blob", "Mystery-1.0")])
        (entry,) = result["components"]
        assert entry["name"] == "blob"
        assert entry["license"] == "Mystery-1.0"
        assert entry["license_url"] == ""
        assert entry["rank"] == -1

    def test_known_catalogue_entry_shape(self):
        result = resolve_effective_license([("udpipe", "CC BY-NC-SA 4.0")])
        (entry,) = result["components"]
        assert entry["license_url"] == LICENSE_URL["CC BY-NC-SA 4.0"]
        assert entry["rank"] == LICENSE_RANK["CC BY-NC-SA 4.0"]

    def test_duplicate_unknowns_reported_once_sorted(self):
        result = resolve_effective_license([("a", "Zeta-1"), ("b", "Zeta-1"), ("c", "Alpha-1")])
        assert result["unknown_licenses"] == ["Alpha-1", "Zeta-1"]

    def test_notes_mention_component_count(self):
        result = resolve_effective_license([("a", "MIT"), ("b", "MIT"), ("c", "MIT")])
        assert "3 component(s)" in result["notes"]


# ════════════════════════════════════════════════════════════════════════════
# merge_effective_licenses — the #12 dedup blueprint
# ════════════════════════════════════════════════════════════════════════════


def _block(*components):
    """Build a per-tool license block the way resolve_effective_license does."""
    return resolve_effective_license(list(components))


class TestMergeEffectiveLicenses:
    def test_always_on_component_counted_once_across_stages(self):
        """(#12) The same (name, license) pair appearing in several stage
        blocks must survive as a single catalogue entry after the merge."""
        stage_a = _block(("UDPipe", "CC BY-NC-SA 4.0"), ("mytool", "MIT"))
        stage_b = _block(("UDPipe", "CC BY-NC-SA 4.0"), ("othertool", "MIT"))
        merged = merge_effective_licenses([stage_a, stage_b])
        names = [c["name"] for c in merged["components"]]
        assert names.count("UDPipe") == 1
        assert len(merged["components"]) == 3
        assert "3 component(s)" in merged["notes"]

    def test_same_name_different_license_is_not_deduped(self):
        """Dedup key is (name, license) — a relicensed component is real
        information and must keep both entries."""
        merged = merge_effective_licenses([_block(("tool", "MIT")), _block(("tool", "GPL-3.0"))])
        assert len(merged["components"]) == 2
        assert merged["effective_license"] == "GPL-3.0"

    def test_merge_resolution_matches_flat_resolution(self):
        """Merging blocks must give the same answer as resolving the deduped
        union directly."""
        merged = merge_effective_licenses([_block(("code", "MIT")), _block(("cubbitt", "CC BY-NC-SA 4.0"))])
        flat = resolve_effective_license([("code", "MIT"), ("cubbitt", "CC BY-NC-SA 4.0")])
        assert merged["effective_license"] == flat["effective_license"]
        assert merged["is_non_commercial"] is True
        assert merged["is_share_alike"] is True

    def test_merge_of_single_block_is_a_fixed_point(self):
        block = _block(("a", "MIT"), ("b", "CC BY 4.0"))
        merged = merge_effective_licenses([block])
        assert merged["effective_license"] == block["effective_license"]
        assert len(merged["components"]) == len(block["components"])

    def test_empty_blocks_list_defaults_to_mit(self):
        merged = merge_effective_licenses([])
        assert merged["effective_license"] == "MIT"
        assert merged["components"] == []

    def test_block_without_components_key_is_tolerated(self):
        merged = merge_effective_licenses([{}, _block(("a", "MIT"))])
        assert merged["effective_license"] == "MIT"
        assert len(merged["components"]) == 1

    def test_unknowns_propagate_through_merge(self):
        merged = merge_effective_licenses([_block(("blob", "Mystery-1.0")), _block(("code", "MIT"))])
        assert merged["unknown_licenses"] == ["Mystery-1.0"]
        assert "WARNING" in merged["notes"]

    def test_realistic_atrium_pipeline_merge(self):
        """LINDAT CUBBITT + UDPipe stages resolve to CC BY-NC-SA 4.0, the
        documented effective license of the translation+enrichment chain."""
        translate = _block(("LINDAT CUBBITT", "CC BY-NC-SA 4.0"), ("lxml", "BSD-3-Clause"))
        enrich = _block(("UDPipe", "CC BY-NC-SA 4.0"), ("NameTag 3", "CC BY-NC-SA 4.0"))
        merged = merge_effective_licenses([translate, enrich])
        assert merged["effective_license"] == "CC BY-NC-SA 4.0"
        assert merged["is_non_commercial"] is True
        assert merged["is_share_alike"] is True
        assert len(merged["components"]) == 4
