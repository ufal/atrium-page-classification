"""
tests/test_para_licenses.py
===========================
Unit tests for para_licenses.py: the licence FACTS of a run's components — one spelling per
licence, its URL, and the row the paradata list for a component. Nothing is ranked, resolved or
merged into a licence (D2, atrium-project#84), and these tests hold that.

Design notes
------------
* No ML models, no network, no GPU required — pure stdlib + pytest.
* This file is **intentionally identical** across the ATRIUM repositories; para_licenses.py is
  repo-agnostic. The canonical copy lives in atrium-project `docs/templates/shared/`, next to the
  canonical para_licenses.py enforced by the para-drift workflow.
"""

import pytest

import para_licenses
from para_licenses import LICENSE_URL, component_terms, normalise_license

# ════════════════════════════════════════════════════════════════════════════
# Catalogue invariants
# ════════════════════════════════════════════════════════════════════════════


class TestCatalogue:
    def test_every_url_is_https(self):
        assert all(url.startswith("https://") for url in LICENSE_URL.values())

    def test_agpl_is_catalogued(self):
        assert "AGPL-3.0" in LICENSE_URL
        assert "agpl.org" not in LICENSE_URL["AGPL-3.0"]  # gnu.org, not a squat
        assert LICENSE_URL["AGPL-3.0"].startswith("https://www.gnu.org/")

    def test_nothing_combines_licences(self):
        """D2: no effective, most-restrictive or calculated licence anywhere."""
        for gone in ("LICENSE_RANK", "resolve_effective_license", "merge_effective_licenses"):
            assert not hasattr(para_licenses, gone), gone


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

    def test_cdla_permissive_long_names(self):
        """Model cards spell the data licence out; both dash styles land on the SPDX key."""
        assert normalise_license("CDLA Permissive 2.0") == "CDLA-Permissive-2.0"
        long_ascii = "Community Data License Agreement - Permissive - Version 2.0"
        assert normalise_license(long_ascii) == "CDLA-Permissive-2.0"
        long_typo = "Community Data License Agreement \u2013 Permissive \u2013 Version 2.0"
        assert normalise_license(long_typo) == "CDLA-Permissive-2.0"

    def test_glm_variants(self):
        assert normalise_license("glm4") == "glm-4"
        assert normalise_license("GLM-4 License") == "glm-4"

    def test_unknown_license_passes_through_stripped(self):
        assert normalise_license("  Proprietary-EULA ") == "Proprietary-EULA"

    def test_canonical_key_not_in_alias_table_survives_exact_case(self):
        """Canonical keys like BSD-3-Clause have no alias entry; the exact
        string must survive so it still finds its URL in the catalogue."""
        assert normalise_license("BSD-3-Clause") == "BSD-3-Clause"
        assert normalise_license("BSD-3-Clause") in LICENSE_URL

    @pytest.mark.parametrize(
        "spdx, canonical",
        [
            ("CC0-1.0", "CC0"),
            ("CC-BY-4.0", "CC BY 4.0"),
            ("CC-BY-SA-4.0", "CC BY-SA 4.0"),
            ("CC-BY-NC-4.0", "CC BY-NC 4.0"),
            ("CC-BY-NC-SA-4.0", "CC BY-NC-SA 4.0"),
            ("MIT", "MIT"),
            ("Apache-2.0", "Apache-2.0"),
            ("BSD-3-Clause", "BSD-3-Clause"),
            ("CDLA-Permissive-2.0", "CDLA-Permissive-2.0"),
            ("MPL-2.0", "MPL 2.0"),
            ("LGPL-3.0", "LGPL-3.0"),
            ("LGPL-3.0-only", "LGPL-3.0"),
            ("LGPL-3.0-or-later", "LGPL-3.0"),
            ("GPL-3.0-only", "GPL-3.0"),
            ("GPL-3.0-or-later", "GPL-3.0"),
            ("AGPL-3.0-only", "AGPL-3.0"),
            ("AGPL-3.0-or-later", "AGPL-3.0"),
        ],
    )
    def test_spdx_identifiers_resolve(self, spdx, canonical):
        """SPDX ids are what model cards and package metadata actually carry;
        each must land on a catalogued name, not fall through as given."""
        assert normalise_license(spdx) == canonical
        assert normalise_license(spdx.lower()) == canonical
        assert canonical in LICENSE_URL

    def test_every_canonical_key_resolves_in_any_case(self):
        for key in LICENSE_URL:
            assert normalise_license(key.lower()) == key
            assert normalise_license(key.upper()) == key


# ════════════════════════════════════════════════════════════════════════════
# component_terms — one component's row in the paradata
# ════════════════════════════════════════════════════════════════════════════


class TestComponentTerms:
    def test_a_catalogued_licence_gets_its_canonical_name_and_url(self):
        assert component_terms("fasttext", "cc-by-nc-4.0", "0.9.3") == {
            "name": "fasttext",
            "version": "0.9.3",
            "license": "CC BY-NC 4.0",
            "license_url": LICENSE_URL["CC BY-NC 4.0"],
        }

    def test_unknown_terms_are_stated_not_guessed(self):
        assert component_terms("in-house model", None) == {"name": "in-house model", "license": "UNKNOWN"}
        assert component_terms("x", "Proprietary-EULA") == {"name": "x", "license": "Proprietary-EULA"}

    def test_rows_are_facts_never_combined(self):
        """Two components keep two rows, whatever their terms: nothing is merged (D2)."""
        rows = [component_terms("UDPipe 2", "CC BY-NC-SA 4.0"), component_terms("fasttext", "CC BY-NC 4.0")]
        assert [r["license"] for r in rows] == ["CC BY-NC-SA 4.0", "CC BY-NC 4.0"]
        assert all(set(r) <= {"name", "version", "license", "license_url"} for r in rows)

    @pytest.mark.parametrize("version", [None, ""])
    def test_no_version_no_key(self, version):
        assert "version" not in component_terms("x", "MIT", version)
