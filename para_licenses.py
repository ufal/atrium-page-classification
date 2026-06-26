"""
para_licenses.py – License resolution for ATRIUM paradata.

DROP THIS FILE AS-IS into every ATRIUM repository root, next to
atrium_paradata.py.

Purpose
-------
The output license of an ATRIUM pipeline is NOT fixed.  It is the most
restrictive license among the components (models, data, APIs) that a given
run actually exercised.  This module turns a set of "components used" into a
single effective output license, plus the supporting attribution detail.

IMPORTANT – legal review required
---------------------------------
The restrictiveness ordering and the share-alike propagation rule encoded
below are a *mechanical engineering approximation*, not legal advice.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# License catalogue
# ──────────────────────────────────────────────────────────────────────────────

LICENSE_RANK: Dict[str, int] = {
    "Public Domain": 0,
    "CC0": 0,
    "MIT": 1,
    "Apache-2.0": 1,
    "BSD-3-Clause": 1,
    "MPL 2.0": 2,
    "LGPL-3.0": 3,
    "GPL-3.0": 4,
    "AGPL-3.0": 4,
    "CC BY 4.0": 1,
    "CC BY-SA 4.0": 3,
    "CC BY-NC 4.0": 5,
    "CC BY-NC-SA 4.0": 6,
    "glm-4": 5,
}

LICENSE_URL: Dict[str, str] = {
    "Public Domain": "https://creativecommons.org/publicdomain/mark/1.0/",
    "CC0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "MIT": "https://opensource.org/license/mit/",
    "Apache-2.0": "https://www.apache.org/licenses/LICENSE-2.0",
    "BSD-3-Clause": "https://opensource.org/license/bsd-3-clause/",
    "MPL 2.0": "https://www.mozilla.org/en-US/MPL/2.0/",
    "LGPL-3.0": "https://www.gnu.org/licenses/lgpl-3.0.html",
    "GPL-3.0": "https://www.gnu.org/licenses/gpl-3.0.html",
    "AGPL-3.0": "https://www.gnu.org/licenses/agpl-3.0.html",
    "CC BY 4.0": "https://creativecommons.org/licenses/by/4.0/",
    "CC BY-SA 4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
    "CC BY-NC 4.0": "https://creativecommons.org/licenses/by-nc/4.0/",
    "CC BY-NC-SA 4.0": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "glm-4": "https://huggingface.co/THUDM/glm-4v-9b/blob/main/LICENSE",
}

_SHARE_ALIKE = {"CC BY-SA 4.0", "CC BY-NC-SA 4.0", "GPL-3.0", "AGPL-3.0", "LGPL-3.0", "MPL 2.0"}
_NON_COMMERCIAL = {"CC BY-NC 4.0", "CC BY-NC-SA 4.0", "glm-4"}

_ALIASES: Dict[str, str] = {
    "apache 2.0": "Apache-2.0",
    "apache-2.0": "Apache-2.0",
    "apache license 2.0": "Apache-2.0",
    "mit": "MIT",
    "mit license": "MIT",
    "mpl-2.0": "MPL 2.0",
    "mpl 2.0": "MPL 2.0",
    "gpl-3.0": "GPL-3.0",
    "gpl 3.0": "GPL-3.0",
    "gpl-3.0 license": "GPL-3.0",
    "agpl-3.0": "AGPL-3.0",
    "agpl 3.0": "AGPL-3.0",
    "agpl-3.0 license": "AGPL-3.0",
    "agplv3": "AGPL-3.0",
    "gnu affero general public license v3.0": "AGPL-3.0",
    "cc0": "CC0",
    "public domain": "Public Domain",
    "cc by 4.0": "CC BY 4.0",
    "cc-by-4.0": "CC BY 4.0",
    "cc by-sa 4.0": "CC BY-SA 4.0",
    "cc by-nc 4.0": "CC BY-NC 4.0",
    "cc-by-nc-4.0": "CC BY-NC 4.0",
    "attribution-noncommercial 4.0 international": "CC BY-NC 4.0",
    "cc by-nc-sa 4.0": "CC BY-NC-SA 4.0",
    "cc-by-nc-sa-4.0": "CC BY-NC-SA 4.0",
    "attribution-noncommercial-sharealike 4.0 international": "CC BY-NC-SA 4.0",
    "glm-4": "glm-4",
    "glm4": "glm-4",
    "glm-4 license": "glm-4",
}


def normalise_license(name: str) -> str:
    """Map a free-text license string to a canonical key. Unknown -> as-is."""
    if not name:
        return ""
    key = name.strip()
    return _ALIASES.get(key.lower(), key)


def resolve_effective_license(
    components_used: Iterable[Tuple[str, str]],
) -> Dict[str, object]:
    """Compute the effective output license from the components actually used."""
    catalogue: List[Dict[str, object]] = []
    unknown: List[str] = []
    best_rank = -1
    best_license = "MIT"

    for name, raw_lic in components_used:
        lic = normalise_license(raw_lic)
        rank = LICENSE_RANK.get(lic)
        catalogue.append(
            {
                "name": name,
                "license": lic,
                "license_url": LICENSE_URL.get(lic, ""),
                "rank": rank if rank is not None else -1,
            }
        )
        if rank is None:
            unknown.append(lic)
            rank = max(LICENSE_RANK.values())
            if rank > best_rank:
                best_rank = rank
                best_license = lic
            continue
        if rank > best_rank:
            best_rank = rank
            best_license = lic

    determined_by = [
        c["name"] for c in catalogue if (LICENSE_RANK.get(str(c["license"]), max(LICENSE_RANK.values())) == best_rank)
    ]

    is_nc = best_license in _NON_COMMERCIAL
    is_sa = best_license in _SHARE_ALIKE

    parts: List[str] = []
    parts.append(
        f"Effective output license is {best_license}, the most restrictive among {len(catalogue)} component(s) used."
    )
    if is_nc:
        parts.append(
            "Outputs are NON-COMMERCIAL: downstream commercial use is not "
            "permitted while this component is in the pipeline."
        )
    if is_sa:
        parts.append("SHARE-ALIKE applies: derivatives must be licensed under the same terms.")
    if unknown:
        parts.append(
            "WARNING: unrecognised license(s) "
            + ", ".join(sorted(set(unknown)))
            + " were treated as maximally restrictive. Update para_licenses.py."
        )

    return {
        "effective_license": best_license,
        "effective_license_url": LICENSE_URL.get(best_license, ""),
        "is_non_commercial": is_nc,
        "is_share_alike": is_sa,
        "determined_by": determined_by,
        "components": catalogue,
        "unknown_licenses": sorted(set(unknown)),
        "notes": " ".join(parts),
    }


def merge_effective_licenses(
    license_blocks: Iterable[Dict[str, object]],
) -> Dict[str, object]:
    """
    Merge several per-tool license resolutions into one effective license.

    (#12) The union of components across blocks is DEDUPLICATED on
    (name, license) before resolving. Always-on components otherwise repeat once
    per stage, inflating the component catalogue. Deduping ensures the reported
    count reflects the unique set.
    """
    seen: set = set()
    union: List[Tuple[str, str]] = []
    for block in license_blocks:
        for comp in block.get("components", []):  # type: ignore[union-attr]
            key = (str(comp["name"]), str(comp["license"]))
            if key in seen:
                continue
            seen.add(key)
            union.append(key)
    return resolve_effective_license(union)
