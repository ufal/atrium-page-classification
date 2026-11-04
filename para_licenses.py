"""
para_licenses.py – License resolution for ATRIUM paradata.

DROP THIS FILE AS-IS into every ATRIUM repository root, next to
atrium_paradata.py.

Purpose
-------
This module lists the components (models, data, APIs) a run exercised, each
with its own licence, and derives a calculated label from them: the most
restrictive of those licences. The label is NOT the licence of the run's
output, and no output licence is asserted (atrium-project#6, 2026-10-29):
records mark it `provenance.license_basis: "calculated"`, and an RO-Crate root
names `#no-licence-asserted` instead.

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
    # Community Data License Agreement – Permissive 2.0: the licence of model
    # weights and datasets (e.g. Docling's TableFormer). It places no terms on
    # "Results" (the output of a computation over the Data) — sharing the Data
    # itself only needs the licence text alongside — so it ranks with MIT.
    "CDLA-Permissive-2.0": 1,
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
    "CDLA-Permissive-2.0": "https://cdla.dev/permissive-2-0/",
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
    "gpl-3.0-only": "GPL-3.0",
    "gpl-3.0-or-later": "GPL-3.0",
    "lgpl-3.0": "LGPL-3.0",
    "lgpl-3.0-only": "LGPL-3.0",
    "lgpl-3.0-or-later": "LGPL-3.0",
    "agpl-3.0": "AGPL-3.0",
    "agpl 3.0": "AGPL-3.0",
    "agpl-3.0 license": "AGPL-3.0",
    "agplv3": "AGPL-3.0",
    "agpl-3.0-only": "AGPL-3.0",
    "agpl-3.0-or-later": "AGPL-3.0",
    "gnu affero general public license v3.0": "AGPL-3.0",
    "bsd-3-clause": "BSD-3-Clause",
    "cdla permissive 2.0": "CDLA-Permissive-2.0",
    "cdla-permissive 2.0": "CDLA-Permissive-2.0",
    "community data license agreement - permissive - version 2.0": "CDLA-Permissive-2.0",
    "community data license agreement – permissive – version 2.0": "CDLA-Permissive-2.0",
    "cc0": "CC0",
    "cc0-1.0": "CC0",
    "public domain": "Public Domain",
    "cc by 4.0": "CC BY 4.0",
    "cc-by-4.0": "CC BY 4.0",
    "cc by-sa 4.0": "CC BY-SA 4.0",
    "cc-by-sa-4.0": "CC BY-SA 4.0",
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


# Case-insensitive fallback onto the canonical keys themselves, so a catalogued
# license in any casing ("mit", "Cc By 4.0", "bsd-3-clause") resolves without an
# alias row per spelling. _ALIASES wins: it holds the spellings that are more than
# a case change (SPDX ids, long names).
_CANONICAL: Dict[str, str] = {k.lower(): k for k in LICENSE_RANK}


def normalise_license(name: str) -> str:
    """Map a free-text license string to a canonical key. Unknown -> as-is."""
    if not name:
        return ""
    key = name.strip()
    low = key.lower()
    return _ALIASES.get(low) or _CANONICAL.get(low) or key


def resolve_effective_license(
    components_used: Iterable[Tuple[str, str]],
) -> Dict[str, object]:
    """List the components used with their licences, and the calculated label over them."""
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

    # An unrecognised effective license is "treated as maximally restrictive" in
    # rank, so it is in its terms too: both flags set, never both false. A consumer
    # that reads only the flags must not take an unknown license (atrium_paradata
    # logs an undeclared component as "UNKNOWN") for a permissive one. No URL can
    # be claimed for it, so effective_license_url stays "".
    effective_unknown = best_license not in LICENSE_RANK
    is_nc = effective_unknown or best_license in _NON_COMMERCIAL
    is_sa = effective_unknown or best_license in _SHARE_ALIKE

    # The notes describe the components, not the output: a component's licence is not the
    # licence of what the pipeline produced, and no output licence is asserted
    # (atrium-project#6, 2026-10-29). effective_license and the two flags stay as the
    # calculated label the record carries with license_basis "calculated".
    nc_names = [str(c["name"]) for c in catalogue if c["license"] in _NON_COMMERCIAL]
    sa_names = [str(c["name"]) for c in catalogue if c["license"] in _SHARE_ALIKE]
    parts: List[str] = [
        f"{len(catalogue)} component(s) used; each keeps its own licence or terms (see components). "
        "No output licence is asserted."
    ]
    if nc_names:
        parts.append("Components under a NON-COMMERCIAL licence: " + ", ".join(nc_names) + ".")
    if sa_names:
        parts.append("Components under a SHARE-ALIKE licence: " + ", ".join(sa_names) + ".")
    if unknown:
        parts.append(
            "WARNING: unrecognised license(s) "
            + ", ".join(u or "(empty)" for u in sorted(set(unknown)))
            + " are counted as NON-COMMERCIAL and SHARE-ALIKE in the calculated label until their terms"
            " are catalogued; no license URL can be given. Update para_licenses.py."
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
