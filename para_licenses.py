"""
para_licenses.py – the licence facts of the components an ATRIUM run uses.

DROP THIS FILE AS-IS into every ATRIUM repository root, next to atrium_paradata.py.

Purpose
-------
A run's paradata list each component it used (software, model, dataset, service) with its
version and its terms, as facts (contract §6, layer 2). Nothing here combines them: no
"effective", "most restrictive" or "calculated" licence is derived, for a run, a record or a
crate (D2, atrium-project#84). This module only normalises a licence name to one spelling and
gives its URL, so the same terms read the same way in every tool's paradata.

The rights in a document's content are the document's own licence, otherwise AMČR's default,
stated by AMČR in the record's RO-Crate (layer 1); AMČR's own conditions are layer 3. Neither is
this module's business.
"""

from __future__ import annotations

from typing import Dict, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Licence catalogue: canonical names and their URLs
# ──────────────────────────────────────────────────────────────────────────────

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


# Spellings that are more than a case change (SPDX ids, long names) -> the canonical name.
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


# Case-insensitive fallback onto the canonical names themselves, so a catalogued licence in any
# casing ("mit", "Cc By 4.0") resolves without an alias row per spelling. _ALIASES wins.
_CANONICAL: Dict[str, str] = {k.lower(): k for k in LICENSE_URL}


def normalise_license(name: str) -> str:
    """Map a free-text licence string to its canonical name. Unknown -> as given (stripped)."""
    if not name:
        return ""
    key = name.strip()
    low = key.lower()
    return _ALIASES.get(low) or _CANONICAL.get(low) or key


def component_terms(name: str, license: Optional[str], version: Optional[str] = None) -> Dict[str, str]:
    """One component as the paradata list it: its name, its version when known, and its terms —
    the canonical licence name and, for a catalogued one, its URL. `UNKNOWN` when no terms are
    recorded. A fact about the component, never combined with another (D2)."""
    terms = normalise_license(license or "") or "UNKNOWN"
    row: Dict[str, str] = {"name": str(name)}
    if version:
        row["version"] = str(version)
    row["license"] = terms
    if terms in LICENSE_URL:
        row["license_url"] = LICENSE_URL[terms]
    return row
