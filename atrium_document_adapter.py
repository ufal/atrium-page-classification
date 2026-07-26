import os
from pathlib import Path

import pandas as pd

from atrium_document import AtriumDocument  # Ensure this is vendored from the ATRIUM hub


def write_classification_document_records(
    rdf,
    *,
    document_input_dir=None,
    document_output_dir=None,
    classification_csv_ref=None,
    paradata_logger=None,
    strict=False,
):
    """
    Converts a normalized classification DataFrame into AtriumDocument JSON records.
    """
    if not document_output_dir:
        return  # Opt-in behavior: Do nothing if no output directory is specified

    os.makedirs(document_output_dir, exist_ok=True)

    # Process each document independently based on the FILE stem
    for doc_id, group in rdf.groupby("FILE"):
        doc = AtriumDocument()

        # 1. Load baseline if one exists
        if document_input_dir:
            input_path = Path(document_input_dir) / f"{doc_id}.document.json"
            if input_path.exists():
                doc = AtriumDocument.from_file(input_path, strict=strict)

        # 2. Build page_categories and pages patches
        page_categories = {}
        page_patches = []

        for _, row in group.iterrows():
            page_str = str(row["PAGE"])
            cat = row["CLASS-1"]

            # Populate top-level routing map
            page_categories[page_str] = cat

            # Populate field-level page patch
            patch = {"page": page_str, "category": cat}

            # Only add confidence if it exists (handles top_N == 1 missing score safely)
            if "SCORE-1" in row and not pd.isna(row["SCORE-1"]):
                patch["category_confidence"] = float(row["SCORE-1"])

            page_patches.append(patch)

        # 3. Apply updates using canonical merge semantics
        doc.set_block("page_categories", page_categories)
        doc.merge_block(
            "pages",
            page_patches,
            key_fields=["page"],
            own_fields=["category", "category_confidence"],
        )

        # 4. Handle Provenance, License, and References
        if classification_csv_ref:
            doc.add_derived_from("classification", classification_csv_ref)

        if paradata_logger:
            # Append accumulated license restrictions
            license_block = paradata_logger.get_license_block()
            if license_block:
                doc.add_license_detail(license_block)

            # Extract exact paradata run identity
            run_id = getattr(paradata_logger, "_run_id", "unknown_run")
            paradata_ref = f"paradata/{run_id}_page-classification.json"

            # Stamp the blocks we legally own/updated
            provenance_meta = {"program": "page-classification", "run_id": run_id, "paradata_ref": paradata_ref}
            doc.stamp_block("page_categories", **provenance_meta)
            doc.stamp_block("pages", **provenance_meta)

        # 5. Atomic Write
        out_path = Path(document_output_dir) / f"{doc_id}.document.json"
        temp_path = out_path.with_suffix(".json.tmp")

        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(doc.to_json(indent=2))

        temp_path.replace(out_path)
