#!/usr/bin/env python3
"""
downscale.py
------------
Downscale all images in a source directory to a given percentage and save them
into a destination directory, preserving the category subdirectory structure.

Usage:
    python downscale.py                          # 30%, data_samples/ → data_samples_small/
    python downscale.py --scale 50               # 50% of original size
    python downscale.py --src my_data --dst small
    python downscale.py --dry-run                # preview without writing any files
    python downscale.py --overwrite              # re-process already-existing outputs
    python downscale.py --ext jpg                # process JPEG files instead of PNG
    python downscale.py --quiet                  # suppress per-file lines; show per-dir summary

Options:
    --src DIR      Source root directory (default: data_samples)
    --dst DIR      Destination root directory (default: data_samples_small)
    --scale FLOAT  Scale percentage; 30 means 30% of original size (default: 30)
    --ext EXT      File extension to process, without leading dot (default: png)
    --overwrite    Re-process files that already exist in the destination
    --dry-run      Preview what would be created without writing any files
    --quiet        Print one summary line per directory instead of one per file
"""

import argparse
from pathlib import Path
from PIL import Image


def downscale(
    src_root: Path,
    dst_root: Path,
    scale: float,
    ext: str,
    overwrite: bool,
    dry_run: bool,
    quiet: bool,
) -> None:
    factor = scale / 100.0
    ext_pattern = f"*.{ext.lstrip('.')}"

    src_dirs = sorted(p for p in src_root.iterdir() if p.is_dir())
    if not src_dirs:
        print(f"No subdirectories found under {src_root}")
        return

    total = created = skipped = 0

    for src_dir in src_dirs:
        images = sorted(src_dir.glob(ext_pattern))

        # P2 FIX: skip (and note) source subdirs with no matching images
        # instead of creating an empty destination directory.
        if not images:
            if not quiet:
                print(f"  [{src_dir.name}] no .{ext} file(s) — skipped.")
            continue

        dst_dir = dst_root / src_dir.name
        # Only create the destination directory when there is actual work to do
        # and we are not in dry-run mode.
        if not dry_run:
            dst_dir.mkdir(parents=True, exist_ok=True)

        dir_created = dir_skipped = 0

        for src_img in images:
            total += 1
            dst_img = dst_dir / src_img.name

            if dst_img.exists() and not overwrite:
                skipped += 1
                dir_skipped += 1
                continue

            # Fast path: dry-run + quiet needs no image I/O at all.
            if dry_run and quiet:
                created += 1
                dir_created += 1
                continue

            with Image.open(src_img) as im:
                new_w = max(1, round(im.width  * factor))
                new_h = max(1, round(im.height * factor))
                if not dry_run:
                    resized = im.resize((new_w, new_h), Image.LANCZOS)
                    resized.save(dst_img, optimize=True)

            created += 1
            dir_created += 1

            if not quiet:
                prefix = "[dry-run] " if dry_run else ""
                print(f"  {prefix}{src_dir.name}/{src_img.name}  →  {new_w}×{new_h}")

        # Per-directory summary line shown in --quiet mode (or when all files
        # were skipped so per-file output would have been empty anyway).
        if quiet or (dir_created == 0 and dir_skipped > 0):
            prefix = "[dry-run] " if dry_run else ""
            print(
                f"  {prefix}[{src_dir.name}]  "
                f"{dir_created} created, {dir_skipped} skipped"
            )

    action = "would create" if dry_run else "created"
    print(f"\nDone. {created} {action}, {skipped} skipped (already existed), {total} total.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Downscale a page-image dataset while preserving the category subdirectory structure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--src", default="data_samples", metavar="DIR",
        help="Source root directory (default: data_samples)",
    )
    parser.add_argument(
        "--dst", default="data_samples_small", metavar="DIR",
        help="Destination root directory (default: data_samples_small)",
    )
    parser.add_argument(
        "--scale", type=float, default=30.0,
        help="Scale percentage, e.g. 30 means 30%% of original (default: 30)",
    )
    parser.add_argument(
        "--ext", default="png", metavar="EXT",
        help="Image file extension to process, without leading dot (default: png)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-process files that already exist in the destination",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be created without writing any files",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Print one summary line per directory instead of one line per file",
    )
    args = parser.parse_args()

    # ── P1 FIX: validate scale before touching any files ─────────────────
    # --scale 0 would silently produce 1×1 pixel images (max(1, round(0))=1).
    # --scale -N does the same.  Both are clear user errors.
    if args.scale <= 0:
        raise SystemExit(
            f"[ERROR] --scale must be greater than 0 (got {args.scale}). "
            f"Did you mean --scale 30 for 30%?"
        )
    # P1 FIX: upscaling is silently allowed by the math but contradicts the
    # script's purpose; warn clearly so the user can catch a typo like
    # --scale 300 instead of --scale 30.
    if args.scale > 100:
        print(
            f"Warning: --scale {args.scale} is above 100% — images will be "
            f"upscaled, not downscaled."
        )

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    if not src_root.exists():
        raise SystemExit(f"[ERROR] Source directory not found: {src_root.resolve()}")

    print(f"Source   : {src_root.resolve()}")
    print(f"Dest     : {dst_root.resolve()}")
    print(f"Scale    : {args.scale}%")
    print(f"Ext      : .{args.ext}")
    if args.dry_run:
        print("Dry-run  : yes (no files will be written)")
    if args.overwrite:
        print("Overwrite: yes")
    print()

    downscale(
        src_root=src_root,
        dst_root=dst_root,
        scale=args.scale,
        ext=args.ext,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()