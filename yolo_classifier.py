"""
yolo_classifier.py  –  YOLO-cls wrapper that mirrors the ImageClassifier interface.
Requires: pip install ultralytics
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Ultralytics import guarded so the rest of the project still loads without it
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# ── helper ───────────────────────────────────────────────────────────────────

def _assert_ultralytics():
    if YOLO is None:
        raise ImportError(
            "ultralytics is not installed. Run: pip install ultralytics"
        )


def build_yolo_dataset(
    image_paths: list,
    labels: np.ndarray,
    categories: list,
    dest_dir: str | Path,
    split: str = "train",
) -> Path:
    """
    Copy images into the directory tree YOLO expects:
        <dest_dir>/<split>/<CLASS_NAME>/<filename>

    Args:
        image_paths : list of absolute image path strings
        labels      : one-hot label array, shape (N, C)
        categories  : ordered list of class-name strings
        dest_dir    : root of the temporary dataset
        split       : "train" | "val" | "test"

    Returns:
        Path to <dest_dir>/<split>
    """
    dest = Path(dest_dir) / split
    dest.mkdir(parents=True, exist_ok=True)

    label_indices = np.argmax(labels, axis=-1)

    for src, idx in zip(image_paths, label_indices):
        cls_dir = dest / categories[idx]
        cls_dir.mkdir(exist_ok=True)
        dst_file = cls_dir / Path(src).name
        if not dst_file.exists():
            shutil.copy2(src, dst_file)

    return dest


# ── main class ────────────────────────────────────────────────────────────────

class YOLOClassifier:
    """
    YOLO classification wrapper.

    The public API deliberately mirrors ImageClassifier so that run.py can
    route to either backend with minimal branching:

        classifier.train_model(...)
        classifier.create_dataloader(...)
        classifier.infer_dataloader(...)
        classifier.top_n_predictions(...)
        classifier.save_model(...)
        classifier.load_model(...)
    """

    # Mapping from short version tag to Ultralytics model identifier.
    # REVIEW FIX (Minor I): keys corrected to the documented short tags
    # (config.txt comment lists `yv8s`, `y11s`, …) and the map is now actually
    # USED via _resolve_checkpoint() below — previously it was dead code.
    BASE_MODELS = {
        "yv8n":  "yolov8n-cls.pt",
        "yv8s":  "yolov8s-cls.pt",
        "yv8m":  "yolov8m-cls.pt",
        "yv8l":  "yolov8l-cls.pt",
        "yv8x":  "yolov8x-cls.pt",
        "y11n":  "yolo11n-cls.pt",
        "y11s":  "yolo11s-cls.pt",
        "y11m":  "yolo11m-cls.pt",
        "y11l":  "yolo11l-cls.pt",
        "y11x":  "yolo11x-cls.pt",
    }

    @classmethod
    def _resolve_checkpoint(cls, checkpoint: str) -> str:
        """Resolve a checkpoint identifier.

        Accepts, in order of precedence:
          1. a short tag from BASE_MODELS (e.g. "yv8s")  → mapped to "yolov8s-cls.pt"
          2. an existing local .pt path                  → returned unchanged
          3. any other string (e.g. "yolov8s-cls.pt")    → returned unchanged
             (Ultralytics resolves / downloads it itself)
        """
        if checkpoint in cls.BASE_MODELS:
            return cls.BASE_MODELS[checkpoint]
        return checkpoint

    def __init__(
        self,
        checkpoint: str,          # short tag ("yv8s"), Ultralytics id, or local .pt path
        num_labels: int,
        categories: list,
        store_dir: str = "./checkpoint",
        imgsz: int = 224,
    ):
        _assert_ultralytics()

        self.checkpoint  = self._resolve_checkpoint(checkpoint)
        self.num_labels  = num_labels
        self.categories  = categories
        self.store_dir   = store_dir
        self.imgsz       = imgsz
        self.model: YOLO | None = None
        # Records the Ultralytics run directory (output_dir/out_model) so we can
        # reliably locate best.pt after training even if trainer.best is missing.
        self._last_run_dir: Path | None = None

        # device string for Ultralytics ("0" for first GPU, "cpu")
        if torch.cuda.is_available():
            self.device = "0"
            self._device_label = "cuda"
        else:
            self.device = "cpu"
            self._device_label = "cpu"

    # ── training ─────────────────────────────────────────────────────────────

    def train_model(
        self,
        trainfiles: list,
        trainLabels: np.ndarray,
        valfiles: list,
        valLabels: np.ndarray,
        out_model: str,
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 5e-5,
        output_dir: str = "./yolo_output",
        logging_steps: int = 10,     # kept for API parity, unused by YOLO
        patience: int = 100,         # early-stopping patience (epochs)
        dropout: float = 0.0,        # classifier head dropout
        cache: bool = False,         # cache images in RAM/disk for faster epochs
    ):
        """
        Prepare a temporary YOLO dataset, fine-tune, and save.

        Note: YOLO trains on-disk directory trees, so images are temporarily
        copied to a scratch directory and cleaned up afterwards.
        """
        _assert_ultralytics()
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Record where Ultralytics will write this run (project/name), so that
        # save_model() can locate weights/best.pt even on Ultralytics versions
        # where trainer.best is unset after train() returns.
        self._last_run_dir = Path(output_dir) / out_model

        # ── 1. Build temporary on-disk dataset ───────────────────────────────
        tmp_root = Path(tempfile.mkdtemp(prefix="yolo_data_"))
        try:
            print(f"[YOLO] Building dataset in {tmp_root} …")
            build_yolo_dataset(trainfiles, trainLabels, self.categories, tmp_root, "train")
            build_yolo_dataset(valfiles,   valLabels,   self.categories, tmp_root, "val")

            # ── 2. Initialise model from pretrained weights ───────────────────
            self.model = YOLO(self.checkpoint)

            # ── 3. Train ─────────────────────────────────────────────────────
            print(f"[YOLO] Training {self.checkpoint} for {num_epochs} epochs "
                  f"(batch={batch_size}, lr={learning_rate}, patience={patience}, "
                  f"dropout={dropout}, cache={cache}) on {self._device_label}")

            self.model.train(
                data=str(tmp_root),
                epochs=num_epochs,
                imgsz=self.imgsz,
                batch=batch_size,
                lr0=learning_rate,
                lrf=learning_rate * 0.01,   # final LR fraction
                patience=patience,
                dropout=dropout,
                cache=cache,
                device=self.device,
                project=output_dir,
                name=out_model,
                exist_ok=True,
                verbose=True,
                # Disable built-in augmentations that are irrelevant for
                # document pages (flips, perspective distortion, mosaic)
                fliplr=0.0,
                flipud=0.0,
                mosaic=0.0,
                perspective=0.0,
            )

            # Prefer the actual run directory the trainer reports, if available.
            trainer = getattr(self.model, "trainer", None)
            if trainer is not None and getattr(trainer, "save_dir", None):
                self._last_run_dir = Path(trainer.save_dir)

            # ── 4. Persist fine-tuned weights ────────────────────────────────
            self.save_model(f"model/{out_model}")

        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    # ── inference helpers ────────────────────────────────────────────────────

    def top_n_predictions(self, image_input, top_n: int = 1) -> list:
        """
        Single-image inference. Returns [(class_idx, prob), …] of length top_n,
        matching the ImageClassifier API exactly.
        """
        _assert_ultralytics()
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        if isinstance(image_input, str):
            image_input = Image.open(image_input)

        results = self.model.predict(
            source=image_input,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        probs = results[0].probs.data  # torch.Tensor, shape (num_classes,)

        top_indices = probs.argsort(descending=True)[:top_n]
        top_probs   = probs[top_indices]
        # Normalise so scores sum to 1 (matches ViT behaviour)
        top_probs   = top_probs / top_probs.sum()

        return list(zip(top_indices.tolist(), top_probs.tolist()))

    def create_dataloader(self, image_paths: list, batch_size: int,
                          ignored_paths: list = None) -> list:
        """
        For API parity with ImageClassifier.create_dataloader().
        YOLO predict handles batching internally, so we just filter the list
        and return it.  infer_dataloader() accepts this format.
        """
        if ignored_paths:
            ignored = set(ignored_paths)
            image_paths = [p for p in image_paths if p not in ignored]
        # Store batch_size so infer_dataloader can use it
        self._batch_size = batch_size
        print(f"[YOLO] Dataloader ready: {len(image_paths)} images "
              f"(batch_size={batch_size})")
        return image_paths   # plain list – YOLO streams from paths

    def infer_dataloader(self, image_paths: list, top_n: int,
                         raw: bool = False) -> tuple[list, list | None]:
        """
        Batch inference over a list of image paths.

        Returns
        -------
        predictions : list
            * top_n == 1  → list of int class indices
            * top_n > 1   → list of [(idx, prob), …]
        raw_scores  : list | None
            Full probability vectors if raw=True, else None.
        """
        _assert_ultralytics()
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        batch_size  = getattr(self, "_batch_size", 16)
        predictions = []
        raw_scores  = [] if raw else None

        total = len(image_paths)
        print(f"[YOLO] Running inference on {total} images …")

        for start in range(0, total, batch_size):
            chunk  = image_paths[start : start + batch_size]
            results = self.model.predict(
                source=chunk,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
                stream=True,
            )

            for res in results:
                probs = res.probs.data  # torch.Tensor (num_classes,)

                if raw:
                    raw_scores.append(probs.cpu().tolist())

                if top_n > 1:
                    top_idx   = probs.argsort(descending=True)[:top_n]
                    top_probs = probs[top_idx]
                    top_probs = top_probs / top_probs.sum()
                    predictions.append(
                        list(zip(top_idx.tolist(), top_probs.tolist()))
                    )
                else:
                    predictions.append(int(probs.argmax().item()))

            if (start // batch_size) % 10 == 0:
                print(f"[YOLO]   Processed {min(start + batch_size, total)}/{total}")

        return predictions, raw_scores

    # ── persistence ──────────────────────────────────────────────────────────

    def save_model(self, save_directory: str):
        """
        Persist the fine-tuned YOLO weights to <save_directory>/model.pt.

        Prefers the trainer's best.pt (best-epoch weights) over the current
        in-memory model. If best.pt cannot be located, falls back to exporting
        the in-memory model but warns loudly, because that may NOT be the
        best-epoch checkpoint.
        """
        if self.model is None:
            raise RuntimeError("Nothing to save – model has not been trained or loaded.")
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        dest = Path(save_directory) / "model.pt"

        best_pt = self._find_best_pt()
        if best_pt and best_pt.resolve() != dest.resolve():
            shutil.copy2(best_pt, dest)
            print(f"[YOLO] Best weights copied: {best_pt} → {dest}")
        elif best_pt and best_pt.resolve() == dest.resolve():
            print(f"[YOLO] Best weights already at {dest}")
        else:
            print("[YOLO] WARNING: could not locate best.pt — saving the "
                  "current in-memory model instead. This may not be the "
                  "best-epoch checkpoint.")
            self.model.save(str(dest))

        # Verify the save actually landed where run.py expects to load it from.
        if not dest.exists():
            raise RuntimeError(
                f"[YOLO] Save failed: {dest} does not exist after save_model()."
            )
        print(f"[YOLO] Model saved → {dest}  ({dest.stat().st_size / 1e6:.1f} MB)")

    def load_model(self, load_directory: str):
        _assert_ultralytics()
        pt_path = Path(load_directory) / "model.pt"
        if not pt_path.exists():
            # Fallback: search for any .pt in the directory
            candidates = list(Path(load_directory).glob("*.pt"))
            if not candidates:
                raise FileNotFoundError(
                    f"No .pt file found in {load_directory}"
                )
            pt_path = candidates[0]
        self.model = YOLO(str(pt_path))
        print(f"[YOLO] Model loaded from {pt_path}")

    def load_from_hub(self, *args, **kwargs):
        raise NotImplementedError(
            "YOLO models are not stored on HuggingFace. "
            "Use --model to point to a local directory."
        )

    def push_to_hub(self, *args, **kwargs):
        raise NotImplementedError("YOLO hub push is not supported.")

    # ── internal ─────────────────────────────────────────────────────────────

    def _find_best_pt(self) -> Path | None:
        """Locate the best.pt written by Ultralytics trainer.

        Tries, in order:
          1. trainer.best (the canonical attribute, may be str or Path)
          2. trainer.save_dir / "weights" / "best.pt"
          3. a recursive glob for best.pt under the recorded run directory
        """
        # 1. canonical trainer.best
        trainer = getattr(self.model, "trainer", None) if self.model else None
        if trainer is not None:
            best_attr = getattr(trainer, "best", None)
            if best_attr:
                best = Path(best_attr)
                if best.exists():
                    return best
            # 2. reconstruct from save_dir
            save_dir = getattr(trainer, "save_dir", None)
            if save_dir:
                cand = Path(save_dir) / "weights" / "best.pt"
                if cand.exists():
                    return cand

        # 3. fall back to globbing the known run output directory
        if self._last_run_dir is not None and Path(self._last_run_dir).exists():
            matches = sorted(Path(self._last_run_dir).rglob("best.pt"))
            if matches:
                # most-recently modified wins
                return max(matches, key=lambda p: p.stat().st_mtime)

        return None
