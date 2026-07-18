from __future__ import annotations

import datetime
import os

import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ═══════════════════════════════════════════════════════════════════════════
# agent-skill branch: inference-only classifier.
#
# The training / evaluation machinery of the full classifier.py (train_model,
# BalancedBatchSampler, PageImageDataset, split_data_*, average_model_weights,
# push_to_hub, compute_metrics and the augmentation transforms) lives on the
# `test` branch. This branch keeps exactly the surface consumed by
# service/inference.py, parallel_best.py and the slim run.py CLI:
#
#   ImageClassifier(checkpoint, num_labels, store_dir)
#       .load_model() / .load_from_hub() / .save_model()
#       .infer() / .top_n_predictions()
#       .create_dataloader() / .infer_dataloader()
#   ImageDataset, custom_collate
# ═══════════════════════════════════════════════════════════════════════════

# Archival scans can be very large and occasionally truncated; keep PIL
# permissive instead of raising DecompressionBombError / truncation errors.
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 4221790634


def custom_collate(batch: list):
    """
    Custom collate function to filter out None entries (unreadable images)
    from an inference batch.
    """
    # Filter out None entries
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None, None

    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    return {"pixel_values": pixel_values}


class ImageDataset(Dataset):
    """Unlabeled image dataset for batched inference."""

    def __init__(self, image_paths: list, transform=None, ignored_paths: list = None):
        self.image_paths = image_paths
        self.transform = transform

        if ignored_paths is not None:
            # Filter out ignored paths
            self.image_paths = [
                path for path in image_paths if path not in ignored_paths
            ]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path)
            # Check the image mode
            if image.mode != "RGB":
                # Convert RGBA to RGB
                image_alpha = image.convert("RGBA")
                new_image = Image.new("RGBA", image_alpha.size, "WHITE")
                new_image.paste(image_alpha, (0, 0), image_alpha)
                image = new_image.convert("RGB")
            if self.transform:
                image = self.transform(image)

            return {"pixel_values": image}
        except Exception as e:
            print(image_path, e)
            return None


class ImageClassifier:
    # REVIEW FIX (Minor I): default store_dir corrected from the typo
    # "./chekcpoint" to "./checkpoint", matching config.txt FOLDER_CPOINTS and
    # avoiding a stray misspelled cache directory (service/inference.py
    # constructs ImageClassifier without store_dir, so it relied on this default).
    def __init__(
        self, checkpoint: str, num_labels: int, store_dir: str = "./checkpoint"
    ):
        """
        Initialize the image classifier with the specified checkpoint.
        """
        # --- REFINED: Comprehensive hardware acceleration check ---
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model_name = checkpoint

        self.model = AutoModelForImageClassification.from_pretrained(
            checkpoint,
            num_labels=num_labels,
            cache_dir=store_dir,
            ignore_mismatched_sizes=True,
        ).to(self.device)

        if checkpoint.startswith("timm"):
            # For timm models, input_size is [batch_size, channels, height, width]
            image_size = self.model.config.pretrained_cfg["input_size"][-1]
            image_mean = self.model.config.pretrained_cfg["mean"]
            image_std = self.model.config.pretrained_cfg["std"]
            self.processor = None
        else:
            self.processor = AutoImageProcessor.from_pretrained(checkpoint)
            image_size = self.processor.size["height"]
            image_mean = self.processor.image_mean
            image_std = self.processor.image_std

        # Inference-only: the augmentation (train) transforms of the full
        # classifier are dropped on this branch; only the eval pipeline stays.
        self.eval_transforms = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=image_mean, std=image_std),
            ]
        )

    def preprocess_image(self, image_input) -> torch.Tensor:
        """
        Preprocess a single image (path or PIL.Image) for inference.
        """
        if isinstance(image_input, Image.Image):
            image = image_input
        else:
            # Assume it is a string path
            image = Image.open(image_input)
        # Check the image mode
        if image.mode != "RGB":
            # Convert RGBA to RGB
            image_alpha = image.convert("RGBA")
            new_image = Image.new(
                "RGBA", image_alpha.size, "WHITE"
            )  # Create a white rgba background
            new_image.paste(image_alpha, (0, 0), image_alpha)
            image = new_image.convert("RGB")
        tensor = self.eval_transforms(image).unsqueeze(0).to(self.device)
        return tensor

    def infer(self, image_input) -> int:
        """
        Perform inference on a single image.
        """
        self.model.eval()
        with torch.no_grad():
            inputs = self.preprocess_image(image_input)
            outputs = self.model(pixel_values=inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            return predicted_class_idx

    def top_n_predictions(self, image_input, top_n: int = 1) -> list:
        """
        Perform inference and return top-N predictions with normalized probabilities.
        """
        self.model.eval()
        with torch.no_grad():
            inputs = self.preprocess_image(image_input)
            outputs = self.model(pixel_values=inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            top_n_probs, top_n_indices = torch.topk(probabilities, top_n, dim=-1)
            top_n_probs = top_n_probs / top_n_probs.sum()
        return list(
            zip(top_n_indices.squeeze().tolist(), top_n_probs.squeeze().tolist())
        )

    def create_dataloader(
        self, image_paths: list, batch_size: int, ignored_paths: list = None
    ) -> DataLoader:
        """
        Turn an input list of image paths into a DataLoader without labels.
        """
        dataset = ImageDataset(
            image_paths, transform=self.eval_transforms, ignored_paths=ignored_paths
        )
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate
        )
        print(
            f"Dataloader of directory dataset is ready:\t{len(image_paths)} images split into {len(dataloader)} batches of size {batch_size}"
        )
        return dataloader

    def infer_dataloader(
        self, dataloader, top_n: int, raw: bool = False
    ) -> (list, list):
        """
        Perform inference on a DataLoader, optionally with top-N predictions.
        """
        self.model.eval()
        predictions = []
        raw_scores = []

        start_time = datetime.datetime.now()
        print(
            f"\tProcessing of {len(dataloader)} batches started at\t{start_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        with torch.no_grad():
            for ib, batch in enumerate(dataloader):
                # Check if batch is None or the tuple (None, None) returned by custom_collate
                if batch is None or (isinstance(batch, tuple) and batch[0] is None):
                    print(f"Skipping batch {ib}: No valid images loaded.")
                    continue  # Skip this loop iteration

                inputs = batch["pixel_values"]
                outputs = self.model(pixel_values=inputs.to(self.device))
                logits = outputs.logits

                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                raw_scores.extend(probabilities.tolist())
                if top_n > 1:
                    top_n_probs, top_n_indices = torch.topk(
                        probabilities, top_n, dim=-1
                    )
                    for indices, probs in zip(top_n_indices, top_n_probs):
                        top_n_probs_normalized = probs / probs.sum()
                        predictions.append(
                            list(zip(indices.tolist(), top_n_probs_normalized.tolist()))
                        )
                else:
                    predicted_class_idx = logits.argmax(-1).tolist()
                    predictions.extend(predicted_class_idx)

                if ib % 50 == 0:
                    elapsed_minutes = (
                        datetime.datetime.now() - start_time
                    ).total_seconds() / 60
                    print(
                        f"{ib}-th batch\t\tProcessed {len(predictions)} images in\t{elapsed_minutes:.2f} min"
                    )

        end_time = datetime.datetime.now()
        total_minutes = (end_time - start_time).total_seconds() / 60

        # REVIEW FIX (Minor B): guard against an empty prediction set (every
        # batch skipped because all images were unreadable).  The original
        # divided by len(predictions) unconditionally → ZeroDivisionError.
        n_pred = len(predictions)
        if n_pred == 0:
            print(
                "\tWARNING: no images were successfully processed (all batches skipped)."
            )
            return predictions, (None if not raw else raw_scores)

        avg_seconds_per_image = (end_time - start_time).total_seconds() / n_pred

        print(
            f"\tProcessing of {len(dataloader)} batches ({n_pred} images) finished at\t{end_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print(
            f"\tTotal time: {total_minutes:.2f} min\n\tAverage time: {avg_seconds_per_image:.4f} sec/img"
        )

        raw_scores = None if not raw else raw_scores
        return predictions, raw_scores

    def save_model(self, save_directory: str):
        """
        Save the fine-tuned model and processor to the specified directory.
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        self.model.save_pretrained(save_directory)
        if self.processor is not None:
            self.processor.save_pretrained(save_directory)
        print(f"Model and processor saved to {save_directory}")

    def load_model(self, load_directory: str):
        """
        Load a fine-tuned model and processor from the specified directory.
        """
        self.processor = AutoImageProcessor.from_pretrained(load_directory)
        self.model = AutoModelForImageClassification.from_pretrained(load_directory).to(
            self.device
        )
        print(f"Model and processor loaded from {load_directory}")

    def load_from_hub(self, repo_id: str, revision: str = "main"):
        """
        Load a model and its processor from the Hugging Face Hub.

        Args:
            repo_id (str): The name of the repository on the Hugging Face Hub.
            revision (str, optional): The revision of the repository to load. Defaults to "main".
        """
        print(
            f"Accessing the Hugging Face Hub repository {repo_id}, revision {revision}..."
        )

        model = AutoModelForImageClassification.from_pretrained(
            repo_id,
            revision=revision,
            num_labels=self.model.config.num_labels,  # ← pass through num_labels
            ignore_mismatched_sizes=True,  # ← allow head size mismatch
        )
        processor = AutoImageProcessor.from_pretrained(repo_id, revision=revision)

        # REVIEW FIX (agent-skill): the freshly downloaded model must be moved to
        # the classifier's device. The previous version left it on CPU, so the
        # download-then-predict path in service/inference.py crashed with a
        # device mismatch on CUDA/MPS hosts (inputs on device, weights on CPU).
        self.model, self.processor = model.to(self.device), processor
        print(f"Model and processor loaded from the Hugging Face Hub: {repo_id}")
