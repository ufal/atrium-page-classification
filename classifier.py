import datetime
import os
import pickle
import re
from collections import Counter, defaultdict
import random
from pathlib import Path
import argparse
# from dotenv import load_dotenv

import time

import h5py
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler

import multiprocessing as mp

import sys
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
import torchvision
from torchvision import transforms
from tqdm import tqdm
import clip
from PIL import Image, ImageEnhance, ImageFilter

Image.MAX_IMAGE_PIXELS = 700_000_000
import csv
import string

from utils import *


# Added from few_shot_finetuning.py for balanced sampling during training
class CLIP_BalancedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size


def append_to_csv(df, filepath):
    """
    Appends a DataFrame to a CSV file, or creates a new file if it doesn't exist.
    """
    if not os.path.exists(filepath):
        df.to_csv(filepath, index=False, sep=",")
    else:
        df.to_csv(filepath, mode="a", header=False, index=False, sep=",")


class ImageFolderCustom(torch.utils.data.Dataset):
    """
    Custom Dataset for loading images from a directory and assigning category labels.
    Limits the number of samples per category if specified.
    """

    def __init__(self, targ_dir: str, max_category_samples: int | None, img_size: int, preprocess_fn=None,
                 ignore_dir: str = None) -> None:
        self.targ_dir = targ_dir
        self.max_category_samples = max_category_samples
        self.preprocess = preprocess_fn
        self.size = img_size
        self.classes = []
        self.class_to_idx = {}
        self.paths = []  # Will store image_path
        self.targets = []  # Will store class_idx for compatibility with ImageFolder.targets

        # Logic to find classes and limit samples per category
        all_categories = sorted(entry.name for entry in os.scandir(targ_dir) if entry.is_dir())

        if not all_categories:
            raise FileNotFoundError(f"Couldn't find any classes in {targ_dir}.")

        # Create class_to_idx mapping
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(all_categories)}
        self.classes = all_categories

        print(f"Discovered categories: {self.classes}")

        # Collect and filter image paths per category
        for category_name in self.classes:
            category_path = Path(targ_dir) / category_name
            all_files_in_category = list(category_path.glob("*.png"))  # Assuming .png as per original

            if ignore_dir is not None and os.path.exists(ignore_dir):
                ignored_category = Path(ignore_dir) / category_name
                ignored_files = list(ignored_category.glob("*.png"))
                all_files_in_category = [f for f in all_files_in_category if f not in ignored_files]

            # Shuffle and limit files if max_category_samples is set
            if self.max_category_samples is not None and len(all_files_in_category) > self.max_category_samples:
                import random
                random.seed(42)  # Ensure reproducibility if needed
                random.shuffle(all_files_in_category)
                all_files_in_category = all_files_in_category[:self.max_category_samples]

            class_idx = self.class_to_idx[category_name]
            for file_path in all_files_in_category:
                self.paths.append(file_path)
                self.targets.append(class_idx)  # Populate targets list

        (paths, _, labels, _) = train_test_split(np.array(self.paths),
                                                 np.array(self.targets),
                                                 test_size=0.1,
                                                 random_state=42,
                                                 stratify=np.array(
                                                     self.targets))

        print(f"Total images collected: {len(self.paths)}")

    def load_image(self, index: int) -> Image.Image:
        "opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self) -> int:
        "returns the total number of samples"
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        "returns one sample of data, data and label(X,y)"

        try:
            img = self.load_image(index)
            # Ensure image is loaded eagerly to catch OSError here
            img.load()

            # Get class index from pre-calculated targets
            class_idx = self.targets[index]

            # Transform if necessary
            if self.preprocess:
                transformed_img = self.preprocess(img)
            else:
                transformed_img = img  # This might be a PIL Image if no transform

            return transformed_img, class_idx
        except OSError as e:
            print(f"Skipping image at path {self.paths[index]} due to error: {e}")
            # Return a dummy item. Dummy image size should match expected input for CLIP model (e.g., 3, 336, 336 for ViT-L/14@336px)
            # dummy_image = torch.zeros(3, 336, 336)
            dummy_image = torch.zeros(3, self.size, self.size)
            dummy_label = 0
            return dummy_image, dummy_label


class CLIP:
    """
    CLIP model wrapper for fine-tuning, predictions, and evaluation with custom datasets.
    """

    def __init__(self, max_category_samples: int | None, eval_max_category_samples: int | None,
                 top_N: int, model_name: str, device, categories_tsv: str, categories_dir: str, out: str = None,
                 cat_prefix: str = None, avg: bool = True, zero_shot: bool = False):
        self.upper_category_limit = max_category_samples
        self.upper_category_limit_eval = eval_max_category_samples
        self.top_N = top_N
        self.seed = 42
        self.avg = avg
        self.device = device
        self.zero_shot = zero_shot

        self.output_dir = "results" if out is None else out
        self.download_root = '/lnet/work/projects/atrium/cache/clip'

        # Must set jit=False for training
        self.model, self.preprocess = clip.load(model_name, device=device,
                                                download_root=self.download_root, jit=False)

        image_size = (self.preprocess.transforms[0].size, self.preprocess.transforms[0].size)
        image_mean = self.preprocess.transforms[-1].mean
        image_std = self.preprocess.transforms[-1].std

        # Define transformations
        self.train_transforms = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.5),
                transforms.ColorJitter(contrast=0.5),
                transforms.ColorJitter(saturation=0.5),
                transforms.ColorJitter(hue=0.5),
                transforms.Lambda(lambda img: ImageEnhance.Sharpness(img).enhance(random.uniform(0.5, 1.5))),
                transforms.Lambda(lambda img: img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 2))))
            ], p=0.5),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean, std=image_std)
        ])

        self.eval_transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean, std=image_std)
        ])

        if self.avg:
            loaded_cats = load_categories(categories_tsv, prefix=cat_prefix, directory=categories_dir)
            self.categories = sorted(loaded_cats.keys())
            self.class_to_idx = {cat: i for i, cat in enumerate(self.categories)}
            self.texts = loaded_cats  # dict of lists
            print(f"Categories: {self.categories} with multiple descriptions per category.")
            self.text_features = self._get_averaged_text_features()
            for cat, descs in self.texts.items():
                print(f"Category: {cat}, Descriptions:")
                for desc in descs:
                    print(f"  - {desc}")
        else:
            loaded_cats = load_categories(categories_tsv, directory=categories_dir)
            if type(loaded_cats) is dict:
                self.categories = sorted(loaded_cats.keys())
                self.class_to_idx = {cat: i for i, cat in enumerate(self.categories)}
                self.texts = loaded_cats  # dict of lists
                print(f"Categories: {self.categories} with multiple descriptions per category.")
                self.text_features = self._get_averaged_text_features()
                for cat, descs in self.texts.items():
                    print(f"Category: {cat}, Descriptions:")
                    for desc in descs:
                        print(f"  - {desc}")
                self.avg = True
            else:
                self.categories = [label for label, desc in loaded_cats]
                self.texts = [desc for label, desc in loaded_cats]
                self.text_inputs = torch.cat([clip.tokenize(f"a scan of {description}") for description in self.texts]).to(
                        device)
                print(f"Categories: {self.categories} with single description per category.")

        self.model_name = model_name

        self.num_prediction_classes = len(self.categories)
        print(f"Number of prediction classes: {self.num_prediction_classes}")
        print(f"Model name: {self.model_name}")

    def _get_averaged_text_features(self):
        """Computes averaged text features for each category."""
        all_features = []
        with torch.no_grad():
            for category in self.categories:
                descriptions = [f"a scan of {desc}" for desc in self.texts[category]]
                tokens = torch.cat([clip.tokenize(desc) for desc in descriptions]).to(self.device)
                features = self.model.encode_text(tokens)
                features /= features.norm(dim=-1, keepdim=True)
                mean_features = features.mean(dim=0)
                mean_features /= mean_features.norm()
                all_features.append(mean_features)
        return torch.stack(all_features)

    def train(self, train_dir: str, eval_dir: str, log_dir: str, num_epochs: int = 10, batch_size: int = 8,
              learning_rate: float = 1e-7, save_interval: int = 1):
        """
        Fine-tunes the CLIP model based on the provided training and evaluation directories.
        """
        print("Starting CLIP model fine-tuning...")
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        remove_punctuation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        model_name_sanitized = self.model_name.translate(remove_punctuation).replace(" ", "")

        print(f"Current model name: \t{model_name_sanitized}")

        def convert_models_to_fp32(model):
            for p in model.parameters():
                if p.grad is not None:
                    p.data = p.data.float()
                    p.grad.data = p.grad.data.float()

        if self.device == "cpu":
            self.model.float()
        else:
            clip.model.convert_weights(self.model)

        writer = SummaryWriter(log_dir=log_dir)
        weights_path = Path("model_checkpoints")
        weights_path.mkdir(exist_ok=True)

        # Data Loaders
        train_dataset = ImageFolderCustom(train_dir, max_category_samples=self.upper_category_limit,
                                          preprocess_fn=self.preprocess, ignore_dir=eval_dir,
                                          img_size=self.preprocess.transforms[0].size)
        train_labels = torch.tensor(train_dataset.targets)
        train_sampler = CLIP_BalancedBatchSampler(train_labels, batch_size, 1)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler)

        test_dataset = ImageFolderCustom(eval_dir, max_category_samples=self.upper_category_limit_eval,
                                         preprocess_fn=self.preprocess, img_size=self.preprocess.transforms[0].size)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

        loss_img = torch.nn.CrossEntropyLoss()
        loss_txt = torch.nn.CrossEntropyLoss()

        num_batches_train = len(train_dataloader)

        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * num_batches_train,
                                                               eta_min=1e-10)

        print(f"Number of training batches: \t{num_batches_train}")
        print(f"Number of evaluation batches: \t{len(test_dataloader)}")

        ever_best_accuracy = 0.0

        for epoch in range(num_epochs):
            print(f"Epoch: {epoch}/{num_epochs}")
            epoch_train_loss, step = 0, 0
            self.model.train()
            for batch in tqdm(train_dataloader, total=num_batches_train, desc="Training"):
                step += 1
                optimizer.zero_grad()

                images, class_ids = batch
                images = images.to(self.device)

                if self.avg:
                    # Encode images
                    image_features = self.model.encode_image(images)
                    text_features = torch.stack([self.text_features[label_id] for label_id in class_ids]).to(
                        self.device)

                    # Normalize features
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                    # Calculate logits
                    logit_scale = self.model.logit_scale.exp()
                    logits_per_image = logit_scale * image_features @ text_features.T.to(image_features.dtype)
                    logits_per_text = logits_per_image.T

                else:
                    texts = [f"a scan of {self.texts[label_id]}" for label_id in class_ids]
                    texts = clip.tokenize(texts).to(self.device)
                    logits_per_image, logits_per_text = self.model(images, texts)

                ground_truth = torch.arange(len(images), dtype=torch.long, device=self.device)

                total_train_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text,
                                                                                        ground_truth)) / 2
                total_train_loss.backward()
                epoch_train_loss += total_train_loss.item()

                torch.nn.utils.clip_grad_norm_(params, 1.0)

                if self.device != "cpu":
                    convert_models_to_fp32(self.model)
                optimizer.step()
                if self.device != "cpu":
                    clip.model.convert_weights(self.model)
                scheduler.step()

                if step % 25 == 0:
                    print(
                        f"Step {step}/{num_batches_train}, Loss: {total_train_loss.item():.4f}, lr: {optimizer.param_groups[0]['lr']:.10f}")

            avg_train_loss = epoch_train_loss / num_batches_train
            writer.add_scalar("Loss/train", avg_train_loss, epoch)
            writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)
            print(f"{model_name_sanitized}\t Epoch {epoch} train loss: {avg_train_loss:.4f}")

            if epoch == num_epochs - 1:
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_train_loss,
                    },
                    weights_path / f"model_{model_name_sanitized}_{self.upper_category_limit}c_{str(learning_rate)}_{num_epochs}e.pt")
                print(
                    f"Saved weights to {weights_path}/model_{model_name_sanitized}_{self.upper_category_limit}c_{str(learning_rate)}_{num_epochs}e.pt.")

            # Evaluation
            self.model.eval()

            # Fix: Use the correct number of classes for torchmetrics
            if self.avg:
                num_classes = len(self.categories)  # Use self.categories when avg=True
            else:
                num_classes = len(test_dataset.classes)  # Use dataset classes when avg=False

            acc_top1_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(self.device)
            acc_top5_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=5).to(self.device)

            # For evaluation, always use ALL pre-computed text features
            if self.avg:
                text_features_eval = self.text_features  # Use the pre-computed full set of averaged text features
                text_features_eval = text_features_eval / text_features_eval.norm(dim=-1,
                                                                                  keepdim=True)  # Ensure normalized
            else:
                # Original logic for non-averaged categories
                all_texts = torch.cat([clip.tokenize(f"a scan of {c}") for c in self.texts]).to(self.device)
                with torch.no_grad():
                    text_features_eval = self.model.encode_text(all_texts)
                    text_features_eval = text_features_eval / text_features_eval.norm(dim=-1, keepdim=True)

            with torch.no_grad():
                for batch in tqdm(test_dataloader, desc="Evaluating"):
                    images, class_ids = batch
                    images, class_ids = images.to(self.device), class_ids.to(self.device)

                    image_features = self.model.encode_image(images)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                    similarity = (100.0 * image_features @ text_features_eval.T)

                    acc_top1_metric.update(similarity, class_ids)
                    acc_top5_metric.update(similarity, class_ids)

            mean_top1_accuracy = acc_top1_metric.compute()
            mean_top5_accuracy = acc_top5_metric.compute()

            if mean_top1_accuracy > ever_best_accuracy:
                ever_best_accuracy = mean_top1_accuracy
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_train_loss,
                    },
                    weights_path / f"model_{model_name_sanitized}_{self.upper_category_limit}c_{str(learning_rate)}_cp.pt")
                print(
                    f"Saved checkpoint weights to {weights_path}/model_{model_name_sanitized}_{self.upper_category_limit}c_{str(learning_rate)}_cp.pt.")

            print(f"Mean Top 1 Accuracy: {mean_top1_accuracy.item() * 100:.2f}%.")
            print(f"Mean Top 5 Accuracy: {mean_top5_accuracy.item() * 100:.2f}%.")
            writer.add_scalar("Test Accuracy/Top1", mean_top1_accuracy, epoch)
            writer.add_scalar("Test Accuracy/Top5", mean_top5_accuracy, epoch)

            acc_top1_metric.reset()
            acc_top5_metric.reset()

        writer.flush()
        writer.close()
        print("Fine-tuning finished.")

        self.test(test_dataloader)

    def evaluate_saved_model(self, model_path: str, eval_dir: str, batch_size: int = 8):
        """
        Loads a saved model and evaluates its performance on the specified evaluation directory.
        """
        if model_path is not None:
            print(f"Loading model from {model_path} for evaluation...")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}.")

        eval_dataset = ImageFolderCustom(eval_dir, max_category_samples=self.upper_category_limit_eval,
                                         preprocess_fn=self.preprocess, img_size=self.preprocess.transforms[0].size)
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size)

        print("Starting evaluation of the loaded model...")
        self.test(eval_dataloader, image_files=eval_dataset.paths,)
        print("Evaluation finished.")

    def top_N_prediction(self, image_data: torch.Tensor, N: int):
        """
        Predicts the top N categories for a given image tensor.
        :param image_data: A tensor of shape (1, 3, H, W) representing the image.
        :param N: The number of top categories to return.
        """
        image_data = image_data.to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_data)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            if self.avg or self.zero_shot:
                # Use pre-computed averaged features for efficiency
                text_features = self.text_features
                similarity = (100.0 * image_features @ text_features.T)
                probs = similarity.softmax(dim=-1).cpu().numpy()
            else:
                logits_per_image, _ = self.model(image_data, self.text_inputs)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        pred_scores = probs[0]
        best_n_indices = np.argsort(pred_scores)[-N:][::-1]
        best_n_scores = pred_scores[best_n_indices]
        return best_n_scores, best_n_indices, pred_scores

    def prediction(self, image_data: torch.Tensor) -> (np.array, int):
        """
        Predicts the top N categories for a single image tensor and returns the scores and indices.
        :param image_data:
        :return:
        """
        # This method is less used, but we'll align it.
        scores, indices, _ = self.top_N_prediction(image_data.unsqueeze(0), len(self.categories))
        return scores, indices[0]

    def test(self, test_dataloader: torch.utils.data.DataLoader, model_name_sanitized: str = None,
             vis: bool = True, tab: bool = True, image_files: list = []):
        """
        Evaluates the model on the provided test dataloader and generates a confusion matrix plot.
        :param test_dataloader:
        :return:
        """
        remove_punctuation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        model_name_sanitized = self.model_name.translate(remove_punctuation).replace(" ", "") if model_name_sanitized is None else model_name_sanitized

        plot_path = Path(f'{self.output_dir}/plots')
        table_path = Path(f'{self.output_dir}/tables')
        plot_path.mkdir(parents=True, exist_ok=True)
        time_stamp = time.strftime("%Y%m%d-%H%M")
        plot_image = plot_path / f'EVAL_conf_{self.top_N}n_{self.upper_category_limit}c_{model_name_sanitized}_{time_stamp}.png'
        table_file = table_path / f'EVAL_table_{self.top_N}n_{self.upper_category_limit}c_{model_name_sanitized}_{time_stamp}.csv'

        all_predictions = []
        all_true_labels = []

        self.model.eval()

        if self.avg:
            text_features_test = self.text_features  # Use the pre-computed full set of averaged text features
            text_features_test /= text_features_test.norm(dim=-1, keepdim=True)  # Ensure normalized
        else:
            # Original logic for non-averaged categories
            # The `test_dataloader.dataset.texts` is not directly accessible if not `self.avg`.
            # Instead, use the `self.texts` which holds all descriptions.
            all_texts = torch.cat(
                [clip.tokenize(f"a scan of {c}") for c in self.texts]).to(self.device)
            with torch.no_grad():
                text_features_test = self.model.encode_text(all_texts)
                text_features_test /= text_features_test.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            for images, class_ids in tqdm(test_dataloader, desc="Testing"):
                images = images.to(self.device)
                image_features = self.model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                similarity = (100.0 * image_features @ text_features_test.T)

                _, predicted_labels = similarity.max(dim=1)

                all_predictions.extend(predicted_labels.cpu().numpy())
                all_true_labels.extend(class_ids.cpu().numpy())

        acc = round(100 * np.sum(np.array(all_predictions) == np.array(all_true_labels)) / len(all_true_labels), 2)
        print('Accuracy: ', acc)

        if vis:
            # Ensure display labels match the order of predictions
            display_labels = self.categories if self.avg else test_dataloader.dataset.classes

            disp = ConfusionMatrixDisplay.from_predictions(
                np.array(all_true_labels), np.array(all_predictions), cmap='inferno',
                normalize="true", display_labels=np.array(display_labels)
            )

            tick_positions = disp.ax_.get_xticks()
            short_labels = [f"{label[0]}{label.split('_')[-1][0] if '_' in label else ''}" for label in disp.display_labels]
            disp.ax_.set_xticks(tick_positions)
            disp.ax_.set_xticklabels(short_labels)

            disp.ax_.set_title(f"TOP {self.top_N} {self.upper_category_limit_eval}c {self.model_name} CM")
            plt.savefig(plot_image, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Confusion matrix saved to {plot_image}")

        if tab:
            out_df, _ = dataframe_results(image_files, all_predictions, self.categories, raw_scores= None,
                                  top_N=self.top_N)

            print(out_df)
            print(out_df.size)
            print(len(all_true_labels))
            out_df["TRUE"] = [self.categories[i] for i in all_true_labels]
            out_df.sort_values(['FILE', 'PAGE'], ascending=[True, True], inplace=True)
            out_df.to_csv(table_file, sep=",", index=False)
            print(f"Results for TOP-{self.top_N} predictions are recorded into {self.output_dir}/tables/ directory")

        return acc

    def predict_single(self, image_file: str) -> str:
        """
        Predicts the category of a single image file.
        :param image_file:
        :return:
        """
        image = Image.open(image_file)
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        _, best_n_indices, _ = self.top_N_prediction(image_input, self.top_N)
        pred_label = self.categories[best_n_indices[0]]
        return pred_label

    def predict_top(self, image_file: str) -> (list, list):
        """
        Predicts the TOP-N categories of a single image file.
        :param image_file:
        :return:
        """
        image = Image.open(image_file)
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        best_n_scores, best_n_indices, _ = self.top_N_prediction(image_input, self.top_N)
        best_n_scores = np.round(best_n_scores, 3).tolist()
        pred_labels = [self.categories[i] for i in best_n_indices]
        return best_n_scores, pred_labels

    def predict_directory(self, folder_path: str, raw: bool = False, out_table: str = None):
        """
        Predicts categories for all images in a directory and saves results to a CSV file.
        :param folder_path:
        :param raw:
        :param out_table:
        :return:
        """
        images = directory_scraper(Path(folder_path), "png")
        print(f"Predicting {len(images)} images from {folder_path}")

        time_stamp = time.strftime("%Y%m%d-%H%M")  # for results files

        res_list, raw_list, tru_images = [], [], []

        for img_path in tqdm(images, desc="Predicting directory"):
            try:
                image = Image.open(img_path)
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                scores, indices, raw_scores = self.top_N_prediction(image_input, self.top_N)
                res_list.append(indices)
                if raw:
                    raw_list.append(raw_scores.tolist())

                tru_images.append(img_path.name)

            except Exception as e:
                print(f"Error processing file {img_path}: {e}")

        res_list = np.concatenate(res_list, axis=0)
        # print(res_list)
        out_df, raw_df = dataframe_results(test_images=tru_images, test_predictions=res_list, raw_scores=raw_list,
                                           top_N=self.top_N, categories=self.categories)

        out_df.sort_values(['FILE', 'PAGE'], ascending=[True, True], inplace=True)
        out_df.to_csv(out_table, sep=",", index=False)
        print(f"Results for TOP-{self.top_N} predictions are recorded into {self.output_dir}/tables/ directory")

        if raw:
            raw_df.sort_values(self.categories, ascending=[False] * len(self.categories), inplace=True)
            raw_df.to_csv(f"{self.output_dir}/tables/{time_stamp}_{self.model_name.replace('/', '')}_RAW.csv", sep=",", index=False)
            print(f"RAW Results are recorded into {self.output_dir}/tables/ directory")


def load_categories(tsv_file, directory = None, prefix=None):
    """
    Loads categories and descriptions from TSV files for the CLIP model.
    If prefix is provided, it loads all files starting with that prefix from the directory.
    """
    categories_data = defaultdict(list)

    dir = directory if directory else "category_descriptions"
    base_dir = Path(__file__).parent / dir  # directory of tables
    if not base_dir.is_dir():
        print(f"Error: {base_dir} not a directory")
        return {}

    if prefix:
        files_to_load = list(base_dir.glob(f"{prefix}*.tsv")) + list(base_dir.glob(f"{prefix}*.csv"))
        if not files_to_load:
            print(f"Warning: No TSV files with prefix '{prefix}' found in {base_dir}. Using default categories.")
            return {"DRAW": ["a drawing"], "PHOTO": ["a photo"], "TEXT": ["text"], "LINE": ["a table"]}

        print(f"Found {len(files_to_load)} category files with prefix '{prefix}'.")
        for file_path in files_to_load:
            try:
                with open(file_path, "r") as file:
                    reader = csv.DictReader(file, delimiter="\t") if file_path.suffix == '.tsv' else csv.DictReader(file, delimiter=",")
                    for row in reader:
                        categories_data[row["label"].replace("+AF8-", "_")].append(row["description"])
            except Exception as e:
                print(f"Error reading categories file {file_path}: {e}")

        # for cat, descs in categories_data.items():
        #     print(f"Category: {cat}, Descriptions: {descs}")
        return categories_data

    else:  # Original behavior
        categories = []
        tsv_file_or_dir = base_dir / tsv_file

        if not os.path.exists(tsv_file_or_dir):
            print(f"Warning: Categories file not found at {tsv_file_or_dir}. Using default categories.")
            return [("DRAW", "a drawing"), ("PHOTO", "a photo"), ("TEXT", "text"), ("LINE", "a table")]
        try:
            with open(tsv_file_or_dir, "r") as file:
                reader = csv.DictReader(file, delimiter="\t") if tsv_file_or_dir.suffix == '.tsv' else csv.DictReader(file, delimiter=",")
                for row in reader:
                    categories.append((row["label"].replace("+AF8-", "_"), row["description"]))

            uniques_categs = set([cat for cat, _ in categories])
            if len(categories) > len(uniques_categs):
                categories_data = defaultdict(list)
                for cat, desc in categories:
                    categories_data[cat].append(desc)
                categories = {cat: descs for cat, descs in categories_data.items()}
        except Exception as e:
            print(f"Error reading categories file: {e}")
            return []  # Fallback
        print(f"Loaded {len(categories)} categories from {tsv_file_or_dir}")
        return categories

def evaluate_multiple_models(model_dir: str, eval_dir: str, device, cat_prefix: str, model_suffix: str = "05.pt", vis: bool = True,
                                 batch_size: int = 8, zero_shot: bool = False):
    """
    Evaluates multiple saved models in a directory and records their Top-1 accuracy.
    :param model_dir: Directory containing the saved model files.
    :param eval_dir: Directory for evaluation data.
    :param model_suffix: Suffix to filter model files (e.g., ".pt", "_cp.pt").
    :param vis: If True, visualize results in a bar graph.
    :param batch_size: Batch size for evaluation.
    """
    map_base_name = {
        "ViTB32_": "ViT-B/32",
        "ViTB16_": "ViT-B/16",
        "ViTL14_": "ViT-L/14",
        "ViTL14336px_": "ViT-L/14@336px",
    }

    category_sufix = {
        "000c": "average",
        "01c": "detail",  # 9
        "02c": "extra",  # 8
        "03c": "gemini",  # 6
        "04c": "gpt",  # 4
        "05c": "mid",  # 2
        "06c": "min",  # 3
        "07c": "short",  # 5
        "08c": "init",  # 1
    }

    model_dir_path = Path(model_dir)
    if not model_dir_path.is_dir():
        print(f"Error: Model directory not found at {model_dir}")
        return

    model_files = list(model_dir_path.rglob(f"*{model_suffix}"))
    if not model_files:
        print(f"No model files found with suffix '{model_suffix}' in {model_dir}")
        return

    print(f"Found {len(model_files)} models with suffix '{model_suffix}'.")
    accuracies = {}  # To store filename_stem: top1_accuracy

    cur = Path(__file__).parent if '__file__' in globals() else Path.cwd()
    output_dir = cur / "results"
    output_dir.mkdir(exist_ok=True)

    if zero_shot:
        print("Zero-shot evaluation mode enabled. Using pre-computed text features.")
        for base_name in map_base_name.values():
            print(f"Using base model: {base_name}")
            vis_model_name = f"{base_name} zero"

            try:
                clip_instance = CLIP(None, None, 1, base_name, device,
                                     cat_prefix, str(output_dir), cat_prefix, True, False)

                # Prepare evaluation dataset and dataloader once
                eval_dataset = ImageFolderCustom(eval_dir, max_category_samples=None,
                                                 preprocess_fn=clip_instance.preprocess,
                                                 img_size=clip_instance.preprocess.transforms[0].size)
                eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size)

                accuracies[vis_model_name] = clip_instance.test(eval_dataloader, vis=False)
                print(f"Top 1 Accuracy for {vis_model_name} {base_name}: {accuracies[vis_model_name]:.2f}%")
            except Exception as e:
                print(f"Error evaluating model {base_name}: {e}")
                if vis_model_name not in accuracies.keys():
                    accuracies[vis_model_name] = "Error"  # Indicate error
    else:
        for model_path in sorted(model_files):
            model_name_stem = model_path.stem
            print(f"\nEvaluating model: {model_name_stem}")

            base_name = None
            for short, full in map_base_name.items():
                if short in model_name_stem:
                    base_name = full
                    break

            vis_categ = "UNK"  # Default category
            for code, categ in category_sufix.items():
                if code in model_name_stem:
                    vis_categ = categ
                    break

            if base_name is None:
                vis_model_name = f"{model_name_stem} {vis_categ}"
                accuracies[vis_model_name] = "Error"  # Indicate error
            else:
                vis_model_name = f"{base_name.replace('@', '-')} {vis_categ}"
                print(vis_model_name)
                try:
                    # Load model state dict
                    clip_instance = CLIP(None, None, 1, base_name, device,
                                         cat_prefix, str(output_dir), cat_prefix, True, False)

                    # Prepare evaluation dataset and dataloader once
                    eval_dataset = ImageFolderCustom(eval_dir, max_category_samples=None,
                                                     preprocess_fn=clip_instance.preprocess,
                                                     img_size=clip_instance.preprocess.transforms[0].size)
                    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size)

                    checkpoint = torch.load(model_path, map_location=device)
                    clip_instance.model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Model loaded from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}.")

                    accuracies[vis_model_name] = clip_instance.test(eval_dataloader, vis=False)
                    print(f"Top 1 Accuracy for {vis_model_name} {model_name_stem}: {accuracies[vis_model_name]:.2f}%")

                except Exception as e:
                    print(f"Error evaluating model {model_name_stem}: {e}")
                    if vis_model_name not in accuracies.keys():
                        accuracies[vis_model_name] = "Error"  # Indicate error

    # Save results to a table
    if accuracies:
        results_df = pd.DataFrame(list(accuracies.items()), columns=['model_name', 'accuracy'])

        # Create a dedicated directory for evaluation statistics
        eval_stats_output_dir = Path(output_dir) / 'stats'
        eval_stats_output_dir.mkdir(parents=True, exist_ok=True)

        csv_output_path = eval_stats_output_dir / f"model_accuracies{'_zero' if zero_shot else ''}.csv"
        results_df.to_csv(csv_output_path, index=False)
        print(f"Evaluation results saved to {csv_output_path}")

        if vis:
            # sort results by vis order
            visualize_results(str(csv_output_path), str(Path(output_dir) / 'stats'), zero_shot)
    else:
        print("No models were successfully evaluated.")

def visualize_results(csv_file: str, output_dir: str, zero_shot: bool = False):
    """
    Generate a bar plot from a CSV file of model accuracies.

    :param csv_file: Path to the CSV file containing model accuracies.
    :param output_dir: Directory where the plot will be saved.
    :param vis_orders: Dictionary to define custom sorting order.
    :param base_model_colors: Dictionary to map base model names to specific colors.
    """
    base_model_colors = {
        "ViT-B/32 ": "steelblue",
        "ViT-B/16 ": "indigo",
        "ViT-L/14 ": "orange",
        "ViT-L/14-336 ": "gold"
    }

    category_codes = {
        "average": 10,
        "detail": 9,  # 9
        "extra": 8,  # 8
        "gemini": 6,  # 6
        "gpt": 4,  # 4
        "mid": 2,  # 2
        "min": 3,  # 3
        "short": 5,  # 5
        "init": 1,  # 1
    }

    vis_order = {}

    results_df = pd.read_csv(csv_file)

    for vis_model_name in results_df['model_name'].tolist():
        for code, order in category_codes.items():
            if code in vis_model_name:
                vis_order[vis_model_name] = order
                break

    # Load the CSV into a DataFrame

    if not zero_shot:
        # Apply custom sorting based on vis_orders
        results_df['vis_order'] = results_df['model_name'].apply(lambda x: vis_order.get(x, 0))
        results_df.sort_values(by='vis_order', inplace=True, ascending=True)
        results_df.drop(columns='vis_order', inplace=True)

    # Assign colors based on base model
    results_df['color'] = results_df['model_name'].apply(
        lambda x: next((color for base, color in base_model_colors.items() if base in x), 'black')
    )

    # Generate the bar plot
    plt.figure(figsize=(12, 7))
    plt.bar(results_df['model_name'], results_df['accuracy'], color=results_df['color'])
    plt.xlabel("Model Name")
    plt.ylabel("Top-1 Accuracy (%)")
    plt.title("Model Accuracy Comparison")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # set min-max y-axis values
    plt.ylim(results_df['accuracy'].min()-1, 100 if results_df['accuracy'].max() == 100 else results_df['accuracy'].max()+1)

    # Save the plot
    plot_output_dir = Path(output_dir)
    plot_output_dir.mkdir(parents=True, exist_ok=True)
    plot_output_path = plot_output_dir / f"model_accuracy_plot{'_zero' if zero_shot else ''}.png"
    plt.savefig(plot_output_path, dpi=300)
    plt.close()
    print(f"Accuracy plot saved to {plot_output_path}")
