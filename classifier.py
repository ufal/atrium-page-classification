import datetime

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import transformers

from utils import *

from sklearn.model_selection import train_test_split, cross_val_score

from PIL import ImageEnhance, ImageFilter
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer, default_data_collator
from huggingface_hub import whoami
# import timm
# from timm.data import resolve_data_config
# from transformers import TimmWrapperModel, TimmWrapperImageProcessor

import random
from collections import defaultdict, OrderedDict
import numpy as np

import torch
import numpy as np
from torch.utils.data import DataLoader


def custom_collate(batch: list) -> dict:
    """
    Custom collate function to filter out None entries from the batch
    and efficiently handle one-hot encoded labels.
    """
    # Filter out None entries
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None, None

    # Extract pixel_values and labels separately
    pixel_values = []
    labels = []

    for item in batch:
        pixel_values.append(item['pixel_values'])
        if item['label'] is not None:
            labels.append(item['label'])

    # Stack pixel_values (assuming they're already tensors from transforms)
    if len(pixel_values) > 0:
        pixel_values = torch.stack(pixel_values)
    else:
        pixel_values = torch.tensor([])

    # Convert one-hot encoded labels efficiently
    if len(labels) > 0:
        # Handle one-hot encoded labels
        processed_labels = []

        for label in labels:
            if isinstance(label, torch.Tensor):
                processed_labels.append(label)
            elif isinstance(label, np.ndarray):
                # Convert numpy array to tensor
                processed_labels.append(torch.from_numpy(label))
            elif isinstance(label, (list, tuple)):
                # Convert list/tuple to tensor
                processed_labels.append(torch.tensor(label))
            else:
                # Single value - assume it's a class index, convert to one-hot if needed
                processed_labels.append(torch.tensor(label))

        # Stack all labels into a single tensor
        # This creates a 2D tensor where each row is a one-hot vector
        labels = torch.stack(processed_labels).float()  # Use float for one-hot encodings
    else:
        labels = torch.tensor([])

    return {
        'pixel_values': pixel_values,
        'labels': labels
    }


# Added from few_shot_finetuning.py for balanced sampling during training
class BalancedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels_one_hot: np.array , n_classes_per_batch, n_samples_of_class):
        self.labels = labels_one_hot
        self.labels_set = list(np.unique(self.labels.argmax(axis=-1))) # Modified for one-hot
        self.label_to_indices = {label: np.where(self.labels.argmax(axis=-1) == label)[0]  # Modified for one-hot
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes_per_batch
        self.n_samples = n_samples_of_class
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



class ImageClassifier:
    def __init__(self, checkpoint: str, num_labels: int, store_dir: str = "/lnet/work/projects/atrium/transformers/local/chekcpoint"):
        """
        Initialize the image classifier with the specified checkpoint.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = checkpoint

        if checkpoint.startswith("timm"):
            self.model = AutoModelForImageClassification.from_pretrained(
                checkpoint,
                num_labels=num_labels,
                cache_dir=store_dir,
                ignore_mismatched_sizes=True,
            ).to(self.device)

            # print(self.model.config)

            image_size = self.model.config.pretrained_cfg["input_size"][-1]  # For timm models, input_size is [batch_size, channels, height, width]
            image_mean = self.model.config.pretrained_cfg["mean"]
            image_std = self.model.config.pretrained_cfg["std"]
            self.processor = None

        else:
            self.model = AutoModelForImageClassification.from_pretrained(
                checkpoint,
                num_labels=num_labels,
                cache_dir=store_dir,
                ignore_mismatched_sizes=True,
            ).to(self.device)
            self.processor = AutoImageProcessor.from_pretrained(checkpoint)
            image_size = self.processor.size['height']
            image_mean = self.processor.image_mean
            image_std = self.processor.image_std

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
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean, std=image_std)
        ])

        self.eval_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean, std=image_std)
        ])




    def process_images(self, image_paths: list, image_labels: list, batch_size: int, train: bool = True, ignored_paths: list = None) -> DataLoader:
        """
        Process a list of image file paths into batches.
        """
        dataset = ImageDataset(image_paths, image_labels, self.train_transforms if train else self.eval_transforms, ignored_paths=ignored_paths)
        if train:
            dataloader = DataLoader(dataset, collate_fn=custom_collate,
                                batch_sampler=BalancedBatchSampler(image_labels, batch_size, 1))
        else:
            dataloader = DataLoader(dataset, collate_fn=custom_collate,
                                batch_size=batch_size)
       
        print(f"Dataloader of {'train' if train else 'eval'} dataset is ready:\t{len(image_paths)} images split into {len(dataloader)} batches of size {batch_size}")
        return dataloader

    def preprocess_image(self, image_path: str, train: bool = True) -> torch.Tensor:
        """
        Preprocess a single image for training or evaluation.
        """
        image = Image.open(image_path)
        # Check the image mode
        if image.mode != 'RGB':
            # Convert RGBA to RGB
            image_alpha = image.convert('RGBA')
            new_image = Image.new("RGBA", image_alpha.size, "WHITE")  # Create a white rgba background
            new_image.paste(image_alpha, (0, 0),
                            image_alpha)  # Paste the image on the background. Go to the links given below for details.
            image = new_image.convert('RGB')
        transform = self.train_transforms if train else self.eval_transforms
        tensor = transform(image).unsqueeze(0).to(self.device)
        return tensor

    def train_model(self, train_dataloader, eval_dataloader, output_dir: str, out_model: str,
                    num_epochs: int = 3, learning_rate: float = 1e-5, logging_steps: int = 10):
        """
        Train the model using the provided training and evaluation data loaders.
        """
        print(f"Training for {num_epochs} epochs on {len(train_dataloader)} train samples and evaluation on {len(eval_dataloader)} samples")

        # Define optimizer and scheduler
        # For the optimizer, we typically pass model.parameters()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.0001)
        num_batches_train = len(train_dataloader)
        scheduler_poly = transformers.get_scheduler(
            name="polynomial",
            optimizer=optimizer,
            num_warmup_steps=250,
            num_training_steps=num_epochs * num_batches_train,
            scheduler_specific_kwargs={
                "power": 1.0,  # Polynomial decay power
                "lr_end": 1e-10,  # Final learning rate at the end of training
            },
        )
        scheduler_linear = transformers.get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_training_steps=num_epochs * num_batches_train,
            num_warmup_steps=250,
        )
        scheduler_cosine = transformers.get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=250,
            num_training_steps=num_epochs * num_batches_train,
            scheduler_specific_kwargs={
                "num_cycles": 0.5,  # Number of cosine cycles
            }
        )

        # Generate log_dir dynamically, similar to clip_full.py
        current_file_name = os.path.basename(__file__)
        log_dir = "{}-{}-{}".format(
            os.path.basename(current_file_name),
            datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
            "lr={:.1e}".format(learning_rate).replace("-", "_")  # Simplified version
        )
        log_dir += f"-e={num_epochs}-m={self.model_name}-{out_model}"  # Example of adding more info
        log_dir = os.path.join("logs", log_dir.replace(" ", "_").replace("/", "_")).replace(".", "")
        # make sure logdir exists
        cur = Path(__file__).parent
        log_dir = cur / log_dir
        Path(log_dir).mkdir(exist_ok=True, parents=True)

        # Correct way to get the batch size from the BalancedBatchSampler
        if isinstance(train_dataloader.batch_sampler, BalancedBatchSampler):
            effective_train_batch_size = train_dataloader.batch_sampler.batch_size
        else:
            effective_train_batch_size = train_dataloader.batch_size  # Fallback if not using balanced sampler

        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="best",
            learning_rate=learning_rate,
            per_device_train_batch_size=effective_train_batch_size,
            per_device_eval_batch_size=eval_dataloader.batch_size,
            num_train_epochs=num_epochs,
            warmup_ratio=0.1,
            logging_steps=logging_steps,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            push_to_hub=False,
            report_to="tensorboard",
            logging_dir=log_dir,  # Use the dynamically generated log director
            # fp16=True if torch.cuda.is_available() else False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataloader.dataset,
            eval_dataset=eval_dataloader.dataset,
            data_collator=lambda data: custom_collate(data),
            compute_metrics=self.compute_metrics,
            #optimizers=(optimizer, scheduler_cosine)  # Pass the optimizer and scheduler
            # optimizer_cls_and_kwargs=(torch.optim.AdamW, {'lr': learning_rate, 'weight_decay': 0.0001}),
        )

        trainer.train()

        self.save_model(f"model/{out_model}")

    def infer(self, image_path: str) -> int:
        """
        Perform inference on a single image.
        """
        self.model.eval()
        with torch.no_grad():
            inputs = self.preprocess_image(image_path, train=False)
            outputs = self.model(pixel_values=inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            return predicted_class_idx

    def top_n_predictions(self, image_path: str, top_n: int = 1) -> list:
        """
        Perform inference and return top-N predictions with normalized probabilities.
        """
        self.model.eval()
        with torch.no_grad():
            inputs = self.preprocess_image(image_path, train=False)
            outputs = self.model(pixel_values=inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            top_n_probs, top_n_indices = torch.topk(probabilities, top_n, dim=-1)
            top_n_probs = top_n_probs / top_n_probs.sum()
        return list(zip(top_n_indices.squeeze().tolist(), top_n_probs.squeeze().tolist()))

    def create_dataloader(self, image_paths: list, batch_size: int, ignored_paths: list = None) -> DataLoader:
        """
        Turn an input list of image paths into a DataLoader without labels.
        """
        dataset = ImageDataset(image_paths, transform=self.eval_transforms, ignored_paths=ignored_paths)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
        print(f"Dataloader of directory dataset is ready:\t{len(image_paths)} images split into {len(dataloader)} batches of size {batch_size}")
        return dataloader

    def infer_dataloader(self, dataloader, top_n: int, raw: bool = False) -> (list, list):
        """
        Perform inference on a DataLoader, optionally with top-N predictions.
        """
        self.model.eval()
        predictions = []
        raw_scores = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['pixel_values']
                outputs = self.model(pixel_values=inputs.to(self.device))
                logits = outputs.logits

                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                raw_scores.extend(probabilities.tolist())
                if top_n > 1:
                    top_n_probs, top_n_indices = torch.topk(probabilities, top_n, dim=-1)
                    for indices, probs in zip(top_n_indices, top_n_probs):
                        top_n_probs_normalized = probs / probs.sum()
                        predictions.append(list(zip(indices.tolist(), top_n_probs_normalized.tolist())))
                else:
                    predicted_class_idx = logits.argmax(-1).tolist()
                    predictions.extend(predicted_class_idx)
                print(f"Processed {len(predictions)} images at {time.strftime('%Y-%m-%d %H:%M:%S')}")

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
        self.model = AutoModelForImageClassification.from_pretrained(load_directory).to(self.device)
        print(f"Model and processor loaded from {load_directory}")

    def push_to_hub(self, load_directory: str, repo_id: str, private: bool = False,
                    token: str = None, revision: str = "main"):
        """
        Upload the fine-tuned model and processor to the Hugging Face Model Hub.

        Args:
            load_directory (str): The directory where the model and processor are stored.
            repo_name (str): The name of the repository to create or update on the Hugging Face Hub.
            organization (str, optional): The organization under which to create the repository. Defaults to None.
            private (bool, optional): Whether the repository should be private. Defaults to False.
            token (str, optional): The authentication token for Hugging Face Hub. Defaults to None.
        """

        # Determine the repository ID
        # username = whoami(token=token)['name']
        # repo_id = f"{username}/{repo_name}"

        # Save the model and processor locally
        self.model.save_pretrained(load_directory)
        self.processor.save_pretrained(load_directory)

        print(f"Model and processor saved locally to {load_directory}, preparing to push "
              f"revision {revision} to the Hub repository {repo_id}...")

        # Upload to the Hub
        self.model.push_to_hub(repo_id, private=private, token=token, revision=revision)
        self.processor.push_to_hub(repo_id, private=private, token=token, revision=revision)

        print(f"Model and processor pushed to the Hugging Face Hub: {repo_id}")

    def load_from_hub(self, repo_id: str,  revision: str = "main"):
        """
        Load a model and its processor from the Hugging Face Hub.

        Args:
            repo_id (str): The name of the repository on the Hugging Face Hub.
            revision (str, optional): The revision of the repository to load. Defaults to "main".

        Returns:
            model: The loaded model.
            processor: The loaded processor.
        """
        print(f"Accessing the Hugging Face Hub repository {repo_id}, revision {revision}...")

        # Load the model from the repository
        model = AutoModelForImageClassification.from_pretrained(repo_id, revision=revision)

        # Load the processor from the repository
        processor = AutoImageProcessor.from_pretrained(repo_id, revision=revision)

        self.model, self.processor = model, processor
        print(f"Model and processor loaded from the Hugging Face Hub: {repo_id}")


    @staticmethod
    def compute_metrics(eval_pred: list) -> dict:
        """
        Compute accuracy metrics for evaluation.
        """
        from evaluate import load
        import numpy as np
        accuracy = load("accuracy")
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        labels = np.argmax(labels, axis=-1)
        return accuracy.compute(predictions=predictions, references=labels)


class ImageDataset(Dataset):
    def __init__(self, image_paths: list, image_labels: list = None, transform=None, ignored_paths: list = None):
        self.image_paths = image_paths
        self.image_labels = image_labels
        self.transform = transform

        if ignored_paths is not None:
            # Filter out ignored paths
            self.image_paths = [path for path in image_paths if path not in ignored_paths]
            if image_labels is not None:
                self.image_labels = [label for path, label in zip(image_paths, image_labels) if path not in ignored_paths]

        self.known = True

        if image_labels is None:
            self.known = False

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path)
            # Check the image mode
            if image.mode != 'RGB':
                # Convert RGBA to RGB
                image_alpha = image.convert('RGBA')
                new_image = Image.new("RGBA", image_alpha.size, "WHITE")
                new_image.paste(image_alpha, (0, 0), image_alpha)
                image = new_image.convert('RGB')
            if self.transform:
                image = self.transform(image)

            # Handle one-hot encoded labels - don't try to convert to scalar
            label = self.image_labels[idx] if self.known else None
            # Keep the one-hot encoding as is - the collate function will handle tensor conversion

            return {'pixel_values': image, 'label': label}
        except Exception as e:
            print(image_path, e)
            return None


def split_data_80_10_10(files: list, labels: list, random_seed: int, max_categ: int, safe_check: bool = True) -> (list, list, list, list, list, list):
    """
    Splits the data into training, validation, and test sets with an 80/10/10 ratio.
    The split uses uniform distribution selection to maintain temporal distribution
    across the sorted files (by creation date). Test and dev sets are selected first,
    with remaining samples going to training.

    Args:
        files: List of file paths (should be sorted alphabetically by creation date)
        labels: List of corresponding labels
        random_seed: Random seed for reproducibility
        max_categ: Maximum number of samples per category to consider
    Returns:
        tuple: (train_files, val_files, test_files, train_labels, val_labels, test_labels)
    """
    # Set random seeds for reproducibility
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Convert to numpy arrays for easier manipulation
    files = np.array(files)
    labels = np.array(labels)

    # Group indices by label for stratified sampling
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label.argmax()].append(idx)


    for label, indices in label_to_indices.items():
        indices = np.array(indices)
        n_samples = len(indices)

        if n_samples > max_categ:
            print(f"Label {label} has {n_samples} samples, limiting to {max_categ}.")
            indices = np.random.choice(indices, size=max_categ, replace=False)

        label_to_indices[label] = indices.tolist()

    total_files = [files[indices] for label in label_to_indices for indices in label_to_indices[label]]
    total_labels = [labels[indices] for label in label_to_indices for indices in label_to_indices[label]]

    if safe_check:
        print(f"Checking {len(total_files)} files for corrupted images...")

        good_files, good_labels = [], []
        for file, label in zip(total_files, total_labels):
            try:
                # try to open that image and load the data
                Image.open(file).load()
                good_files.append(file)
                good_labels.append(label)
            except Exception as e:
                print(f"File {file} is corrupted: {e}")
                continue

        print(f"Total usable images found: {len(good_files)} / {len(total_files)}")
    else:
        good_files, good_labels = total_files, total_labels

    files, labels = np.array(good_files), np.array(good_labels)
    # Group indices by label for stratified sampling
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label.argmax()].append(idx)

    # Initialize result indices
    test_indices = []
    val_indices = []
    train_indices = []

    # For each label, perform stratified uniform sampling
    for label, indices in label_to_indices.items():
        indices = np.array(indices)
        n_samples = len(indices)

        # Calculate sizes for each split
        n_test = max(1, int(n_samples * 0.1))  # At least 1 sample
        n_val = max(1, int(n_samples * 0.1))  # At least 1 sample

        # Ensure we don't exceed available samples
        if n_test + n_val > n_samples:
            n_test = n_samples // 2
            n_val = n_samples - n_test

        # n_train = n_samples - n_test - n_val

        # Use uniform distribution to select indices
        # This preserves the temporal distribution better than random shuffling

        # Select test indices using uniform spacing
        if n_test > 0:
            test_step = n_samples / n_test
            test_positions = np.arange(0, n_samples, test_step)[:n_test]
            # Add small random offset to avoid systematic bias
            test_positions += np.random.uniform(-test_step / 4, test_step / 4, size=len(test_positions))
            test_positions = np.clip(test_positions, 0, n_samples - 1).astype(int)
            selected_test = indices[test_positions]
            test_indices.extend(selected_test)

        # Get remaining indices after test selection
        remaining_mask = np.ones(n_samples, dtype=bool)
        if n_test > 0:
            remaining_mask[test_positions] = False
        remaining_indices = indices[remaining_mask]
        n_remaining = len(remaining_indices)

        # Select validation indices from remaining using uniform spacing
        if n_val > 0 and n_remaining > 0:
            val_step = n_remaining / n_val if n_val <= n_remaining else 1
            val_positions = np.arange(0, n_remaining, val_step)[:n_val]
            if len(val_positions) > n_remaining:
                val_positions = np.arange(n_remaining)
            # Add small random offset
            val_positions += np.random.uniform(-val_step / 4 if val_step > 1 else 0,
                                               val_step / 4 if val_step > 1 else 0,
                                               size=len(val_positions))
            val_positions = np.clip(val_positions, 0, n_remaining - 1).astype(int)
            selected_val = remaining_indices[val_positions]
            val_indices.extend(selected_val)

            # Remaining indices go to training
            val_mask = np.ones(n_remaining, dtype=bool)
            val_mask[val_positions] = False
            train_indices.extend(remaining_indices[val_mask])
        else:
            # If no validation samples, all remaining go to training
            train_indices.extend(remaining_indices)

    # Convert to numpy arrays and sort to maintain some order
    test_indices = np.array(test_indices)
    val_indices = np.array(val_indices)
    train_indices = np.array(train_indices)

    # Extract the corresponding files and labels
    test_files = files[test_indices]
    test_labels = labels[test_indices]

    val_files = files[val_indices]
    val_labels = labels[val_indices]

    train_files = files[train_indices]
    train_labels = labels[train_indices]

    return train_files, val_files, test_files, train_labels, val_labels, test_labels


# Alternative simpler version using numpy's choice with uniform probabilities
def split_data_80_10_10_simple(files: list, labels: list, random_seed: int) -> (list, list, list, list, list, list):
    """
    Simplified version using numpy's uniform random selection with stratification.
    """
    np.random.seed(random_seed)

    files = np.array(files)
    labels = np.array(labels)

    # Get unique labels and their counts
    unique_labels, label_counts = np.unique(labels, return_counts=True)

    test_indices = []
    val_indices = []

    # For each label, select test and validation indices
    for label in unique_labels:
        label_mask = labels == label
        label_indices = np.where(label_mask)[0]
        n_samples = len(label_indices)

        # Calculate split sizes
        n_test = max(1, int(n_samples * 0.1))
        n_val = max(1, int(n_samples * 0.1))

        if n_test + n_val > n_samples:
            n_test = n_samples // 2
            n_val = n_samples - n_test

        # Randomly select test indices (uniform probability for each sample)
        selected_test = np.random.choice(label_indices, size=n_test, replace=False)
        test_indices.extend(selected_test)

        # Select validation indices from remaining samples
        remaining_indices = np.setdiff1d(label_indices, selected_test)
        if len(remaining_indices) >= n_val:
            selected_val = np.random.choice(remaining_indices, size=n_val, replace=False)
            val_indices.extend(selected_val)

    # All remaining indices go to training
    all_indices = np.arange(len(files))
    train_indices = np.setdiff1d(all_indices, np.concatenate([test_indices, val_indices]))

    return (files[train_indices], files[val_indices], files[test_indices],
            labels[train_indices], labels[val_indices], labels[test_indices])


def average_model_weights(model_dir: str, model_name_pattern: str, base_model: str,
                          num_labels: int, output_name: str = None):
    """
    Average the weights of multiple fold models using PyTorch state_dict.

    Args:
        model_dir: Directory containing the fold models
        model_name_pattern: Pattern to match model names (e.g., "model_v4")
        base_model: Base model architecture for loading
        num_labels: Number of output classes
        output_name: Name for the averaged model (defaults to pattern + "_averaged")

    Returns:
        Path to the saved averaged model
    """
    model_dir = Path(model_dir)

    # Find all fold models matching the pattern
    fold_pattern = f"{model_name_pattern}_fold_*"
    fold_dirs = list(model_dir.glob(fold_pattern))

    if not fold_dirs:
        raise ValueError(f"No fold models found matching pattern: {fold_pattern}")

    print(f"Found {len(fold_dirs)} fold models to average:")
    for fold_dir in sorted(fold_dirs):
        print(f"  - {fold_dir.name}")

    # Load the first model to get the architecture and initialize averaged weights
    first_model_path = fold_dirs[0]
    print(f"\nLoading first model from: {first_model_path}")

    # Load model and get its state dict
    first_model = AutoModelForImageClassification.from_pretrained(
        str(first_model_path),
        num_labels=num_labels
    )

    # Initialize averaged state dict with first model's weights
    averaged_state_dict = OrderedDict()
    for key, param in first_model.state_dict().items():
        averaged_state_dict[key] = param.clone().float()  # Convert to float for averaging

    # Add weights from remaining models
    for fold_dir in fold_dirs[1:]:
        print(f"Adding weights from: {fold_dir.name}")

        # Load model
        model = AutoModelForImageClassification.from_pretrained(
            str(fold_dir),
            num_labels=num_labels
        )

        # Add weights to running average
        for key, param in model.state_dict().items():
            if key in averaged_state_dict:
                averaged_state_dict[key] += param.float()
            else:
                print(f"Warning: Key {key} not found in first model, skipping...")

    # Divide by number of models to get average
    num_models = len(fold_dirs)
    for key in averaged_state_dict:
        averaged_state_dict[key] /= num_models

    print(f"\nAveraged weights from {num_models} models")

    # Create new model with averaged weights
    averaged_model = AutoModelForImageClassification.from_pretrained(
        base_model,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )

    # Load the averaged weights
    averaged_model.load_state_dict(averaged_state_dict, strict=False)

    # Save the averaged model
    if output_name is None:
        output_name = f"{model_name_pattern}_averaged"

    output_path = model_dir / output_name
    output_path.mkdir(exist_ok=True)

    print(f"Saving averaged model to: {output_path}")
    averaged_model.save_pretrained(str(output_path))

    # Also save the processor from the first fold (they should all be the same)
    try:
        processor = AutoImageProcessor.from_pretrained(str(first_model_path))
        processor.save_pretrained(str(output_path))
        print("Processor saved successfully")
    except Exception as e:
        print(f"Warning: Could not save processor: {e}")

    return output_path


