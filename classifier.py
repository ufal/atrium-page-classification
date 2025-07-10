import datetime

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import transformers

from utils import *

from sklearn.model_selection import train_test_split, cross_val_score

import PIL
PIL.Image.MAX_IMAGE_PIXELS = 886402639
from PIL import Image, ImageEnhance, ImageFilter
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer, default_data_collator
from huggingface_hub import whoami


def custom_collate(batch: list) -> (torch.Tensor, torch.Tensor):
    """
    Custom collate function to filter out None entries from the batch.
    """
    # Filter out None entries
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None, None
    return default_data_collator(batch)


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
            save_strategy="epoch",
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

        # Save the model and processor locally
        self.model.save_pretrained(load_directory)
        self.processor.save_pretrained(load_directory)

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
                new_image = Image.new("RGBA", image_alpha.size, "WHITE")  # Create a white rgba background
                new_image.paste(image_alpha, (0, 0),
                                image_alpha)  # Paste the image on the background. Go to the links given below for details.
                image = new_image.convert('RGB')
            if self.transform:
                image = self.transform(image)
            return {'pixel_values': image, 'label': self.image_labels[idx] if self.known else None}
        except Exception as e:
            print(image_path, e)
            return None





