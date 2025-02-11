import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils import *

from sklearn.model_selection import train_test_split, cross_val_score

import PIL
PIL.Image.MAX_IMAGE_PIXELS = 886402639
from PIL import Image, ImageEnhance, ImageFilter
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer, default_data_collator


def custom_collate(batch):
    """
    Custom collate function to filter out None entries from the batch.
    """
    # Filter out None entries
    # print(batch)
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None, None
    return default_data_collator(batch)


class ImageClassifier:
    def __init__(self, checkpoint: str, num_labels: int, store_dir: str = "/lnet/work/people/lutsai/pythonProject/OCR/ltp-ocr/trans/chekcpoint"):
        """
        Initialize the image classifier with the specified checkpoint.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = AutoImageProcessor.from_pretrained(checkpoint)
        self.model = AutoModelForImageClassification.from_pretrained(
            checkpoint,
            num_labels=num_labels,
            # id2label=id2label,
            # label2id=label2id,
            cache_dir=store_dir,
            ignore_mismatched_sizes=True
        ).to(self.device)

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
            transforms.Resize((self.processor.size['height'], self.processor.size['width'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.processor.image_mean, std=self.processor.image_std)
        ])

        self.eval_transforms = transforms.Compose([
            transforms.Resize((self.processor.size['height'], self.processor.size['width'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.processor.image_mean, std=self.processor.image_std)
        ])

    def process_images(self, image_paths: list, image_labels: list, batch_size: int, train: bool = True):
        """
        Process a list of image file paths into batches.
        """
        dataset = ImageDataset(image_paths, image_labels, self.train_transforms if train else self.eval_transforms)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train, collate_fn=custom_collate)
        print(f"Dataloader of {'train' if train else 'eval'} dataset is ready:\t{len(image_paths)} images split into {len(dataloader)} batches of size {batch_size}")
        return dataloader

    def preprocess_image(self, image_path: str, train: bool = True):
        """
        Preprocess a single image for training or evaluation.
        """
        image = Image.open(image_path).convert('RGB')
        transform = self.train_transforms if train else self.eval_transforms
        tensor = transform(image).unsqueeze(0).to(self.device)
        return tensor

    def train_model(self, train_dataloader, eval_dataloader, output_dir: str, num_epochs: int = 3, learning_rate: float = 5e-5, logging_steps: int = 10):
        """
        Train the model using the provided training and evaluation data loaders.
        """
        print(f"Training for {num_epochs} epochs on {len(train_dataloader)} train samples and evaluation on {len(eval_dataloader)} samples")

        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=train_dataloader.batch_size,
            per_device_eval_batch_size=eval_dataloader.batch_size,
            num_train_epochs=num_epochs,
            warmup_ratio=0.1,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            push_to_hub=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataloader.dataset,
            eval_dataset=eval_dataloader.dataset,
            data_collator=lambda data: custom_collate(data),
            compute_metrics=self.compute_metrics,
        )

        trainer.train()

        self.save_model(f"model/model_{len(train_dataloader)}_{num_epochs}")

    def infer(self, image_path: str):
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

    def top_n_predictions(self, image_path: str, top_n: int = 1):
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
        # print(top_n_indices, top_n_probs)
        return list(zip(top_n_indices.squeeze().tolist(), top_n_probs.squeeze().tolist()))

    def create_dataloader(self, image_paths: list, batch_size: int):
        """
        Turn an input list of image paths into a DataLoader without labels.
        """
        dataset = ImageDataset(image_paths, transform=self.eval_transforms)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
        print(f"Dataloader of directory dataset is ready:\t{len(image_paths)} images split into {len(dataloader)} batches of size {batch_size}")
        return dataloader

    def infer_dataloader(self, dataloader, top_n: int = 1):
        """
        Perform inference on a DataLoader, optionally with top-N predictions.
        """
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['pixel_values']
                # print(inputs)
                outputs = self.model(pixel_values=inputs.to(self.device))
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                if top_n > 1:
                    top_n_probs, top_n_indices = torch.topk(probabilities, top_n, dim=-1)
                    for indices, probs in zip(top_n_indices, top_n_probs):
                        top_n_probs_normalized = probs / probs.sum()
                        predictions.append(list(zip(indices.tolist(), top_n_probs_normalized.tolist())))
                else:
                    predicted_class_idx = logits.argmax(-1).tolist()
                    predictions.extend(predicted_class_idx)
                print(f"Processed {len(predictions)} images")
        return predictions

    def save_model(self, save_directory: str):
        """
        Save the fine-tuned model and processor to the specified directory.
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        self.model.save_pretrained(save_directory)
        self.processor.save_pretrained(save_directory)
        print(f"Model and processor saved to {save_directory}")

    def load_model(self, load_directory: str):
        """
        Load a fine-tuned model and processor from the specified directory.
        """
        self.processor = AutoImageProcessor.from_pretrained(load_directory)
        self.model = AutoModelForImageClassification.from_pretrained(load_directory).to(self.device)
        print(f"Model and processor loaded from {load_directory}")

    def push_to_hub(self, load_directory: str, repo_name: str, private: bool = False,
                    token: str = None):
        """
        Upload the fine-tuned model and processor to the Hugging Face Model Hub.

        Args:
            load_directory (str): The directory where the model and processor are stored.
            repo_name (str): The name of the repository to create or update on the Hugging Face Hub.
            organization (str, optional): The organization under which to create the repository. Defaults to None.
            private (bool, optional): Whether the repository should be private. Defaults to False.
            token (str, optional): The authentication token for Hugging Face Hub. Defaults to None.
        """
        from huggingface_hub import whoami

        # Determine the repository ID
        username = whoami(token=token)['name']
        repo_id = f"{username}/{repo_name}"

        # Save the model and processor locally
        self.model.save_pretrained(load_directory)
        self.processor.save_pretrained(load_directory)

        # Upload to the Hub
        self.model.push_to_hub(repo_id, private=private, token=token)
        self.processor.push_to_hub(repo_id, private=private, token=token)

        print(f"Model and processor pushed to the Hugging Face Hub: {repo_id}")

    @staticmethod
    def compute_metrics(eval_pred):
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
    def __init__(self, image_paths: list, image_labels: list = None, transform=None):
        self.image_paths = image_paths
        self.image_labels = image_labels
        self.transform = transform

        self.known = True

        if image_labels is None:
            self.known = False

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return {'pixel_values': image, 'label': self.image_labels[idx] if self.known else None}
        except Exception as e:
            print(image_path, e)
            return None





