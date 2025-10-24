#!/usr/bin/env python3
"""
TensorBoard Log Statistics Extractor
Extracts comprehensive statistics from TensorBoard event files and generates a CSV report.
"""

import os
import re
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import argparse
import pandas as pd

try:
    from tensorboard.backend.event_processing import event_accumulator
    # Use the protobuf that ships with tensorboard (no tensorflow dependency)
    from tensorboard.compat.proto import event_pb2
except ImportError:
    print("Error: tensorboard package not found. Install with: pip install tensorboard")
    exit(1)

# GPU node mapping
# GPU node mapping
# GPU node mapping
GPU_NODES = {
    'dll-3gpu1': {'gpu_type': 'NVIDIA A40', 'gpuram': '48G', 'threads': 64},
    'dll-3gpu2': {'gpu_type': 'NVIDIA A40', 'gpuram': '48G', 'threads': 64},
    'dll-3gpu3': {'gpu_type': 'NVIDIA A40', 'gpuram': '48G', 'threads': 64},
    'dll-3gpu4': {'gpu_type': 'NVIDIA A40', 'gpuram': '48G', 'threads': 64},
    'dll-3gpu5': {'gpu_type': 'NVIDIA A40', 'gpuram': '48G', 'threads': 64},
    'dll-4gpu1': {'gpu_type': 'NVIDIA RTX 3090', 'gpuram': '24G', 'threads': 40},
    'dll-4gpu2': {'gpu_type': 'NVIDIA RTX 3090', 'gpuram': '24G', 'threads': 40},
    'dll-4gpu3': {'gpu_type': 'NVIDIA L40', 'gpuram': '48G', 'threads': 62},
    'dll-4gpu4': {'gpu_type': 'NVIDIA A40', 'gpuram': '48G', 'threads': 30},
    'dll-8gpu1': {'gpu_type': 'NVIDIA A30', 'gpuram': '24G', 'threads': 64},
    'dll-8gpu2': {'gpu_type': 'NVIDIA A30', 'gpuram': '24G', 'threads': 64},
    'dll-8gpu3': {'gpu_type': 'NVIDIA RTX A4000', 'gpuram': '16G', 'threads': 32},
    'dll-8gpu4': {'gpu_type': 'NVIDIA RTX A4000', 'gpuram': '16G', 'threads': 32},
    'dll-8gpu5': {'gpu_type': 'NVIDIA Quadro RTX 5000', 'gpuram': '16G', 'threads': 40},
    'dll-8gpu6': {'gpu_type': 'NVIDIA Quadro RTX 5000', 'gpuram': '16G', 'threads': 40},
    'dll-10gpu1': {'gpu_type': 'NVIDIA RTX A4000', 'gpuram': '16G', 'threads': 32},
    'dll-10gpu2': {'gpu_type': 'NVIDIA GeForce GTX 1080 Ti', 'gpuram': '11G', 'threads': 32},
    'dll-10gpu3': {'gpu_type': 'NVIDIA GeForce GTX 1080 Ti', 'gpuram': '11G', 'threads': 32},
    # New nodes with 't' prefix
    'tdll-3gpu1': {'gpu_type': 'NVIDIA A40', 'gpuram': '48G', 'threads': 64},
    'tdll-3gpu2': {'gpu_type': 'NVIDIA A40', 'gpuram': '48G', 'threads': 64},
    'tdll-3gpu3': {'gpu_type': 'NVIDIA A40', 'gpuram': '48G', 'threads': 64},
    'tdll-3gpu4': {'gpu_type': 'NVIDIA A40', 'gpuram': '48G', 'threads': 64},
    'tdll-8gpu1': {'gpu_type': 'NVIDIA A100', 'gpuram': '40G', 'threads': 64},
    'tdll-8gpu2': {'gpu_type': 'NVIDIA A100', 'gpuram': '40G', 'threads': 64},
    'tdll-8gpu3': {'gpu_type': 'NVIDIA Quadro P5000', 'gpuram': '16G', 'threads': 32},
    'tdll-8gpu4': {'gpu_type': 'NVIDIA Quadro P5000', 'gpuram': '16G', 'threads': 32},
    'tdll-8gpu5': {'gpu_type': 'NVIDIA Quadro P5000', 'gpuram': '16G', 'threads': 32},
    'tdll-8gpu6': {'gpu_type': 'NVIDIA Quadro P5000', 'gpuram': '16G', 'threads': 32},
    'tdll-8gpu7': {'gpu_type': 'NVIDIA Quadro P5000', 'gpuram': '16G', 'threads': 32},
}
# Revision to base model mapping
REVISION_TO_BASE_MODEL = {
    "v13": "timm/tf_efficientnetv2_s.in21k",
    "v23": "google/vit-base-patch16-224",
    "v15": "timm/tf_efficientnetv2_s.in21k",
    "v25": "google/vit-base-patch16-224",
    "v33": "google/vit-base-patch16-384",
    "v35": "google/vit-base-patch16-384",
    "v43": "timm/tf_efficientnetv2_l.in21k_ft_in1k",
    "v45": "timm/tf_efficientnetv2_l.in21k_ft_in1k",
    "v53": "google/vit-large-patch16-384",
    "v55": "google/vit-large-patch16-384",
    "v63": "timm/regnety_120.sw_in12k_ft_in1k",
    "v73": "timm/regnety_160.swag_ft_in1k",
    "v83": "timm/regnety_640.seer",
    "v93": "microsoft/dit-base-finetuned-rvlcdip",
    "v103": "microsoft/dit-large-finetuned-rvlcdip",
    "v113": "microsoft/dit-large",
    "v123": "timm/tf_efficientnetv2_m.in21k_ft_in1k",
    "v1.1.3": "CLIP ViT-B/16",
    "v1.2.3": "CLIP ViT-B/32",
    "v2.1.3": "CLIP ViT-L/14",
    "v2.2.3": "CLIP ViT-L/14-336",
}

MODEL_CATALOG = {
    # EfficientNet v2
    "timm/tf_efficientnetv2_s.in21k": {"parameters_m": 48.2, "input_size": "300x300", "accuracy_top1": 97.87,
                                       "is_most_efficient": False, "is_most_accurate": False,
                                       "family": "EfficientNet-v2"},
    "timm/tf_efficientnetv2_m.in21k_ft_in1k": {"parameters_m": 54.1, "input_size": "384x384", "accuracy_top1": 98.9,
                                               "is_most_efficient": False, "is_most_accurate": True,
                                               "family": "EfficientNet-v2"},
    "timm/tf_efficientnetv2_l.in21k_ft_in1k": {"parameters_m": 118.5, "input_size": "384x384", "accuracy_top1": 98.77,
                                               "is_most_efficient": False, "is_most_accurate": False,
                                               "family": "EfficientNet-v2"},

    # RegNetY
    "timm/regnety_120.sw_in12k_ft_in1k": {"parameters_m": 51.8, "input_size": "224x224", "accuracy_top1": 98.29,
                                          "is_most_efficient": False, "is_most_accurate": False, "family": "RegNetY"},
    "timm/regnety_160.swag_ft_in1k": {"parameters_m": 83.6, "input_size": "224x224", "accuracy_top1": 99.21,
                                      "is_most_efficient": False, "is_most_accurate": True, "family": "RegNetY"},
    "timm/regnety_640.seer": {"parameters_m": 281.4, "input_size": "384x384", "accuracy_top1": 98.79,
                              "is_most_efficient": False, "is_most_accurate": False, "family": "RegNetY"},

    # DIT (microsoft)
    "microsoft/dit-base-finetuned-rvlcdip": {"parameters_m": 86.0, "input_size": "224x224", "accuracy_top1": 98.72,
                                             "is_most_efficient": False, "is_most_accurate": True, "family": "DiT"},
    "microsoft/dit-large": {"parameters_m": 304.0, "input_size": "224x224", "accuracy_top1": 98.53,
                            "is_most_efficient": False, "is_most_accurate": False, "family": "DiT"},
    "microsoft/dit-large-finetuned-rvlcdip": {"parameters_m": 304.0, "input_size": "224x224", "accuracy_top1": 98.66,
                                              "is_most_efficient": False, "is_most_accurate": False, "family": "DiT"},

    # ViT
    "google/vit-base-patch16-224": {"parameters_m": 86.6, "input_size": "224x224", "accuracy_top1": 98.88,
                                    "is_most_efficient": True, "is_most_accurate": False, "family": "ViT"},
    "google/vit-base-patch16-384": {"parameters_m": 86.9, "input_size": "384x384", "accuracy_top1": 98.99,
                                    "is_most_efficient": False, "is_most_accurate": False, "family": "ViT"},
    "google/vit-large-patch16-384": {"parameters_m": 304.7, "input_size": "384x384", "accuracy_top1": 99.25,
                                     "is_most_efficient": False, "is_most_accurate": True, "family": "ViT"},

    # CLIP
    "CLIP ViT-B/16": {"parameters_m": 150.0, "input_size": "224x224", "accuracy_top1": 99.0, "is_most_efficient": True,
                      "is_most_accurate": True, "family": "CLIP"},
    "CLIP ViT-B/32": {"parameters_m": 151.0, "input_size": "224x224", "accuracy_top1": 98.92,
                      "is_most_efficient": False, "is_most_accurate": False, "family": "CLIP"},
    "CLIP ViT-L/14": {"parameters_m": 428.0, "input_size": "224x224", "accuracy_top1": 98.7, "is_most_efficient": False,
                      "is_most_accurate": False, "family": "CLIP"},
    "CLIP ViT-L/14-336": {"parameters_m": 428.0, "input_size": "336x336", "accuracy_top1": 98.64,
                          "is_most_efficient": False, "is_most_accurate": False, "family": "CLIP"},
}


def get_base_model_from_revision(revision: Optional[str]) -> Optional[str]:
    """Map revision to base model using the revision mapping."""
    if not revision:
        return None

    for prefix, model in REVISION_TO_BASE_MODEL.items():
        if revision.startswith(prefix):
            return model
    return None


def get_model_details(model_name: Optional[str]) -> Dict[str, any]:
    """Get model details from the model catalog."""
    if not model_name or model_name not in MODEL_CATALOG:
        return {}
    return MODEL_CATALOG[model_name]


def parse_text_summaries(log_path: Path) -> Dict[str, any]:
    """Parse training args and model config from text summaries in event files."""
    training_args = {}
    model_config = {}

    event_files = list(log_path.glob('events.out.tfevents.*'))
    if not event_files:
        return training_args, model_config

    for event_file in event_files:
        try:
            with open(event_file, 'rb') as f:
                for record in f:
                    try:
                        event = event_pb2.Event.FromString(record)

                        # Check for text summaries
                        if event.HasField('summary'):
                            for value in event.summary.value:
                                # Parse args/text_summary
                                if value.tag == 'args/text_summary' and value.HasField('tensor'):
                                    text_data = value.tensor.string_val[0].decode('utf-8')
                                    try:
                                        args_dict = json.loads(text_data)
                                        training_args = args_dict
                                    except json.JSONDecodeError:
                                        pass

                                # Parse model_config/text_summary
                                elif value.tag == 'model_config/text_summary' and value.HasField('tensor'):
                                    text_data = value.tensor.string_val[0].decode('utf-8')
                                    try:
                                        config_dict = json.loads(text_data)
                                        model_config = config_dict
                                    except json.JSONDecodeError:
                                        pass
                    except Exception:
                        continue
        except Exception as e:
            print(f"Warning: Could not read event file {event_file.name}: {e}")

    return training_args, model_config


def parse_log_folder_name(folder_name: str) -> Dict[str, Optional[str]]:
    """Extract parameters from log folder name."""
    info = {
        'script': None,
        'timestamp': None,
        'learning_rate': None,
        'epochs': None,
        'model': None,
        'revision': None,
        'batch_size': None,
        'parameters_m': None,
    }

    # Extract script name (classifier.py or run.py)
    if folder_name.startswith('classifierpy'):
        info['script'] = 'classifier.py'
    elif folder_name.startswith('run.py'):
        info['script'] = 'run.py'

    # Extract timestamp
    timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{6})', folder_name)
    if timestamp_match:
        info['timestamp'] = timestamp_match.group(1)

    # Pattern 1: classifier.py format
    lr_match = re.search(r'lr=(\d+e[_-]\d+)', folder_name)
    if lr_match:
        lr_str = lr_match.group(1).replace('_', '-')
        info['learning_rate'] = lr_str

    e_match = re.search(r'-e=(\d+)', folder_name)
    if e_match:
        info['epochs'] = e_match.group(1)

    model_match = re.search(r'-m=([^-]+?)(?:-model_v|$)', folder_name)
    if model_match:
        info['model'] = model_match.group(1)

    version_match = re.search(r'model_v(\d+)', folder_name)
    if version_match:
        info['revision'] = f"v{version_match.group(1)}"

    bs_match = re.search(r'bs=(\d+)', folder_name)
    if bs_match:
        info['batch_size'] = bs_match.group(1)

    # Pattern 2: run.py format with parameters
    param_match = re.search(r'-a=([^,]+),bs=(\d+),e=(\d+),l=([\de-]+),mc=(\d+),mce=(\d+),r=([^,]+),t=(\d+),zs=([^-]+)',
                            folder_name)
    if param_match:
        info['epochs'] = param_match.group(3)
        info['learning_rate'] = param_match.group(4)
        info['revision'] = param_match.group(7)
        info['batch_size'] = param_match.group(2)

    # Extract model name at the end (for run.py format)
    if not info['model']:
        model_end_match = re.search(r'-([^-]+)$', folder_name)
        if model_end_match:
            info['model'] = model_end_match.group(1)

    return info


def extract_gpu_node(event_file_path: str) -> Optional[str]:
    """Extract GPU node name from event file path."""
    match = re.search(r'dll-\d+gpu\d+', event_file_path)
    return match.group(0) if match else None


def extract_performance_metrics(ea: event_accumulator.EventAccumulator) -> Dict[str, any]:
    """Extract detailed performance metrics from TensorBoard logs."""
    metrics = {
        # Training metrics
        'train_loss_final': None,
        'train_loss_initial': None,
        'train_loss_improvement': None,
        'train_samples_per_second_avg': None,
        'train_steps_per_second_avg': None,
        'train_total_steps': None,

        # Evaluation metrics
        'eval_accuracy_final': None,
        'eval_accuracy_best': None,
        'eval_loss_final': None,
        'eval_loss_best': None,
        'eval_samples_per_second_avg': None,
        'eval_steps_per_second_avg': None,
        'eval_count': None,

        # Learning rate tracking
        'learning_rate_initial': None,
        'learning_rate_final': None,

        # Gradient norm (for training stability)
        'grad_norm_avg': None,
        'grad_norm_max': None,

        # FLOPs (floating point operations)
        'total_flos': None,

        # Epoch information
        'epochs_completed': None,
    }

    try:
        scalar_tags = ea.Tags().get('scalars', [])

        # Extract train/loss
        if 'train/loss' in scalar_tags:
            events = ea.Scalars('train/loss')
            if events:
                metrics['train_loss_initial'] = round(events[0].value, 4)
                metrics['train_loss_final'] = round(events[-1].value, 4)
                metrics['train_loss_improvement'] = round(
                    metrics['train_loss_initial'] - metrics['train_loss_final'], 4
                )
                metrics['train_total_steps'] = len(events)

        # Extract train/train_samples_per_second
        if 'train/train_samples_per_second' in scalar_tags:
            events = ea.Scalars('train/train_samples_per_second')
            if events:
                avg = sum(e.value for e in events) / len(events)
                metrics['train_samples_per_second_avg'] = round(avg, 2)

        # Extract train/train_steps_per_second
        if 'train/train_steps_per_second' in scalar_tags:
            events = ea.Scalars('train/train_steps_per_second')
            if events:
                avg = sum(e.value for e in events) / len(events)
                metrics['train_steps_per_second_avg'] = round(avg, 2)

        # Extract eval/accuracy
        if 'eval/accuracy' in scalar_tags:
            events = ea.Scalars('eval/accuracy')
            if events:
                metrics['eval_accuracy_final'] = round(events[-1].value * 100, 2)  # Convert to percentage
                metrics['eval_accuracy_best'] = round(max(e.value for e in events) * 100, 2)
                metrics['eval_count'] = len(events)

        # Extract eval/loss
        if 'eval/loss' in scalar_tags:
            events = ea.Scalars('eval/loss')
            if events:
                metrics['eval_loss_final'] = round(events[-1].value, 4)
                metrics['eval_loss_best'] = round(min(e.value for e in events), 4)

        # Extract eval/samples_per_second
        if 'eval/samples_per_second' in scalar_tags:
            events = ea.Scalars('eval/samples_per_second')
            if events:
                avg = sum(e.value for e in events) / len(events)
                metrics['eval_samples_per_second_avg'] = round(avg, 2)

        # Extract eval/steps_per_second
        if 'eval/steps_per_second' in scalar_tags:
            events = ea.Scalars('eval/steps_per_second')
            if events:
                avg = sum(e.value for e in events) / len(events)
                metrics['eval_steps_per_second_avg'] = round(avg, 2)

        # Extract train/learning_rate
        if 'train/learning_rate' in scalar_tags:
            events = ea.Scalars('train/learning_rate')
            if events:
                metrics['learning_rate_initial'] = events[0].value
                metrics['learning_rate_final'] = events[-1].value

        # Extract train/grad_norm
        if 'train/grad_norm' in scalar_tags:
            events = ea.Scalars('train/grad_norm')
            if events:
                values = [e.value for e in events]
                metrics['grad_norm_avg'] = round(sum(values) / len(values), 4)
                metrics['grad_norm_max'] = round(max(values), 4)

        # Extract train/total_flos
        if 'train/total_flos' in scalar_tags:
            events = ea.Scalars('train/total_flos')
            if events:
                # Convert to TFLOPs for readability
                metrics['total_flos'] = round(events[-1].value / 1e12, 2)

        # Extract train/epoch
        if 'train/epoch' in scalar_tags:
            events = ea.Scalars('train/epoch')
            if events:
                metrics['epochs_completed'] = round(events[-1].value, 2)

    except Exception as e:
        print(f"Warning: Error extracting performance metrics: {e}")

    return metrics


def process_log_folder(log_path: Path) -> Dict[str, any]:
    """Process a single log folder and extract all statistics."""
    folder_name = log_path.name

    # Parse folder name
    info = parse_log_folder_name(folder_name)

    # Find event file
    event_files = list(log_path.glob('events.out.tfevents.*'))
    if not event_files:
        print(f"Warning: No event files found in {folder_name}")
        return None

    event_file = event_files[0]

    # Extract GPU node
    gpu_node = extract_gpu_node(str(event_file))
    gpu_info = GPU_NODES.get(gpu_node, {}) if gpu_node else {}

    # Parse text summaries for training args and model config
    training_args, model_config = parse_text_summaries(log_path)

    # Load TensorBoard data
    try:
        ea = event_accumulator.EventAccumulator(str(log_path))
        ea.Reload()

        # Get training duration
        if ea.Tags()['scalars']:
            first_tag = ea.Tags()['scalars'][0]
            events = ea.Scalars(first_tag)
            if events:
                start_time = events[0].wall_time
                end_time = events[-1].wall_time
                duration_seconds = end_time - start_time
                duration_hours = duration_seconds / 3600
                duration_days = duration_hours / 24
            else:
                duration_seconds = None
                duration_hours = None
                duration_days = None
        else:
            duration_seconds = None
            duration_hours = None
            duration_days = None

        # Extract performance metrics
        performance_metrics = extract_performance_metrics(ea)

        # Extract counts of train and eval events
        train_event_count = 0
        eval_event_count = 0

        train_keywords = ['train', 'training']
        eval_keywords = ['eval', 'evaluation', 'validation', 'val']

        scalar_tags = ea.Tags().get('scalars', [])
        for tag in scalar_tags:
            lower_tag = tag.lower()

            try:
                events = ea.Scalars(tag)
            except Exception:
                events = []

            count = len(events)

            is_train = any(k in lower_tag for k in train_keywords)
            is_eval = any(k in lower_tag for k in eval_keywords)

            if is_train and not is_eval:
                train_event_count += count
            elif is_eval and not is_train:
                eval_event_count += count
            elif is_train and is_eval:
                if any(k in lower_tag for k in eval_keywords):
                    eval_event_count += count
                else:
                    train_event_count += count

    except Exception as e:
        print(f"Error processing {folder_name}: {e}")
        return None

    # Get base model from revision
    base_model = get_base_model_from_revision(info['revision'])

    # Extract key training parameters from training_args
    model_details = get_model_details(base_model)

    # Calculate efficiency metrics
    efficiency_metrics = {}
    if performance_metrics['eval_accuracy_best'] and duration_hours:
        # Accuracy per hour
        efficiency_metrics['accuracy_per_hour'] = round(
            performance_metrics['eval_accuracy_best'] / duration_hours, 2
        )

    if performance_metrics['eval_accuracy_best'] and model_details.get('parameters_m'):
        # Accuracy per million parameters
        efficiency_metrics['accuracy_per_m_params'] = round(
            performance_metrics['eval_accuracy_best'] / model_details['parameters_m'], 4
        )

    if duration_hours and model_details.get('parameters_m'):
        # Training speed: M params per hour
        efficiency_metrics['m_params_per_hour'] = round(
            model_details['parameters_m'] / duration_hours, 2
        )

    # Compile all information
    result = {
        'folder_name': folder_name,
        'script': info['script'],
        'timestamp': info['timestamp'],
        'gpu_type': gpu_info.get('gpu_type'),
        'gpu_ram': gpu_info.get('gpuram'),
        'model': base_model,
        'model_family': model_details.get('family'),
        'parameters_m': model_details.get('parameters_m'),
        'input_size': model_details.get('input_size'),
        'revision': info['revision'],
        'epochs': info['epochs'] or training_args.get('num_train_epochs'),
        'batch_size': info['batch_size'] or training_args.get('per_device_train_batch_size'),
        'learning_rate': info['learning_rate'] or training_args.get('learning_rate'),

        # Duration metrics
        'duration_seconds': round(duration_seconds, 0) if duration_seconds else None,
        'duration_hours': round(duration_hours, 2) if duration_hours else None,
        'duration_days': round(duration_days, 3) if duration_days else None,

        # Training performance
        'train_loss_initial': performance_metrics['train_loss_initial'],
        'train_loss_final': performance_metrics['train_loss_final'],
        'train_loss_improvement': performance_metrics['train_loss_improvement'],
        'train_samples_per_second': performance_metrics['train_samples_per_second_avg'],
        'train_steps_per_second': performance_metrics['train_steps_per_second_avg'],
        'train_total_steps': performance_metrics['train_total_steps'],

        # Evaluation performance
        'eval_accuracy_final': performance_metrics['eval_accuracy_final'],
        'eval_accuracy_best': performance_metrics['eval_accuracy_best'],
        'eval_loss_final': performance_metrics['eval_loss_final'],
        'eval_loss_best': performance_metrics['eval_loss_best'],
        'eval_samples_per_second': performance_metrics['eval_samples_per_second_avg'],
        'eval_steps_per_second': performance_metrics['eval_steps_per_second_avg'],
        'eval_count': performance_metrics['eval_count'],

        # Learning and stability
        'learning_rate_initial': performance_metrics['learning_rate_initial'],
        'learning_rate_final': performance_metrics['learning_rate_final'],
        'grad_norm_avg': performance_metrics['grad_norm_avg'],
        'grad_norm_max': performance_metrics['grad_norm_max'],
        'total_flos_tflops': performance_metrics['total_flos'],
        'epochs_completed': performance_metrics['epochs_completed'],

        # Efficiency metrics
        'accuracy_per_hour': efficiency_metrics.get('accuracy_per_hour'),
        'accuracy_per_m_params': efficiency_metrics.get('accuracy_per_m_params'),
        'm_params_per_hour': efficiency_metrics.get('m_params_per_hour'),

        # Event counts
        'train_event_count': train_event_count,
        'eval_event_count': eval_event_count,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description='Extract statistics from TensorBoard logs')
    parser.add_argument('input_dir', type=str, help='Directory containing log folders')
    parser.add_argument('-o', '--output', type=str, default='tensorboard_stats.csv',
                        help='Output CSV file path (default: tensorboard_stats.csv)')
    parser.add_argument('--pattern', type=str, default='*',
                        help='Pattern to filter log folders (default: *)')

    args = parser.parse_args()

    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return

    # Find all log folders
    log_folders = sorted([d for d in input_path.glob(args.pattern) if d.is_dir()])

    if not log_folders:
        print(f"No log folders found in {args.input_dir}")
        return

    print(f"Found {len(log_folders)} log folders")
    print("Processing...")

    # Process each folder
    results = []
    for i, log_folder in enumerate(log_folders, 1):
        print(f"[{i}/{len(log_folders)}] Processing {log_folder.name}...")
        result = process_log_folder(log_folder)
        if result:
            results.append(result)

    if not results:
        print("No valid results extracted")
        return

    # Get all unique column names
    all_columns = set()
    for result in results:
        all_columns.update(result.keys())

    # Define column order
    fixed_columns = [
        # Identification
        'folder_name', 'script', 'timestamp',

        # Hardware
        'gpu_type', 'gpu_ram',

        # Model info
        'model', 'model_family', 'parameters_m', 'input_size', 'revision',

        # Training config
        'epochs', 'batch_size', 'learning_rate',

        # Duration
        'duration_hours', 'duration_days', 'duration_seconds',

        # Evaluation performance (most important)
        'eval_accuracy_best', 'eval_accuracy_final', 'eval_loss_best', 'eval_loss_final',

        # Training performance
        'train_loss_initial', 'train_loss_final', 'train_loss_improvement',

        # Throughput metrics
        'train_samples_per_second', 'train_steps_per_second',

    ]
    
    scalar_columns = sorted([col for col in all_columns if col not in fixed_columns])
    columns = fixed_columns + scalar_columns
    
    # Write CSV
    output_path = Path(args.output)
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"\n✓ Successfully extracted statistics from {len(results)} log folders")
    print(f"✓ Output saved to: {output_path}")
    print(f"✓ Total columns: {len(columns)}")

    df = pd.read_csv(output_path)

    # Filter out rows with missing values for required columns
    df_filtered = df[df['train_samples_per_second'].notna() & 
                        df['eval_samples_per_second'].notna() &
                      df['accuracy_per_hour'].notna() & 
                      df['parameters_m'].notna()].copy()

    # Define parameter ranges
    def categorize_params(params):
        if params < 70:
            return '0-70M'
        elif params < 100:
            return '70-100M'
        elif params < 200:
            return '100-200M'
        else:
            return '200-350M'

    df_filtered['param_range'] = df_filtered['parameters_m'].apply(categorize_params)

    # Table 1: Average Training Throughput
    print("--- Summary Table: Average Training Throughput (Samples/Second) ---")
    print("Cell content: mean(train_samples_per_second)")
    print("Rows: Model size (parameters_m) range")
    print("Columns: GPU Type")

    throughput_table = df_filtered.pivot_table(
        values='train_samples_per_second',
        index='param_range',
        columns='gpu_type',
        aggfunc='mean'
    )

    # Reorder rows
    row_order = ['0-70M', '70-100M', '100-200M', '200-350M']
    throughput_table = throughput_table.reindex([r for r in row_order if r in throughput_table.index])

    print(throughput_table.to_string(float_format='%.2f'))
    print()


    print("--- Summary Table: Average Evaluation Throughput (Samples/Second) ---")
    print("Cell content: mean(eval_samples_per_second)")
    print("Rows: Model size (parameters_m) range")
    print("Columns: GPU Type")

    throughput_table = df_filtered.pivot_table(
        values='eval_samples_per_second',
        index='param_range',
        columns='gpu_type',
        aggfunc='mean'
    )

    # Reorder rows
    row_order = ['0-70M', '70-100M', '100-200M', '200-350M']
    throughput_table = throughput_table.reindex([r for r in row_order if r in throughput_table.index])

    print(throughput_table.to_string(float_format='%.2f'))
    print()


    # Table 2: Average Efficiency
    print("--- Summary Table: Average Efficiency (Accuracy per Hour) ---")
    print("Cell content: mean(accuracy_per_hour)")
    print("Rows: Model size (parameters_m) range")
    print("Columns: GPU Type")

    efficiency_table = df_filtered.pivot_table(
        values='accuracy_per_hour',
        index='param_range',
        columns='gpu_type',
        aggfunc='mean'
    )

    # Reorder rows
    efficiency_table = efficiency_table.reindex([r for r in row_order if r in efficiency_table.index])

    print(efficiency_table.to_string(float_format='%.2f'))


if __name__ == '__main__':
    main()

