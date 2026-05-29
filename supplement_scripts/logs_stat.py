#!/usr/bin/env python3
"""
TensorBoard Log Statistics Extractor.

Extracts comprehensive statistics from TensorBoard event files and generates
a CSV report.  Optionally loads GPU node and model revision mappings from
external JSON files so the script works outside the original UFAL cluster.
"""

import os
import re
import csv
import json
import struct
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
import argparse
import pandas as pd

try:
    from tensorboard.backend.event_processing import event_accumulator
    from tensorboard.compat.proto import event_pb2
except ImportError:
    print("Error: tensorboard package not found.  Install with:  pip install tensorboard")
    exit(1)


# ── Built-in GPU node specs (UFAL cluster) ────────────────────────────────
# These can be overridden or extended at runtime via --gpu-map <json_file>.
# Format:  { "node-name": { "gpu_type": "...", "gpuram": "...", "threads": N } }
GPU_NODES: Dict[str, Dict[str, Any]] = {
    'dll-3gpu1': {'gpu_type': 'NVIDIA A40',                 'gpuram': '48G', 'threads': 64},
    'dll-3gpu2': {'gpu_type': 'NVIDIA A40',                 'gpuram': '48G', 'threads': 64},
    'dll-3gpu3': {'gpu_type': 'NVIDIA A40',                 'gpuram': '48G', 'threads': 64},
    'dll-3gpu4': {'gpu_type': 'NVIDIA A40',                 'gpuram': '48G', 'threads': 64},
    'dll-3gpu5': {'gpu_type': 'NVIDIA A40',                 'gpuram': '48G', 'threads': 64},
    'dll-4gpu1': {'gpu_type': 'NVIDIA RTX 3090',            'gpuram': '24G', 'threads': 40},
    'dll-4gpu2': {'gpu_type': 'NVIDIA RTX 3090',            'gpuram': '24G', 'threads': 40},
    'dll-4gpu3': {'gpu_type': 'NVIDIA L40',                 'gpuram': '48G', 'threads': 62},
    'dll-4gpu4': {'gpu_type': 'NVIDIA A40',                 'gpuram': '48G', 'threads': 30},
    'dll-8gpu1': {'gpu_type': 'NVIDIA A30',                 'gpuram': '24G', 'threads': 64},
    'dll-8gpu2': {'gpu_type': 'NVIDIA A30',                 'gpuram': '24G', 'threads': 64},
    'dll-8gpu3': {'gpu_type': 'NVIDIA RTX A4000',           'gpuram': '16G', 'threads': 32},
    'dll-8gpu4': {'gpu_type': 'NVIDIA RTX A4000',           'gpuram': '16G', 'threads': 32},
    'dll-8gpu5': {'gpu_type': 'NVIDIA Quadro RTX 5000',     'gpuram': '16G', 'threads': 40},
    'dll-8gpu6': {'gpu_type': 'NVIDIA Quadro RTX 5000',     'gpuram': '16G', 'threads': 40},
    'dll-10gpu1':{'gpu_type': 'NVIDIA RTX A4000',           'gpuram': '16G', 'threads': 32},
    'dll-10gpu2':{'gpu_type': 'NVIDIA GeForce GTX 1080 Ti', 'gpuram': '11G', 'threads': 32},
    'dll-10gpu3':{'gpu_type': 'NVIDIA GeForce GTX 1080 Ti', 'gpuram': '11G', 'threads': 32},
    'tdll-3gpu1':{'gpu_type': 'NVIDIA A40',                 'gpuram': '48G', 'threads': 64},
    'tdll-3gpu2':{'gpu_type': 'NVIDIA A40',                 'gpuram': '48G', 'threads': 64},
    'tdll-3gpu3':{'gpu_type': 'NVIDIA A40',                 'gpuram': '48G', 'threads': 64},
    'tdll-3gpu4':{'gpu_type': 'NVIDIA A40',                 'gpuram': '48G', 'threads': 64},
    'tdll-8gpu1':{'gpu_type': 'NVIDIA A100',                'gpuram': '40G', 'threads': 64},
    'tdll-8gpu2':{'gpu_type': 'NVIDIA A100',                'gpuram': '40G', 'threads': 64},
    'tdll-8gpu3':{'gpu_type': 'NVIDIA Quadro P5000',        'gpuram': '16G', 'threads': 32},
    'tdll-8gpu4':{'gpu_type': 'NVIDIA Quadro P5000',        'gpuram': '16G', 'threads': 32},
    'tdll-8gpu5':{'gpu_type': 'NVIDIA Quadro P5000',        'gpuram': '16G', 'threads': 32},
    'tdll-8gpu6':{'gpu_type': 'NVIDIA Quadro P5000',        'gpuram': '16G', 'threads': 32},
    'tdll-8gpu7':{'gpu_type': 'NVIDIA Quadro P5000',        'gpuram': '16G', 'threads': 32},
}

# ── Built-in revision → base model mapping ────────────────────────────────
# Extend at runtime via --revision-map <json_file>.
# Format:  { "vXX": "org/model-name" }
REVISION_TO_BASE_MODEL: Dict[str, str] = {
    "v13":  "timm/tf_efficientnetv2_s.in21k",
    "v23":  "google/vit-base-patch16-224",
    "v33":  "google/vit-base-patch16-384",
    "v43":  "timm/tf_efficientnetv2_l.in21k_ft_in1k",
    "v53":  "google/vit-large-patch16-384",
    "v63":  "timm/regnety_120.sw_in12k_ft_in1k",
    "v73":  "timm/regnety_160.swag_ft_in1k",
    "v83":  "timm/regnety_640.seer",
    "v93":  "microsoft/dit-base-finetuned-rvlcdip",
    "v103": "microsoft/dit-large-finetuned-rvlcdip",
    "v113": "microsoft/dit-large",
    "v123": "timm/tf_efficientnetv2_m.in21k_ft_in1k",
    "v1.1.3": "CLIP ViT-B/16",
    "v1.2.3": "CLIP ViT-B/32",
    "v2.1.3": "CLIP ViT-L/14",
    "v2.2.3": "CLIP ViT-L/14-336",
}

MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "timm/tf_efficientnetv2_s.in21k":            {"parameters_m": 48.2,  "input_size": "300x300", "accuracy_top1": 97.87, "family": "EfficientNet-v2"},
    "timm/tf_efficientnetv2_m.in21k_ft_in1k":   {"parameters_m": 54.1,  "input_size": "384x384", "accuracy_top1": 98.90, "family": "EfficientNet-v2"},
    "timm/tf_efficientnetv2_l.in21k_ft_in1k":   {"parameters_m": 118.5, "input_size": "384x384", "accuracy_top1": 98.77, "family": "EfficientNet-v2"},
    "timm/regnety_120.sw_in12k_ft_in1k":         {"parameters_m": 51.8,  "input_size": "224x224", "accuracy_top1": 98.29, "family": "RegNetY"},
    "timm/regnety_160.swag_ft_in1k":             {"parameters_m": 83.6,  "input_size": "224x224", "accuracy_top1": 99.21, "family": "RegNetY"},
    "timm/regnety_640.seer":                      {"parameters_m": 281.4, "input_size": "384x384", "accuracy_top1": 98.79, "family": "RegNetY"},
    "microsoft/dit-base-finetuned-rvlcdip":       {"parameters_m": 86.0,  "input_size": "224x224", "accuracy_top1": 98.72, "family": "DiT"},
    "microsoft/dit-large":                         {"parameters_m": 304.0, "input_size": "224x224", "accuracy_top1": 98.53, "family": "DiT"},
    "microsoft/dit-large-finetuned-rvlcdip":      {"parameters_m": 304.0, "input_size": "224x224", "accuracy_top1": 98.66, "family": "DiT"},
    "google/vit-base-patch16-224":                {"parameters_m": 86.6,  "input_size": "224x224", "accuracy_top1": 98.88, "family": "ViT"},
    "google/vit-base-patch16-384":                {"parameters_m": 86.9,  "input_size": "384x384", "accuracy_top1": 98.99, "family": "ViT"},
    "google/vit-large-patch16-384":               {"parameters_m": 304.7, "input_size": "384x384", "accuracy_top1": 99.25, "family": "ViT"},
    "CLIP ViT-B/16":                              {"parameters_m": 150.0, "input_size": "224x224", "accuracy_top1": 99.00, "family": "CLIP"},
    "CLIP ViT-B/32":                              {"parameters_m": 151.0, "input_size": "224x224", "accuracy_top1": 98.92, "family": "CLIP"},
    "CLIP ViT-L/14":                              {"parameters_m": 428.0, "input_size": "224x224", "accuracy_top1": 98.70, "family": "CLIP"},
    "CLIP ViT-L/14-336":                          {"parameters_m": 428.0, "input_size": "336x336", "accuracy_top1": 98.64, "family": "CLIP"},
}


# ── TFRecord parsing ──────────────────────────────────────────────────────

def _iter_tfrecords(filepath: Path):
    """Yield raw protobuf bytes from a TFRecord file without tensorflow.

    P4 FIX: the original code did `for record in open(event_file, 'rb')` which
    iterates by newline characters, not TFRecord boundaries.  TFRecord format
    uses length-prefixed binary framing:

        [8-byte uint64 length] [4-byte masked CRC32 of length]
        [<length> bytes data]  [4-byte masked CRC32 of data]

    We skip CRC validation (safe for reading) and yield only the data payload.
    """
    with open(filepath, 'rb') as f:
        while True:
            len_header = f.read(12)       # 8-byte length + 4-byte CRC
            if len(len_header) < 12:
                break
            data_length = struct.unpack('<Q', len_header[:8])[0]
            data = f.read(data_length)
            if len(data) < data_length:
                break
            f.read(4)                     # 4-byte data CRC — skip
            yield data


def parse_text_summaries(log_path: Path) -> tuple:
    """Extract training_args and model_config from TensorBoard text summaries.

    Returns:
        (training_args dict, model_config dict) — both empty on failure.
    """
    training_args: Dict[str, Any] = {}
    model_config:  Dict[str, Any] = {}

    event_files = list(log_path.glob('events.out.tfevents.*'))
    if not event_files:
        return training_args, model_config

    for event_file in event_files:
        try:
            for data in _iter_tfrecords(event_file):
                try:
                    event = event_pb2.Event()
                    event.ParseFromString(data)
                    if not event.HasField('summary'):
                        continue
                    for value in event.summary.value:
                        if not value.HasField('tensor') or not value.tensor.string_val:
                            continue
                        try:
                            text_data = value.tensor.string_val[0].decode('utf-8')
                        except (UnicodeDecodeError, IndexError):
                            continue
                        if value.tag == 'args/text_summary':
                            try:
                                training_args = json.loads(text_data)
                            except json.JSONDecodeError:
                                pass
                        elif value.tag == 'model_config/text_summary':
                            try:
                                model_config = json.loads(text_data)
                            except json.JSONDecodeError:
                                pass
                except Exception:
                    continue
        except Exception as e:
            print(f"Warning: Could not read event file {event_file.name}: {e}")

    return training_args, model_config


# ── Helper functions ──────────────────────────────────────────────────────

def get_base_model_from_revision(revision: Optional[str]) -> Optional[str]:
    if not revision:
        return None
    for prefix, model in REVISION_TO_BASE_MODEL.items():
        if revision.startswith(prefix):
            return model
    return None


def get_model_details(model_name: Optional[str]) -> Dict[str, Any]:
    if not model_name or model_name not in MODEL_CATALOG:
        return {}
    return MODEL_CATALOG[model_name]


def parse_log_folder_name(folder_name: str) -> Dict[str, Optional[str]]:
    info: Dict[str, Optional[str]] = {
        'script': None, 'timestamp': None, 'learning_rate': None,
        'epochs': None, 'model': None, 'revision': None, 'batch_size': None,
    }

    if folder_name.startswith('classifierpy'):
        info['script'] = 'classifier.py'
    elif folder_name.startswith('run.py'):
        info['script'] = 'run.py'

    ts_match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{6})', folder_name)
    if ts_match:
        info['timestamp'] = ts_match.group(1)

    lr_match = re.search(r'lr=(\d+e[_-]\d+)', folder_name)
    if lr_match:
        info['learning_rate'] = lr_match.group(1).replace('_', '-')

    e_match = re.search(r'-e=(\d+)', folder_name)
    if e_match:
        info['epochs'] = e_match.group(1)

    m_match = re.search(r'-m=([^-]+?)(?:-model_v|$)', folder_name)
    if m_match:
        info['model'] = m_match.group(1)

    v_match = re.search(r'model_v(\d+)', folder_name)
    if v_match:
        info['revision'] = f"v{v_match.group(1)}"

    bs_match = re.search(r'bs=(\d+)', folder_name)
    if bs_match:
        info['batch_size'] = bs_match.group(1)

    param_match = re.search(
        r'-a=([^,]+),bs=(\d+),e=(\d+),l=([\de-]+),mc=(\d+),mce=(\d+),r=([^,]+),t=(\d+),zs=([^-]+)',
        folder_name,
    )
    if param_match:
        info['epochs']       = param_match.group(3)
        info['learning_rate']= param_match.group(4)
        info['revision']     = param_match.group(7)
        info['batch_size']   = param_match.group(2)

    if not info['model']:
        end_match = re.search(r'-([^-]+)$', folder_name)
        if end_match:
            info['model'] = end_match.group(1)

    return info


def extract_gpu_node(event_file_path: str) -> Optional[str]:
    match = re.search(r'[t]?dll-\d+gpu\d+', event_file_path)
    return match.group(0) if match else None


def extract_performance_metrics(ea: event_accumulator.EventAccumulator) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        'train_loss_final': None, 'train_loss_initial': None, 'train_loss_improvement': None,
        'train_samples_per_second_avg': None, 'train_steps_per_second_avg': None,
        'train_total_steps': None, 'eval_accuracy_final': None, 'eval_accuracy_best': None,
        'eval_loss_final': None, 'eval_loss_best': None, 'eval_samples_per_second_avg': None,
        'eval_steps_per_second_avg': None, 'eval_count': None,
        'learning_rate_initial': None, 'learning_rate_final': None,
        'grad_norm_avg': None, 'grad_norm_max': None,
        'total_flos': None, 'epochs_completed': None,
    }

    try:
        scalar_tags = ea.Tags().get('scalars', [])

        def _scalars(tag):
            return ea.Scalars(tag) if tag in scalar_tags else []

        evts = _scalars('train/loss')
        if evts:
            metrics['train_loss_initial']    = round(evts[0].value, 4)
            metrics['train_loss_final']      = round(evts[-1].value, 4)
            metrics['train_loss_improvement']= round(evts[0].value - evts[-1].value, 4)
            metrics['train_total_steps']     = len(evts)

        for tag, key in [
            ('train/train_samples_per_second', 'train_samples_per_second_avg'),
            ('train/train_steps_per_second',   'train_steps_per_second_avg'),
            ('eval/samples_per_second',        'eval_samples_per_second_avg'),
            ('eval/steps_per_second',          'eval_steps_per_second_avg'),
        ]:
            evts = _scalars(tag)
            if evts:
                metrics[key] = round(sum(e.value for e in evts) / len(evts), 2)

        evts = _scalars('eval/accuracy')
        if evts:
            metrics['eval_accuracy_final'] = round(evts[-1].value * 100, 2)
            metrics['eval_accuracy_best']  = round(max(e.value for e in evts) * 100, 2)
            metrics['eval_count']          = len(evts)

        evts = _scalars('eval/loss')
        if evts:
            metrics['eval_loss_final'] = round(evts[-1].value, 4)
            metrics['eval_loss_best']  = round(min(e.value for e in evts), 4)

        evts = _scalars('train/learning_rate')
        if evts:
            metrics['learning_rate_initial'] = evts[0].value
            metrics['learning_rate_final']   = evts[-1].value

        evts = _scalars('train/grad_norm')
        if evts:
            vals = [e.value for e in evts]
            metrics['grad_norm_avg'] = round(sum(vals) / len(vals), 4)
            metrics['grad_norm_max'] = round(max(vals), 4)

        evts = _scalars('train/total_flos')
        if evts:
            metrics['total_flos'] = round(evts[-1].value / 1e12, 2)

        evts = _scalars('train/epoch')
        if evts:
            metrics['epochs_completed'] = round(evts[-1].value, 2)

    except Exception as e:
        print(f"Warning: Error extracting performance metrics: {e}")

    return metrics


def process_log_folder(log_path: Path) -> Optional[Dict[str, Any]]:
    folder_name = log_path.name
    info = parse_log_folder_name(folder_name)

    event_files = list(log_path.glob('events.out.tfevents.*'))
    if not event_files:
        print(f"Warning: No event files found in {folder_name}")
        return None

    gpu_node = extract_gpu_node(str(event_files[0]))
    gpu_info = GPU_NODES.get(gpu_node, {}) if gpu_node else {}

    training_args, model_config = parse_text_summaries(log_path)

    try:
        ea = event_accumulator.EventAccumulator(str(log_path))
        ea.Reload()

        duration_seconds = duration_hours = duration_days = None
        if ea.Tags()['scalars']:
            evts = ea.Scalars(ea.Tags()['scalars'][0])
            if evts:
                duration_seconds = evts[-1].wall_time - evts[0].wall_time
                duration_hours   = duration_seconds / 3600
                duration_days    = duration_hours   / 24

        performance_metrics = extract_performance_metrics(ea)

        train_event_count = eval_event_count = 0
        for tag in ea.Tags().get('scalars', []):
            lower = tag.lower()
            try:
                cnt = len(ea.Scalars(tag))
            except Exception:
                cnt = 0
            is_train = 'train' in lower or 'training' in lower
            is_eval  = 'eval'  in lower or 'validation' in lower or 'val' in lower
            if is_train and not is_eval:
                train_event_count += cnt
            elif is_eval:
                eval_event_count += cnt

    except Exception as e:
        print(f"Error processing {folder_name}: {e}")
        return None

    base_model   = get_base_model_from_revision(info['revision'])
    model_details = get_model_details(base_model)

    efficiency: Dict[str, Any] = {}
    if performance_metrics['eval_accuracy_best'] and duration_hours:
        efficiency['accuracy_per_hour'] = round(
            performance_metrics['eval_accuracy_best'] / duration_hours, 2)
    if performance_metrics['eval_accuracy_best'] and model_details.get('parameters_m'):
        efficiency['accuracy_per_m_params'] = round(
            performance_metrics['eval_accuracy_best'] / model_details['parameters_m'], 4)
    if duration_hours and model_details.get('parameters_m'):
        efficiency['m_params_per_hour'] = round(
            model_details['parameters_m'] / duration_hours, 2)

    return {
        'folder_name': folder_name, 'script': info['script'], 'timestamp': info['timestamp'],
        'gpu_type': gpu_info.get('gpu_type'), 'gpu_ram': gpu_info.get('gpuram'),
        'model': base_model, 'model_family': model_details.get('family'),
        'parameters_m': model_details.get('parameters_m'), 'input_size': model_details.get('input_size'),
        'revision': info['revision'],
        'epochs': info['epochs']      or training_args.get('num_train_epochs'),
        'batch_size': info['batch_size'] or training_args.get('per_device_train_batch_size'),
        'learning_rate': info['learning_rate'] or training_args.get('learning_rate'),
        'duration_seconds': round(duration_seconds, 0) if duration_seconds else None,
        'duration_hours':   round(duration_hours,   2) if duration_hours   else None,
        'duration_days':    round(duration_days,    3) if duration_days    else None,
        'train_loss_initial':    performance_metrics['train_loss_initial'],
        'train_loss_final':      performance_metrics['train_loss_final'],
        'train_loss_improvement':performance_metrics['train_loss_improvement'],
        'train_samples_per_second': performance_metrics['train_samples_per_second_avg'],
        'train_steps_per_second':   performance_metrics['train_steps_per_second_avg'],
        'train_total_steps':        performance_metrics['train_total_steps'],
        'eval_accuracy_final':   performance_metrics['eval_accuracy_final'],
        'eval_accuracy_best':    performance_metrics['eval_accuracy_best'],
        'eval_loss_final':       performance_metrics['eval_loss_final'],
        'eval_loss_best':        performance_metrics['eval_loss_best'],
        'eval_samples_per_second': performance_metrics['eval_samples_per_second_avg'],
        'eval_steps_per_second':   performance_metrics['eval_steps_per_second_avg'],
        'eval_count':              performance_metrics['eval_count'],
        'learning_rate_initial':   performance_metrics['learning_rate_initial'],
        'learning_rate_final':     performance_metrics['learning_rate_final'],
        'grad_norm_avg':           performance_metrics['grad_norm_avg'],
        'grad_norm_max':           performance_metrics['grad_norm_max'],
        'total_flos_tflops':       performance_metrics['total_flos'],
        'epochs_completed':        performance_metrics['epochs_completed'],
        'accuracy_per_hour':       efficiency.get('accuracy_per_hour'),
        'accuracy_per_m_params':   efficiency.get('accuracy_per_m_params'),
        'm_params_per_hour':       efficiency.get('m_params_per_hour'),
        'train_event_count': train_event_count,
        'eval_event_count':  eval_event_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Extract training statistics from TensorBoard event log folders.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The --gpu-map and --revision-map files use simple JSON objects.

gpu_nodes.json example:
  {
    "my-gpu-node-1": {"gpu_type": "NVIDIA RTX 4090", "gpuram": "24G", "threads": 32}
  }
  Keys in this file are merged with (and override) the built-in UFAL cluster map.

revision_map.json example:
  {
    "v14": "timm/my-custom-model"
  }
  Keys in this file are merged with (and override) the built-in revision map.

Examples:
  python logs_stat.py ./logs
  python logs_stat.py ./logs -o stats.csv --pattern "run.py*"
  python logs_stat.py ./logs --gpu-map my_gpu_nodes.json --revision-map my_revisions.json
        """,
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help="Directory containing TensorBoard log sub-folders",
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='tensorboard_stats.csv',
        help="Output CSV filename (default: tensorboard_stats.csv)",
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*',
        help="Glob pattern to filter log sub-folders (default: *)",
    )
    # P4: external JSON overrides for institution-specific mappings
    parser.add_argument(
        '--gpu-map',
        type=str,
        default=None,
        metavar='JSON_FILE',
        help="JSON file with additional/override GPU node specs. "
             "Merged with the built-in UFAL cluster map; provided entries take priority.",
    )
    parser.add_argument(
        '--revision-map',
        type=str,
        default=None,
        metavar='JSON_FILE',
        help="JSON file with additional/override model revision → base model mappings. "
             "Merged with the built-in map; provided entries take priority.",
    )
    args = parser.parse_args()

    # ── Apply external JSON overrides to module-level dicts ──────────────
    global GPU_NODES, REVISION_TO_BASE_MODEL

    if args.gpu_map:
        with open(args.gpu_map, 'r', encoding='utf-8') as f:
            extra = json.load(f)
        GPU_NODES = {**GPU_NODES, **extra}
        print(f"Loaded {len(extra)} GPU node override(s) from {args.gpu_map}")

    if args.revision_map:
        with open(args.revision_map, 'r', encoding='utf-8') as f:
            extra = json.load(f)
        REVISION_TO_BASE_MODEL = {**REVISION_TO_BASE_MODEL, **extra}
        print(f"Loaded {len(extra)} revision override(s) from {args.revision_map}")

    # ── Locate and process log folders ───────────────────────────────────
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return

    log_folders = sorted([d for d in input_path.glob(args.pattern) if d.is_dir()])
    if not log_folders:
        print(f"No log folders matching '{args.pattern}' found in {args.input_dir}")
        return

    print(f"Found {len(log_folders)} log folder(s)")

    results = []
    for i, log_folder in enumerate(log_folders, 1):
        print(f"[{i}/{len(log_folders)}] {log_folder.name}")
        result = process_log_folder(log_folder)
        if result:
            results.append(result)

    if not results:
        print("No valid results extracted.")
        return

    all_columns: set = set()
    for r in results:
        all_columns.update(r.keys())

    fixed_columns = [
        'folder_name', 'script', 'timestamp', 'gpu_type', 'gpu_ram',
        'model', 'model_family', 'parameters_m', 'input_size', 'revision',
        'epochs', 'batch_size', 'learning_rate',
        'duration_hours', 'duration_days', 'duration_seconds',
        'eval_accuracy_best', 'eval_accuracy_final', 'eval_loss_best', 'eval_loss_final',
        'train_loss_initial', 'train_loss_final', 'train_loss_improvement',
        'train_samples_per_second', 'train_steps_per_second',
    ]
    extra_columns = sorted(c for c in all_columns if c not in fixed_columns)
    columns = fixed_columns + extra_columns

    output_path = Path(args.output)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"\nExtracted statistics from {len(results)} folder(s)")
    print(f"Output saved to: {output_path}  ({len(columns)} columns)")

    # ── Optional inline throughput/efficiency summary ─────────────────────
    df = pd.read_csv(output_path)
    df_filtered = df[
        df['train_samples_per_second'].notna() &
        df['eval_samples_per_second'].notna()  &
        df['accuracy_per_hour'].notna()        &
        df['parameters_m'].notna()
    ].copy()

    if df_filtered.empty:
        return

    def _param_range(p):
        if p < 70:   return '0-70M'
        if p < 100:  return '70-100M'
        if p < 200:  return '100-200M'
        return '200-350M'

    df_filtered['param_range'] = df_filtered['parameters_m'].apply(_param_range)
    row_order = ['0-70M', '70-100M', '100-200M', '200-350M']

    for metric, label in [
        ('train_samples_per_second', 'Average Training Throughput (samples/sec)'),
        ('eval_samples_per_second',  'Average Evaluation Throughput (samples/sec)'),
        ('accuracy_per_hour',        'Average Efficiency (accuracy per hour)'),
    ]:
        tbl = df_filtered.pivot_table(
            values=metric, index='param_range', columns='gpu_type', aggfunc='mean'
        ).reindex([r for r in row_order if r in df_filtered['param_range'].values])
        print(f"\n--- {label} ---")
        print(tbl.to_string(float_format='%.2f'))


if __name__ == '__main__':
    main()
