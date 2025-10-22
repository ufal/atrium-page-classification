import shutil
import argparse
import configparser
from pathlib import Path
from PIL import Image
from typing import Tuple, List, Dict, Any
from datetime import datetime

Image.MAX_IMAGE_PIXELS = 721583750


def setup_folders(*folders: str) -> None:
    """Create multiple folders if they don't exist."""
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)


def get_image_files(folder: Path) -> List[Path]:
    """Get all image files from a given folder (non-recursively)."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}

    image_files = sorted([
        file for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in image_extensions
    ])
    return image_files


def process_tiff_image(img: Image.Image) -> Image.Image:
    """Process TIFF images, handling multiple frames and bit depths."""
    # Count frames
    try:
        n_frames = 0
        while True:
            img.seek(n_frames)
            n_frames += 1
    except EOFError:
        pass

    # Return single-frame TIFF as is
    if n_frames <= 1:
        img.seek(0)
        return img

    # Find best frame (with most content)
    best_frame, max_std = 0, 0
    for i in range(n_frames):
        img.seek(i)
        # Calculate standard deviation of pixel values in the grayscale version
        std_dev = sum(abs(px - 128) for px in img.convert('RGB').convert('L').getdata()) / (img.width * img.height)
        if std_dev > max_std:
            max_std, best_frame = std_dev, i

    img.seek(best_frame)
    return img


def convert_image_mode(img: Image.Image) -> Image.Image:
    """Convert image to appropriate mode for JPEG saving."""
    original_mode = img.mode

    # Convert based on mode
    if original_mode in ('I;16', 'I'):
        img = img.point(lambda i: i * (255.0 / 65535.0)).convert('L')
    elif original_mode == 'F':
        img = img.convert('RGB')
    elif original_mode in ('RGBA', 'LA') or (original_mode == 'P' and 'transparency' in img.info):
        # Handle transparency by pasting on a white background
        background = Image.new('RGB', img.size, (255, 255, 255))
        if original_mode == 'P':
            img = img.convert('RGBA')

        # Ensure image has an alpha channel before trying to use it as a mask
        if 'A' in img.getbands():
            background.paste(img, mask=img.getchannel('A'))
        else:
            background.paste(img)
        img = background
    elif original_mode not in ('RGB', 'L'):
        img = img.convert('RGB')

    return img


def process_single_image(image_path: Path, output_path: Path, quality: int = 90) -> Tuple[bool, Dict[str, Any]]:
    """Process a single image file and return success status and metadata.
       If output already exists, skip conversion but record metadata.
    """
    metadata = {
        'original_path': str(image_path),
        'output_path': str(output_path),
        'original_size': image_path.stat().st_size if image_path.exists() else 0,
        'success': False,
        'skipped_existing': False,
        'error': None,
        'original_format': None,
        'mode_before': None,
        'mode_after': None,
        'dimensions': None,
        'quality': quality,
        'compression_ratio': None,
        'output_size': None,
    }

    # Case 1: Output already exists -> just record metadata, skip conversion
    if output_path.exists():
        try:
            with Image.open(output_path) as out_img:
                metadata.update({
                    'original_format': out_img.format,
                    'mode_before': None,
                    'mode_after': out_img.mode,
                    'dimensions': out_img.size,
                    'output_size': output_path.stat().st_size,
                    'success': True,
                    'skipped_existing': True
                })
            if metadata['original_size'] > 0 and metadata['output_size']:
                metadata['compression_ratio'] = metadata['original_size'] / metadata['output_size']
        except Exception as e:
            metadata['error'] = f"Failed to read existing output: {e}"
        return False, metadata

    # Case 2: Convert image because output doesn’t exist
    try:
        with Image.open(image_path) as img:
            metadata.update({
                'original_format': img.format,
                'mode_before': img.mode,
                'dimensions': img.size
            })

            if image_path.suffix.lower() in ('.tiff', '.tif'):
                img = process_tiff_image(img)

            img = convert_image_mode(img)
            metadata['mode_after'] = img.mode

            # Save as JPEG
            img.save(output_path, 'JPEG', quality=quality)

        # After saving
        if output_path.exists():
            metadata['output_size'] = output_path.stat().st_size
            if metadata['original_size'] > 0:
                metadata['compression_ratio'] = metadata['original_size'] / metadata['output_size']
            metadata['success'] = True

        return True, metadata

    except Exception as e:
        metadata['error'] = str(e)
        return False, metadata



def img2jpeg_recursive(
        source_folder: str,
        backup_folder: str,
        output_folder: str,
        quality: int = 90
) -> Tuple[int, int, List[Dict[str, Any]]]:
    """
    Recursively finds all images in each subdirectory of source_folder,
    converts them to JPEG in batches per subdirectory,
    and moves originals to a mirrored directory structure in backup_folder.
    """
    source_path = Path(source_folder)
    backup_path_base = Path(backup_folder)
    output_path_base = Path(output_folder)

    conversion_count, skipped_count, metadata_list = 0, 0, []

    # Process each subdirectory including the root
    for subdir in [source_path] + [p for p in source_path.rglob('*') if p.is_dir()]:
        image_files = get_image_files(subdir)

        if not image_files:
            continue

        print(f"\tProcessing batch from '{subdir}': {len(image_files)} images")

        for image_path in image_files:
            relative_path = image_path.relative_to(source_path)
            output_jpeg_path = output_path_base / relative_path.with_suffix('.jpeg')
            backup_image_path = backup_path_base / relative_path

            setup_folders(str(output_jpeg_path.parent), str(backup_image_path.parent))

            success, metadata = process_single_image(image_path, output_jpeg_path, quality)
            metadata_list.append(metadata)

            print(f"At {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                  f"processed {len(metadata_list)} images total")

            if success:
                # Conversion done → move source to backup
                shutil.move(str(image_path), str(backup_image_path))
                conversion_count += 1
            elif metadata.get('skipped_existing'):
                shutil.move(str(image_path), str(backup_image_path))
                skipped_count += 1
                print(f"Skipped {image_path}, output already exists.")
            else:
                skipped_count += 1
                print(f"Failed to process {image_path}: {metadata.get('error')}")

    return conversion_count, skipped_count, metadata_list


def generate_report(metadata_list: List[Dict[str, Any]], output_file: str = None) -> None:
    """Generate a summary report from the processing metadata."""
    if not metadata_list:
        print("No images were processed - skipping report generation.")
        return

    # Calculate statistics
    total_files = len(metadata_list)
    successful = sum(1 for m in metadata_list if m.get('success', False))
    failed = total_files - successful
    total_original_size = sum(m.get('original_size', 0) for m in metadata_list)
    total_output_size = sum(m.get('output_size', 0) for m in metadata_list if m.get('success', False))

    # Build report string
    report = [
        "Image Conversion Report",
        "=====================",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total files processed: {total_files}",
        f"Successfully converted: {successful}",
        f"Failed conversions: {failed}",
        ""
    ]

    # Add storage statistics if data is available
    if total_original_size > 0 and total_output_size > 0:
        overall_ratio = total_original_size / total_output_size
        space_saved = total_original_size - total_output_size
        space_saved_mb = space_saved / (1024 * 1024)

        report.extend([
            "Storage Statistics:",
            f"Original total size: {total_original_size / (1024 * 1024):.2f} MB",
            f"Converted total size: {total_output_size / (1024 * 1024):.2f} MB",
            f"Space saved: {space_saved_mb:.2f} MB ({space_saved / total_original_size * 100:.1f}%)",
            f"Overall compression ratio: {overall_ratio:.2f}x",
            ""
        ])

    # Add successful conversions details
    if successful > 0:
        report.append("Successful Conversions:")
        for m in metadata_list:
            if m.get('success', False):
                compression = m.get('compression_ratio', 0)
                compression_info = f"Compression ratio = {compression:.2f}x" if compression > 0 else "N/A"
                orig_path = Path(m['original_path'])
                out_path = Path(m['output_path'])
                report.append(f"- {orig_path.name} ({m['original_size'] / 1024:.1f} KB) to "
                              f"{out_path.name} ({m.get('output_size', 0) / 1024:.1f} KB) | {compression_info}")
        report.append("")

    # Add failed conversions details
    if failed > 0:
        report.append("Failed Conversions:")
        report.extend(
            [f"- {Path(m['original_path']).name}: {m.get('error', 'Unknown error')}"
             for m in metadata_list if not m.get('success', False)])
        report.append("")

    # Print and save report
    report_text = "\n".join(report)
    print("\n" + report_text)

    if output_file:
        Path(output_file).write_text(report_text, encoding='utf-8')
        print(f"Report saved to {output_file}")


if __name__ == "__main__":
    # Read configuration file
    config = configparser.ConfigParser()
    config.read('data_config.txt')

    # Set default values from config file or use hardcoded fallbacks
    default_source = config.get('FOLDERS', 'SOURCE', fallback="input")
    default_backup = config.get('FOLDERS', 'BACKUP', fallback="backup")
    default_output = config.get('FOLDERS', 'OUTPUT', fallback="converted")
    default_report = config.get('OUTPUT', 'REPORT_FILE', fallback="report.txt")
    default_quality = config.getint('SETTINGS', 'JPEG_QUALITY', fallback=90)

    # Initialize argument parser
    parser = argparse.ArgumentParser(description='Convert images in subdirectories to JPEG format with compression.')
    parser.add_argument('-s', '--source', type=str, default=default_source,
                        help=f"Source folder containing image subdirectories (default: {default_source})")
    parser.add_argument('-b', '--backup', type=str, default=default_backup,
                        help=f"Backup folder for original images (default: {default_backup})")
    parser.add_argument('-o', '--output', type=str, default=default_output,
                        help=f"Output folder for converted JPEGs (default: {default_output})")
    parser.add_argument('-r', '--report', type=str, default=default_report,
                        help=f"Report file path (default: {default_report})")
    parser.add_argument('-q', '--quality', type=int, default=default_quality,
                        help=f"JPEG quality (1-100, default: {default_quality})")

    args = parser.parse_args()

    # Run conversion process with command line arguments
    print("Starting recursive image conversion process...")
    converted, skipped, metadata = img2jpeg_recursive(args.source, args.backup, args.output, args.quality)

    if metadata:
        generate_report(metadata, args.report)

    # Summary
    print("\n---")
    print(f"Conversion completed: {converted} images converted, {skipped} skipped.")
    print(f"Original images backed up to '{args.backup}' with original folder structure.")
    print(f"Converted JPEGs saved to '{args.output}' with original folder structure.")