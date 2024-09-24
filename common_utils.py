import os
from pathlib import Path


# get list of all files in the folder and nested folders by file format
def directory_scraper(folder_path: Path, file_format: str = "png", file_list: list = None) -> list[Path]:
    if file_list is None:
        file_list = []
    file_list += list(folder_path.rglob(f"*.{file_format}"))
    print(f"From directory {folder_path} collected {len(file_list)} {file_format} files")
    return file_list
