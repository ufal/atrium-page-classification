import os

# get list of all files in the folder and nested folders by file format
def directory_scraper(folder_path: str, file_list=[], format="png") -> list[str]:
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(f".{format}"):
                file_list.append(f"{root}/{file}")

    print(f"From directory {folder_path} collected {len(file_list)} {format} files")
    return file_list