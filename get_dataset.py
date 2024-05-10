from pathlib import Path


def get_dataset_csv(string: str, path_to_save_csv: Path) -> Path:
    if string == "official":
        return Path("AVA_1")
    elif string == "custom":
        return Path("AVA_2")
    elif string == "use_path":
        return path_to_save_csv
