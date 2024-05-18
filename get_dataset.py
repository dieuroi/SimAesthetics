from pathlib import Path
from dataset import preprocess_AVA


def get_dataset_csv(string: str, path_to_save_csv: Path) -> Path:
    if string == "official":
        preprocess_AVA(Path("AVA_1"))
        return Path("AVA_1/preprocess")
        #return Path("AVA_1")
    elif string == "custom":
        preprocess_AVA(Path("AVA_2"))
        return Path("AVA_2/preprocess")
        #return Path("AVA_2")
    elif string == "use_path":
        return path_to_save_csv
