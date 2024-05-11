import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision.datasets.folder import default_loader
from tqdm import tqdm

logger = logging.getLogger(__file__)


def _remove_all_not_found_image(df: pd.DataFrame, path_to_images: Path) -> pd.DataFrame:
    clean_rows = []
    for _, row in df.iterrows():
        image = row["image"]
        try:
            file_name = path_to_images / image
            _ = default_loader(file_name)
        except (FileNotFoundError, OSError, UnboundLocalError) as ex:
            logger.info(f"broken image {file_name} : {ex}")
        else:
            clean_rows.append(row)
    df_clean = pd.DataFrame(clean_rows)
    return df_clean


def remove_all_not_found_image(df: pd.DataFrame, path_to_images: Path, num_workers: int) -> pd.DataFrame:
    futures = []
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for df_batch in np.array_split(df, num_workers):
            future = executor.submit(_remove_all_not_found_image, df=df_batch, path_to_images=path_to_images)
            futures.append(future)
        for future in tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())
    new_df = pd.concat(results)
    return new_df


def read_ava_txt(path_to_ava: Path) -> pd.DataFrame:
    # NIMA origin file format and indexes
    df = pd.read_csv(path_to_ava / "AVA.txt", header=None, sep=" ")
    del df[0]
    score_first_column = 2
    score_last_column = 12
    tag_first_column = 1
    tag_last_column = 4
    score_names = [f"score{i}" for i in range(score_first_column, score_last_column)]
    tag_names = [f"tag{i}" for i in range(tag_first_column, tag_last_column)]
    df.columns = ["image"] + score_names + tag_names
    # leave only score columns
    df = df[["image"] + score_names]
    df["image"] = df["image"].astype("str")
    for i in range(df.shape[0]):
        df.iloc[i, 0] = df.iloc[i, 0] + ".jpg"
    return df


def spilt(df, train_size):
    logger.info("train val test split")
    df_train, df_val_test = train_test_split(df, train_size=train_size)
    df_val, df_test = train_test_split(df_val_test, train_size=0.5)
    return df_train, df_val, df_test


def read_TAD66K_csv(path_to_dataset: Path):
    df_test = pd.read_csv(path_to_dataset / "test.csv", header=0, sep=",")
    TAD66K_train_val = pd.read_csv(path_to_dataset / "train.csv", header=0, sep=",")
    df_train, df_val = train_test_split(TAD66K_train_val, test_size=0.2)
    return df_train, df_val, df_test


def read_EVA_csv(path_to_dataset: Path):
    df = pd.read_csv(path_to_dataset / "votes_filtered.csv", header=0, sep="=")
    df = df.filter(items=["image_id", "score"])
    df = df.rename(columns={'image_id': 'image'})
    df["image"] = df["image"].astype("str")
    for i in range(df.shape[0]):
        df.iloc[i, 0] = df.iloc[i, 0] + ".jpg"
    return df


def clean_and_split(
        dataset: str, path_to_dataset: Path, path_to_save_csv: Path, path_to_images: Path, train_size: float,
        num_workers: int
):
    if dataset == "official" or dataset == "custom":
        logger.info("read ava txt")
        df = read_ava_txt(path_to_dataset)
        logger.info("removing broken images")
        df = remove_all_not_found_image(df, path_to_images, num_workers=num_workers)
        df_train, df_val, df_test = spilt(df, train_size)
    elif dataset == "TAD66K":
        logger.info("read TAD66K csv")
        df_train, df_val, df_test = read_TAD66K_csv(path_to_dataset)
        logger.info("removing broken images")
        df_train = remove_all_not_found_image(df_train, path_to_images, num_workers=num_workers)
        df_val = remove_all_not_found_image(df_val, path_to_images, num_workers=num_workers)
        df_test = remove_all_not_found_image(df_test, path_to_images, num_workers=num_workers)
    elif dataset == "EVA":
        logger.info("read EVA csv")
        df = read_EVA_csv(path_to_dataset)
        logger.info("removing broken images")
        df = remove_all_not_found_image(df, path_to_images, num_workers=num_workers)
        df_train, df_val, df_test = spilt(df, train_size)
    train_path = path_to_save_csv / "train.csv"
    val_path = path_to_save_csv / "val.csv"
    test_path = path_to_save_csv / "test.csv"
    logger.info(f"saving to {train_path} {val_path} and {test_path}")
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)
