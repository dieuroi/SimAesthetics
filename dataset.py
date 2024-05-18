import logging
from pathlib import Path
from typing import Tuple

import os
import numpy as np
import pandas as pd
import torch
import pickle
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from PIL import Image


class AVADataset(Dataset):
    def __init__(self, path_to_csv: Path, images_path: Path, transform):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        self.transform = transform
        with open('data/ratio_dict.pkl', 'rb') as f:
            self.ratios = pickle.load(f)

    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, item: int):
        row = self.df.iloc[item]

        image = row["image"]
        ratio = np.array(self.ratios.get(image, 1.0)).astype('float32')

        image_path = self.images_path / image
        image = default_loader(image_path)
        x = self.transform(image)

        y = row[1:].values.astype("float32")
        p = y / y.sum()

        return x, p, ratio


def preprocess_AVA(path_to_csv: Path):
    csvs = os.listdir(path_to_csv)
    csvs = [csv for csv in csvs if csv.endswith(".csv")]
    for csv in csvs:
        df = pd.read_csv(path_to_csv / csv)
        df = df.rename(columns={'image_id': 'image'})
        df["image"] = df["image"].astype("str")
        for i in range(df.shape[0]):
            df.iloc[i, 0] = df.iloc[i, 0] + ".jpg"
        if not os.path.exists(path_to_csv/"preprocess"):
            os.makedirs(path_to_csv/"preprocess")
        df.to_csv(path_to_csv/"preprocess"/csv, index=False)


def preprocess(file_dir):
    ratio_dict = dict()
    imgs = os.listdir(file_dir)
    img_files = [os.path.join(file_dir, img) for img in imgs]

    for img, img_file in zip(imgs, img_files):
        if 'jpg' not in img_file and 'jpeg' not in img_file:
            continue
        try:
            img_obj = Image.open(img_file)
            w, h = img_obj.size
            ratio_dict[img.split('.')[0]] = w / h
        except:
            print(img)

    with open('data/ratio_dict.pkl', 'wb') as f:
        pickle.dump(ratio_dict, f)


if '__main__' == __name__:
    preprocess('/home/bnu/SimAesthetics/ava_dataset/images')
