from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
from datetime import datetime
from torchvision.transforms import v2
import torch
from transformers import AutoTokenizer
import glob
from tree import random_sample_by_rank
import random
import pandas as pd
import numpy as np


class ImageTextDataset(Dataset):
    def __init__(self, work_dir, csv_path, mode="train"):
        self.paths = glob.glob(os.path.join(work_dir, mode + "_folders", "*"))
        self.paths = [folder for folder in self.paths if os.path.isdir(folder)]
        self.paths_base = [os.path.basename(p) for p in self.paths]
        self.work_dir = work_dir
        self.csv = pd.read_csv(os.path.join(self.work_dir, csv_path))
        self.species_text = list(
            set(
                [
                    " ".join(d["folder_name"].split("_")[1:])
                    for i, d in self.csv.iterrows()
                ]
            )
        )
        if mode == "train":
            self.transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop((224, 224)),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.GaussianBlur(5, (0.01, 1.0)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        self.mode = mode

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        pos = {}
        neg = {}
        pos_list = []
        neg_list = []
        image_pos_list = []
        image_neg_list = []
        species_text = " ".join(self.csv.iloc[idx]["folder_name"].split("_")[1:])
        for i in range(7):
            rank = species_text.split(" ")
            pos[i], neg[i] = random_sample_by_rank(self.paths_base, rank, i)
            img_path_pos = random.choice(
                glob.glob(
                    os.path.join(self.work_dir, self.mode + "_folders", pos[i], "*")
                )
            )
            img_path_neg = random.choice(
                glob.glob(
                    os.path.join(self.work_dir, self.mode + "_folders", neg[i], "*")
                )
            )
            pos[i] = " ".join(pos[i].split("_")[1:][: i + 1])
            neg[i] = " ".join(neg[i].split("_")[1:][: i + 2])
            pos_list.append(pos[i])
            neg_list.append(neg[i])
            img_pos = self.transform(Image.open(img_path_pos))
            img_neg = self.transform(Image.open(img_path_neg))
            image_pos_list.append(img_pos)
            image_neg_list.append(img_neg)
        return (
            species_text,
            self.species_text.index(species_text),
            pos,
            neg,
            pos_list,
            neg_list,
            image_pos_list,
            image_neg_list,
        )


class ImageTextDatasetMERU(Dataset):
    def __init__(self, work_dir, csv_path, mode="train"):
        self.paths = glob.glob(os.path.join(work_dir, mode + "_folders", "*"))
        self.paths = [folder for folder in self.paths if os.path.isdir(folder)]
        self.paths_base = [os.path.basename(p) for p in self.paths]
        self.work_dir = work_dir
        self.csv = pd.read_csv(os.path.join(self.work_dir, csv_path))
        self.species_text = list(
            set(
                [
                    " ".join(d["folder_name"].split("_")[1:])
                    for i, d in self.csv.iterrows()
                ]
            )
        )
        if mode == "train":
            self.transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop((224, 224)),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.GaussianBlur(5, (0.01, 1.0)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        self.mode = mode

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        pos = {}
        neg = {}
        pos_list = []
        neg_list = []
        image_pos_list = []
        image_neg_list = []
        species_text = " ".join(self.csv.iloc[idx]["folder_name"].split("_")[1:])
        random_index = np.random.randint(0, 7)
        rank = species_text.split(" ")
        pos, neg = random_sample_by_rank(self.paths_base, rank, random_index)
        img_path_pos = random.choice(
            glob.glob(os.path.join(self.work_dir, self.mode + "_folders", pos, "*"))
        )
        img_path_neg = random.choice(
            glob.glob(os.path.join(self.work_dir, self.mode + "_folders", neg, "*"))
        )
        pos = " ".join(pos.split("_")[1:][: random_index + 1])
        neg = " ".join(neg.split("_")[1:][: random_index + 1])
        img_pos = self.transform(Image.open(img_path_pos))
        img_neg = self.transform(Image.open(img_path_neg))
        return (
            species_text,
            self.species_text.index(species_text),
            pos,
            neg,
            img_pos,
            img_neg,
        )


def collate_fn_meru(batch):
    species_text, species_index, pos, neg, img_pos, img_neg = zip(*batch)
    return species_text, species_index, pos, neg, img_pos, img_neg


def collate_fn(batch):
    pos_list = []
    neg_list = []
    image_pos_list = []
    image_neg_list = []

    pos_list = [item[-4] for item in batch]
    neg_list = [item[-3] for item in batch]
    image_pos_list = [torch.stack(item[-2]) for item in batch]
    image_neg_list = [torch.stack(item[-1]) for item in batch]

    return image_pos_list, image_neg_list, pos_list, neg_list


if __name__ == "__main__":
    dataset = ImageTextDataset(
        "bioclip/data/TreeOfLife-10M/dataset/evobio10m-dev/256x256",
        "train_folders/image_metadata.csv",
        mode="train",
    )
    print(len(dataset))
    batch = dataset[0]
    print(batch[0])
    print(batch[4])
    print()
    print(batch[5])

    dataset_meru = ImageTextDatasetMERU(
        "bioclip/data/TreeOfLife-10M/dataset/evobio10m-dev/256x256",
        "train_folders/image_metadata.csv",
        mode="train",
    )
    print(len(dataset_meru))
    batch_meru = dataset_meru[0]
    print(batch_meru[0])
    print(batch_meru[2])
    print()
    print(batch_meru[3])
