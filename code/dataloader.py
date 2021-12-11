from glob import glob
from itertools import combinations
from tqdm.auto import tqdm
import os

import pandas as pd
import numpy as np

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision import transforms

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2


def extract_day(file_name):
    day = int(file_name.split('.')[-2][-2:])
    return day


def make_day_array(image_pathes):
    day_array = np.array([extract_day(file_name) for file_name in image_pathes])
    return day_array


def get_combination(image_pathes):
    combination = [sorted(comb) for comb in list(map(list, combinations(image_pathes, 2)))]
    return combination


def get_time_delta(days_array):
    after, before = max(days_array), min(days_array)
    return int(after-before)


def get_image_directories(root_path=None):
    if root_path is None:
        root_path = "../data/train_dataset/"
    bc_directories = glob(os.path.join(root_path + 'BC/', "*"))
    lt_directories = glob(os.path.join(root_path + 'LT/', "*"))
    return bc_directories, lt_directories


def get_pair(directories):
    set_path = []
    set_delta = []
    for directory in tqdm(directories):
        img_path = glob(os.path.join(directory, "*"))
        combinations = get_combination(img_path)
        time_deltas = [get_time_delta(make_day_array(path)) for path in combinations]
        set_path.extend(combinations)
        set_delta.extend(time_deltas)
    return set_path, set_delta


def get_dataset(root_path="../data/train_dataset/"):
    bc_image_path, lt_image_path = get_image_directories(root_path)
    print(bc_image_path, lt_image_path)
    bc_imgs, bc_deltas = get_pair(bc_image_path)
    lt_imgs, lt_deltas = get_pair(lt_image_path)
    species = ["bc"]*len(bc_deltas) + ["lt"]*len(lt_deltas)
    assert len(bc_imgs)+len(lt_imgs)==len(bc_deltas)+len(lt_deltas), "Wrong in Length"
    print(f"Length of Species : {len(species)}")
    df = pd.DataFrame({
        "before_file_path" : [imgs[0] for imgs in bc_imgs] + [imgs[0] for imgs in lt_imgs],
        "after_file_path" : [imgs[1] for imgs in bc_imgs] + [imgs[1] for imgs in lt_imgs],
        "time_delta" : bc_deltas + lt_deltas,
        "species" : species
    })
    print(f"Length of Dataset : {len(df)}")
    df.to_csv("./train_dataset.csv", index=False)
    print("train_dataset.csv saved!")
    return df


class KistDataset(Dataset):

    def __init__(self, combination_df, is_valid=None, is_test=None):

        self.combination_df = combination_df
        self.is_valid = is_valid
        self.is_test = is_test
        self.rand_aug_factor = 0.5
        if self.is_valid or self.is_test:
            self.rand_aug_factor = 0

        self.transform = A.Compose([
            A.Resize(224, 224),
            A.OneOf([A.HorizontalFlip(p=1),
                     A.RandomRotate90(p=1),
                     A.VerticalFlip(p=1)            
            ], p=self.rand_aug_factor),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        self.before_image = [cv2.cvtColor(cv2.imread(before_path), cv2.COLOR_BGR2RGB) for before_path in tqdm(list(self.combination_df['before_file_path']))]
        self.after_image = [cv2.cvtColor(cv2.imread(after_path), cv2.COLOR_BGR2RGB) for after_path in tqdm(list(self.combination_df['after_file_path']))]
        
        if not self.is_test:    
            self.time_delta = list(self.combination_df['time_delta'])


    def __getitem__(self, idx):
        before_image = self.transform(image=self.before_image[idx])['image']
        after_image =self.transform(image=self.after_image[idx])['image']
        if self.is_test:
            return before_image, after_image
        time_delta = self.time_delta[idx]
        return before_image, after_image, torch.tensor(time_delta)


    def __len__(self):
        return len(self.combination_df)



if __name__ == "__main__":
    get_dataset()