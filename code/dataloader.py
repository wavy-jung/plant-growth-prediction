from glob import glob
from itertools import combinations
from tqdm.auto import tqdm
import os

import pandas as pd
import numpy as np
# import transformers
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision import transforms


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


# def make_dataframe(root_path=None):
#     bc_image_path, lt_image_path = make_image_path_array(root_path)
#     bc_day_array = make_day_array(bc_image_path)
#     lt_day_array = make_day_array(lt_image_path)

#     bc_df = pd.DataFrame({'file_name': bc_image_path,
#                           'day': bc_day_array})
#     bc_df['species'] = 'bc'

#     lt_df = pd.DataFrame({'file_name': lt_image_path,
#                           'day': lt_day_array})
#     lt_df['species'] = 'lt'

#     total_data_frame = pd.concat([bc_df, lt_df]).reset_index(drop=True)

#     return total_data_frame


# def make_combination(length, species, data_frame):
#     before_file_path = []
#     after_file_path = []
#     time_delta = []

#     for i in range(length):
#         sample = data_frame[data_frame['species'] == species].sample(2)
#         after = sample[sample['day'] == max(sample['day'])].reset_index(drop=True)
#         before = sample[sample['day'] == min(sample['day'])].reset_index(drop=True)

#         before_file_path.append(before.iloc[0]['file_name'])
#         after_file_path.append(after.iloc[0]['file_name'])
#         delta = int(after.iloc[0]['day'] - before.iloc[0]['day'])
#         time_delta.append(delta)

#     combination_df = pd.DataFrame({
#         'before_file_path': before_file_path,
#         'after_file_path': after_file_path,
#         'time_delta': time_delta,
#     })

#     combination_df['species'] = species

#     return combination_df


class KistDataset(Dataset):
    def __init__(self, combination_df, is_test= None):
        self.combination_df = combination_df
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ])
        self.is_test = is_test
        # self.images = 

    def __getitem__(self, idx):
        before_image = Image.open(self.combination_df.iloc[idx]['before_file_path'])
        after_image = Image.open(self.combination_df.iloc[idx]['after_file_path'])

        before_image = self.transform(before_image)
        after_image = self.transform(after_image)
        if self.is_test:
            return before_image, after_image
        time_delta = self.combination_df.iloc[idx]['time_delta']
        return before_image, after_image, time_delta

    def __len__(self):
        return len(self.combination_df)


if __name__ == "__main__":
    get_dataset()