import torch
from torch.utils import DataLoader
import pandas as pd
from tqdm.auto import tqdm
from dataloader import KistDataset

TEST_PATH = "../data/test_dataset/"

test_set = pd.read_csv(TEST_PATH + 'test_data.csv')
test_set['l_root'] = test_set['before_file_path'].map(lambda x: TEST_PATH + x.split('_')[1] + '/' + x.split('_')[2])
test_set['r_root'] = test_set['after_file_path'].map(lambda x: TEST_PATH + x.split('_')[1] + '/' + x.split('_')[2])
test_set['l_path'] = test_set['l_root'] + '/' + test_set['before_file_path'] + '.png'
test_set['r_path'] = test_set['r_root'] + '/' + test_set['after_file_path'] + '.png'
test_dataset = KistDataset(test_set, is_test=True)
test_data_loader = DataLoader(test_dataset,
                               batch_size=64)
test_value = []
with torch.no_grad():
    for test_before, test_after in tqdm(test_data_loader):
        test_before = test_before.to(device)
        test_after = test_after.to(device)
        logit = model(test_before, test_after)
        value = logit.squeeze(1).detach().cpu().float()
        
        test_value.extend(value)