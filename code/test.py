import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm.auto import tqdm
from dataloader import KistDataset
import os
from model import CompareNet

TEST_PATH = "../data/test_dataset/"
model = CompareNet()
checkpoint = torch.load("./exp/checkpoint-4944.pt")
model.load_state_dict(checkpoint['model'])
test_set = pd.read_csv(os.path.join(TEST_PATH, 'test_data.csv'))
submission = pd.read_csv(os.path.join(TEST_PATH, 'test_data.csv'))
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

test_set['l_root'] = test_set['before_file_path'].map(lambda x: TEST_PATH + x.split('_')[1] + '/' + x.split('_')[2])
test_set['r_root'] = test_set['after_file_path'].map(lambda x: TEST_PATH + x.split('_')[1] + '/' + x.split('_')[2])
test_set['l_path'] = test_set['l_root'] + '/' + test_set['before_file_path'] + '.png'
test_set['r_path'] = test_set['r_root'] + '/' + test_set['after_file_path'] + '.png'
test_set['before_file_path'] = test_set['l_path']
test_set['after_file_path'] = test_set['r_path']
test_dataset = KistDataset(test_set, is_test=True)
test_data_loader = DataLoader(test_dataset,
                               batch_size=64,
                               shuffle=False)
test_value = []
with torch.no_grad():
    model.eval()
    for test_before, test_after in tqdm(test_data_loader):
        test_before = test_before.to(device)
        test_after = test_after.to(device)
        logit = model(test_before, test_after)
        # value = logit.squeeze(1).detach().cpu().float()
        value = logit.detach().cpu().float()
        
        test_value.extend([float(v) for v in value])

test_value = [v if v >= 0 else 1 for v in test_value]
submission['time_delta'] = test_value
new_sub = pd.DataFrame({
    "idx" : submission['idx'],
    "time_delta" : test_value
})
new_sub.to_csv("./submission.csv", index=False)