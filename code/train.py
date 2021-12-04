import torch
from torch import optim
from random import random
import os
import wandb
import pandas as pd
from toroch.utils import DataLoader
from dataloader import *
import argparse


def seed_everything(seed): # seed 고정
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
lr = 1e-5
epochs = 10
batch_size = 64
valid_batch_size = 50

model = CompareNet().to(device)

total_dataframe = make_dataframe(root_path = "./open/train_dataset/")
bt_combination = make_combination(5000, 'bc', total_dataframe)
lt_combination = make_combination(5000, 'lt', total_dataframe)

bt_train = bt_combination.iloc[:4500]
bt_valid = bt_combination.iloc[4500:]

lt_train = lt_combination.iloc[:4500]
lt_valid = lt_combination.iloc[4500:]

train_set = pd.concat([bt_train, lt_train])
valid_set = pd.concat([bt_valid, lt_valid])



train_dataset = KistDataset(train_set)
valid_dataset = KistDataset(valid_set)

optimizer = optim.Adam(model.parameters(), lr=lr)

train_data_loader = DataLoader(train_dataset,
                               batch_size=batch_size,
                               shuffle=True)

valid_data_loader = DataLoader(valid_dataset,
                               batch_size=valid_batch_size)


for epoch in tqdm(range(epochs)):
    for step, (before_image, after_image, time_delta) in tqdm(enumerate(train_data_loader)):
        before_image = before_image.to(device)
        after_image = after_image.to(device)
        time_delta = time_delta.to(device)

        optimizer.zero_grad()
        logit = model(before_image, after_image)
        train_loss = (torch.sum(torch.abs(logit.squeeze(1).float() - time_delta.float())) /
                      torch.LongTensor([batch_size]).squeeze(0).to(device))
        train_loss.backward()
        optimizer.step()

        if step % 15 == 0:
            print('\n=====================loss=======================')
            print(f'\n=====================EPOCH: {epoch}=======================')
            print(f'\n=====================step: {step}=======================')
            print('MAE_loss : ', train_loss.detach().cpu().numpy())

    valid_losses = []
    with torch.no_grad():
        for valid_before, valid_after, time_delta in tqdm(valid_data_loader):
            valid_before = valid_before.to(device)
            valid_after = valid_after.to(device)
            valid_time_delta = time_delta.to(device)


            logit = model(valid_before, valid_after)
            valid_loss = (torch.sum(torch.abs(logit.squeeze(1).float() - valid_time_delta.float())) /
                          torch.LongTensor([valid_batch_size]).squeeze(0).to(device))
            valid_losses.append(valid_loss.detach().cpu())


    print(f'VALIDATION_LOSS MAE : {sum(valid_losses)/len(valid_losses)}')
    checkpoiont = {
        'model': model.state_dict(),

    }

    torch.save(checkpoiont, 'checkpoiont_128.pt')


if __name__ == "__main__":
    seed_everything(2048)

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--lr', type=float, default=1e-5, help='sum the integers (default: find the max)')

    args = parser.parse_args()