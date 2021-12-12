import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import random
import os
import wandb
import pandas as pd
import numpy as np
import argparse
import time
from glob import glob
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split, KFold

from model import CompareNet
from dataloader import KistDataset


MODEL_SAVE_DIR = "./exp/"


def seed_everything(seed): # seed 고정
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def create_dir(base_dir=MODEL_SAVE_DIR):
    now = time.localtime()
    path_name = f"{now.tm_year}-{now.tm_mon}-{now.tm_mday}-{now.tm_hour}-{now.tm_min}"
    if not os.path.isdir(os.path.join(MODEL_SAVE_DIR, path_name)):
        os.mkdir(os.path.join(MODEL_SAVE_DIR, path_name))

    return os.path.join(MODEL_SAVE_DIR, path_name), path_name


def main(args):
    seed_everything(args.seed)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    base_path, time_exp = create_dir()

    total_dataframe = pd.read_csv("./train_dataset.csv")
    print(f"Length of Total Data : {len(total_dataframe)}")

    # K-Fold Cross Validation
    dataset = pd.read_csv("./train_dataset.csv")
    kfold = KFold(n_splits=5, shuffle=True, random_state=args.seed)

    for fold, (train_set, valid_set) in enumerate(kfold.split(dataset)):

        print("-"*100)
        print(f"Fold-{fold+1} Training")
        print("-"*100)

        fold_path = os.path.join(base_path, f"fold{fold+1}")
        if not os.path.isdir(fold_path):
            os.mkdir(fold_path)

        train_dataset = KistDataset(dataset.iloc[train_set])
        valid_dataset = KistDataset(dataset.iloc[valid_set], is_valid=True)

        train_data_loader = DataLoader(train_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True)

        valid_data_loader = DataLoader(valid_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False)

        model = CompareNet(model_name=args.pretrained).to(device)
        wandb.init(
            project="plant-growth",
            group=time_exp,
            config={"base_model" : model.model_name, "fold" : fold+1},
            name=f'fold-{fold+1}'
            )
        wandb.config.update(args)
        wandb.watch(model)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda epoch: 0.9 ** epoch
        )

        train_losses_avg = []
        valid_losses_avg = []

        for epoch in tqdm(range(args.epochs)):
            # Training
            train_losses = []
            model.train()
            print(f'\n===================FOLD-{fold+1} EPOCH: [{epoch+1}/{args.epochs}]=====================')
            for before_image, after_image, time_delta in tqdm(train_data_loader, desc="Training"):
                before_image = before_image.to(device)
                after_image = after_image.to(device)
                time_delta = time_delta.to(device)

                optimizer.zero_grad()
                logit = model(before_image, after_image)
                train_loss = (torch.sum(torch.abs(logit.squeeze(1).float() - time_delta.float())) /
                        torch.LongTensor([args.batch_size]).squeeze(0).to(device))
                train_loss.backward()
                train_losses.append(train_loss.detach().cpu())
                optimizer.step()

            scheduler.step()
            train_losses_avg.append(float(sum(train_losses)/len(train_losses)))
            print(f'TRAIN_MAE_loss : {train_losses_avg[-1]:.3f}')
            
            # Validation
            valid_losses = []
            with torch.no_grad():
                model.eval()
                for valid_before, valid_after, time_delta in tqdm(valid_data_loader, desc="Validation"):
                    valid_before = valid_before.to(device)
                    valid_after = valid_after.to(device)
                    valid_time_delta = time_delta.to(device)

                    logit = model(valid_before, valid_after)
                    valid_loss = (torch.sum(torch.abs(logit.squeeze(1).float() - valid_time_delta.float())) /
                            torch.LongTensor([args.batch_size]).squeeze(0).to(device))
                    valid_losses.append(valid_loss.detach().cpu())

            valid_losses_avg.append(float(sum(valid_losses)/len(valid_losses)))
            print(f'VALIDATION_LOSS MAE : {valid_losses_avg[-1]:.3f}')
            wandb.log({"train_loss" : train_losses_avg[-1],
                        "valid_loss" : valid_losses_avg[-1]})

            if epoch+1 >= 5:
                if min(valid_losses_avg[:-1]) > valid_losses_avg[-1]:
                    checkpoint = {
                        'model': model.state_dict(),
                    }

                    ckpt = epoch+1
                    pt_list = glob(os.path.join(fold_path, "*"))
                    pt_list = sorted([r for r in pt_list if r.split(".")[-1]=="pt"], key=lambda x: -int(x.split("-")[-1].split(".")[0]))
                    if len(pt_list) == 1:
                        print("Deleting Old Model")
                        os.remove(pt_list[-1])
                    torch.save(checkpoint, os.path.join(fold_path, f'checkpoint-fold{fold+1}-{ckpt}.pt'))
                    print("New Model Saved")
                    
        # wandb.finish() -> re-initialize after fold ends
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--pretrained', type=str, default='regnetx_016', help='pretrained model selection')
    parser.add_argument('--lr', type=float, default=2e-5, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=30, help="num epochs for training")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size")
    parser.add_argument('--seed', type=int, default=2048, help="set random seed")
    args = parser.parse_args()
    main(args)