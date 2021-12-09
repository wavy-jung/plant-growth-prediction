import torch
from torch import optim
import random
import os
import wandb
import pandas as pd
from torch.utils.data import DataLoader
from torch import nn
from model import CompareNet
from dataloader import *
import argparse
from sklearn.model_selection import train_test_split

MODEL_SAVE_DIR = "./exp/"


def seed_everything(seed): # seed 고정
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def train(args):
    seed_everything(args.seed)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = CompareNet().to(device)

    total_dataframe = pd.read_csv("./train_dataset.csv")
    print(f"Length of Total Data : {len(total_dataframe)}")

    train_set, valid_set = train_test_split(
        total_dataframe,
        test_size=0.1,
        stratify=total_dataframe['time_delta'],
        random_state=args.seed)

    train_dataset = KistDataset(train_set)
    valid_dataset = KistDataset(valid_set)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda epoch: 0.9 ** epoch
    )

    train_data_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True)

    valid_data_loader = DataLoader(valid_dataset,
                                batch_size=args.batch_size,
                                shuffle=False)

    train_losses_avg = []
    valid_losses_avg = []
    for epoch in tqdm(range(args.epochs)):
        save_model = False
        train_losses = []
        model.train()
        print(f'\n===================EPOCH: {epoch+1}/{args.epochs}=====================')
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

        valid_losses_avg.append(sum(valid_losses)/len(valid_losses))
        print(f'VALIDATION_LOSS MAE : {sum(valid_losses)/len(valid_losses):.3f}')
        try:
            if min(valid_losses_avg[:-1]) > valid_losses_avg[-1]:
                checkpoint = {
                    'model': model.state_dict(),
                }
                ckpt = epoch+1
                # remove = glob(os.path.join(MODEL_SAVE_DIR, "*"))
                # for path in remove:
                #     if path.split(".")[-1] == "pt":
                #         print("Deleting Old Model")
                #         os.remove(path)
                torch.save(checkpoint, os.path.join(MODEL_SAVE_DIR, f'checkpoint-{ckpt}.pt'))
                print("New Model Saved")
        except:
            pass
            # if len(valid_losses_avg) == 0:
            #     checkpoint = {
            #         'model': model.state_dict(),
            #     }
            #     ckpt = (epoch+1)*len(train_data_loader)
            #     torch.save(checkpoint, f'./exp/checkpoint-{ckpt}.pt')


def main(args):
    train(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--lr', type=float, default=2e-5, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=30, help="num epochs for training")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size")
    parser.add_argument('--valid_batch_size', type=int, default=50, help="batch size vaildation")
    parser.add_argument('--seed', type=int, default=2048, help="set random seed")
    args = parser.parse_args()
    main(args)