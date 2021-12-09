import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from dataloader import KistDataset
import os
from model import CompareNet
import argparse


def main(args):
    TEST_PATH = "../data/test_dataset/"
    MODEL_PATH = os.path.join("./exp/", args.model_name)
    assert os.path.exists(MODEL_PATH), "Wrong Model Path"
    assert args.submission_file.split(".")[-1] == "csv", "Wrong Output File Name"
    assert args.label_type.lower() in ["int", "float"], "Choose Label Type : int or float"

    # Load Model
    model = CompareNet()
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['model'])
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Make Inference Data
    test_set = pd.read_csv(os.path.join(TEST_PATH, 'test_data.csv'))
    submission = pd.read_csv(os.path.join(TEST_PATH, 'test_data.csv'))

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
            value = logit.squeeze(1).detach().cpu().float()

            test_value.extend(value)

    torch_label = torch.FloatTensor(test_value)

    np_label = torch_label.numpy()
    np_label[np.where(np_label<1)] = 1

    if args.label_type.lower() == "int":
        label = [round(n) for n in np_label]
    elif args.label_type.lower() == "float":
        label = np_label

    new_sub = pd.DataFrame({
        "idx" : submission['idx'],
        "time_delta" : label
    })
    new_sub.to_csv(args.submission_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="inference setter")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--submission_file", type=str, default="./submission.csv")
    parser.add_argument("--label_type", type=str, default="int")
    args = parser.parse_args()
    main(args)