from __future__ import print_function
import os
import argparse
import json
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import MammographyModel
from dataset import CustomDataset

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Classification')
    parser.add_argument('--data_direc', type=str, default='./data', help="data directory")
    parser.add_argument('--n_classes', type=int, default=1, help="num of classes")
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=42, help='random seed to use. Default=42')
    parser.add_argument('--testBatchSize', type=int, default=4, help='test batch size')
    parser.add_argument('--model_save_path', type=str, default='./checkpoints', help='Path for save best model')
    opt = parser.parse_args()
    
    if not os.path.isdir(opt.model_save_path):
        raise Exception("checkpoints not found, please run train.py first")

    os.makedirs("test_results", exist_ok=True)
    
    print(opt)
    
    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    
    torch.manual_seed(opt.seed)
    
    if opt.cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print('===> Loading datasets')
    
    test_set = CustomDataset(f"{opt.data_direc}/test", mode='eval')
    test_dataloader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)
    
    print('===> Building model')
    model = MammographyModel(n_classes=opt.n_classes).to(device)
    model.load_state_dict(torch.load(os.path.join(opt.model_save_path, 'model_statedict.pth'), map_location=device))
    model.eval()

    with open(os.path.join(opt.model_save_path, 'metric_logger.json'), 'r') as f:
        metric_logger = json.load(f)
    
    criterion = nn.BCEWithLogitsLoss()
    total_test_num = len(test_dataloader.sampler)
    
    all_pred = []
    all_target = []
    
    all_sensitivity = []
    all_specificity = []
    all_auc = []

    with torch.no_grad():
        for data in tqdm(test_dataloader, total=len(test_dataloader), position=0, desc='Test', colour='green'):
            batch_num = len(data['input'])
        
            image = data['input'].to(device)
            target = data['target'].to(device)
            target = target.unsqueeze(1)

            pred_logit = model(image.float())
            pred = torch.sigmoid(pred_logit)

            all_pred.append(pred.cpu().numpy())
            all_target.append(target.cpu().numpy())

    all_pred = np.concatenate(all_pred)
    all_target = np.concatenate(all_target)

    # Threshold를 사용하여 민감도와 특이도 계산
    threshold = 0.5
    predicted_classes = (all_pred > threshold).astype(int)

    # 민감도(Sensitivity) 계산
    true_positives = np.sum((predicted_classes.flatten() == 1) & (all_target.flatten() == 1))
    false_negatives = np.sum((predicted_classes.flatten() == 0) & (all_target.flatten() == 1))
    sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    # 특이도(Specificity) 계산
    true_negatives = np.sum((predicted_classes.flatten() == 0) & (all_target.flatten() == 0))
    false_positives = np.sum((predicted_classes.flatten() == 1) & (all_target.flatten() == 0))
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0

    # AUC 계산
    auc = roc_auc_score(all_target.flatten(), all_pred.flatten())

    eval_df = pd.DataFrame({
        "Test Sensitivity": [sensitivity],
        "Test Specificity": [specificity],
        "Test AUC": [auc]
    })

    eval_df.to_csv(f"test_results/metric_df.csv", index=None)

    plt.figure()
    for k in ['train_loss', 'val_loss']:
        plt.plot(np.arange(len(metric_logger[k])), metric_logger[k], label=k)
    plt.title("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(f"test_results/learning_graph_loss.png", dpi=200)