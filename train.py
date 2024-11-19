import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import MammographyModel
from dataset import CustomDataset
from util import EarlyStopping
from loss import FocalLoss

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Classification')
    parser.add_argument('--data_direc', type=str, default='./data', help="data directory")
    parser.add_argument('--n_classes', type=int, default=1, help="num of classes")
    parser.add_argument('--batchSize', type=int, default=32, help='training batch size')
    parser.add_argument('--total_epoch', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.001')
    parser.add_argument('--lr_schedule_patience', type=int, default=10, help='Learning Rate schedule patience. Default=10')
    parser.add_argument('--earlystop_patience', type=int, default=20, help='Earlystop patience. Default=20')
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=42, help='random seed to use. Default=123')
    parser.add_argument('--model_save_path', type=str, default='./checkpoints', help='Path for saving the best model')
    opt = parser.parse_args()

    os.makedirs(opt.model_save_path, exist_ok=True)

    print(opt)

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)

    if opt.cuda:
        device = torch.device("cuda:1")
    else:
        device = torch.device("cpu")

    print('===> Loading datasets')

    train_set = CustomDataset(f"{opt.data_direc}/train", mode='train')
    test_set = CustomDataset(f"{opt.data_direc}/val", mode='eval')
    train_dataloader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    val_dataloader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)

    for data in train_dataloader:
        image = data['input']
        print(f"Image shape: {image.shape}")
        break

    print('===> Building model')
    model = MammographyModel(n_classes=opt.n_classes).to(device)

    bce_logits_loss = nn.BCEWithLogitsLoss()
    focal_loss = FocalLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=opt.lr_schedule_patience)
    monitor = EarlyStopping(patience=opt.earlystop_patience, verbose=True, path=os.path.join(opt.model_save_path, 'model.pth'))

    metric_logger = {k: [] for k in [
        'train_loss', 'val_loss', 'lr', 
        'train_bce_loss', 
        'val_bce_loss',
        'train_focal_loss',
        'val_focal_loss'
    ]}
    total_train_num = len(train_dataloader.sampler)
    total_val_num = len(val_dataloader.sampler)

    for epoch in range(opt.total_epoch):
        for param in optimizer.param_groups:
            lr_status = param['lr']
        metric_logger['lr'].append(lr_status)

        epoch_loss = {'train_loss': 0, 'val_loss': 0}
        train_bce_loss = 0
        val_bce_loss = 0
        train_focal_loss = 0
        val_focal_loss = 0

        print(f"Epoch {epoch+1:03d}/{opt.total_epoch:03d}\tLR: {lr_status:.0e}")

        # Training phase
        model.train()
        for data in tqdm(train_dataloader, total=len(train_dataloader), position=0, desc='Train', colour='blue'):
            batch_num = len(data['input'])
            image = data['input'].to(device)
            target = data['target'].to(device)
            target = target.unsqueeze(1)

            pred = model(image.float())

            bce_loss = bce_logits_loss(pred, target.float())
            focal_loss_value = focal_loss(pred, target.float())

            loss = bce_loss + focal_loss_value

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss['train_loss'] += loss.item() * batch_num
            train_bce_loss += bce_loss.item() * batch_num
            train_focal_loss += focal_loss_value.item() * batch_num

        model.eval()
        with torch.no_grad():
            for data in tqdm(val_dataloader, total=len(val_dataloader), position=0, desc='Val', colour='green'):
                batch_num = len(data['input'])
                image = data['input'].to(device)
                target = data['target'].to(device)
                target = target.unsqueeze(1)

                pred = model(image.float())

                val_loss = bce_logits_loss(pred, target.float())
                val_focal_loss_value = focal_loss(pred, target.float())

                val_loss_combined = val_loss + val_focal_loss_value

                epoch_loss['val_loss'] += val_loss_combined.item() * batch_num
                val_bce_loss += val_loss.item() * batch_num
                val_focal_loss += val_focal_loss_value.item() * batch_num

        epoch_loss = {k: (v / total_train_num if 'train' in k else v / total_val_num) for k, v in epoch_loss.items()}
        metric_logger['train_loss'].append(epoch_loss['train_loss'])
        metric_logger['val_loss'].append(epoch_loss['val_loss'])
        metric_logger['train_bce_loss'].append(train_bce_loss / total_train_num)
        metric_logger['val_bce_loss'].append(val_bce_loss / total_val_num)
        metric_logger['train_focal_loss'].append(train_focal_loss / total_train_num)
        metric_logger['val_focal_loss'].append(val_focal_loss / total_val_num)

        monitor(epoch_loss['val_loss'], model)
        if monitor.early_stop:
            print(f"Train early stopped, Minimum validation loss: {monitor.val_loss_min}")
            break

        scheduler.step(epoch_loss['val_loss'])

        print(f"Train loss: {epoch_loss['train_loss']:.7f}\tVal loss: {epoch_loss['val_loss']:.7f}")

        with open(os.path.join(opt.model_save_path, 'metric_logger.json'), 'w') as f:
            json.dump(metric_logger, f)