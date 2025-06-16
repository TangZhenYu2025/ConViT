
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
import sklearn.metrics as sm
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'

def train_model(model, output_path, train_loader, valid_loader, num_epochs, optimizer, ce, patience=100):
    train_losses = []
    valid_losses = []
    train_r2= []
    valid_r2 = []
    epoch_times = []
    highest_r2 = []
    best_r2 = -100
    early_stopping = 0
    best_labels = []
    best_preds = []

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for epoch in range(num_epochs):
        start_time = time.time()
        losses = []
        labels=[]
        preds=[]

        # Training phase
        for inputs, targets in tqdm(train_loader):
            inputs = inputs.to(torch.device(device))
            targets = targets.type(torch.float).to(torch.device(device))
            outputs = model(inputs).squeeze()
            loss = ce(outputs, targets)
            losses.append(loss)
            labels.append(targets)
            preds.append(outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_losses.append((sum(losses) / len(losses)).cpu().tolist())
        labels = torch.cat(labels, dim=0)
        preds = torch.cat(preds, dim=0)
        T_r2 = sm.r2_score(labels.cpu().detach().numpy(), preds.cpu().detach().numpy())
        train_r2.append(T_r2)

        # Validation phase
        labels = []
        preds = []
        losses = []
        with torch.no_grad():
            for inputs, targets in tqdm(valid_loader):
                inputs = inputs.to(torch.device(device))
                targets = targets.type(torch.float).to(torch.device(device))
                outputs = model(inputs).squeeze()
                loss = ce(outputs, targets)
                losses.append(loss)
                labels.append(targets)
                preds.append(outputs)

        valid_losses.append((sum(losses) / len(losses)).cpu().tolist())
        labels = torch.cat(labels, dim=0)
        preds = torch.cat(preds, dim=0)
        r2 = sm.r2_score(labels.cpu().numpy(), preds.cpu().numpy())
        valid_r2.append(r2)

        # Track highest R2 score achieved
        highest_r2.append(max(valid_r2))
        if r2 > best_r2:
            best_r2 = r2
            best_labels = labels.cpu().numpy()
            best_preds = preds.cpu().numpy()
            early_stopping = 0
            torch.save(model.state_dict(), os.path.join(output_path, 'best.pth'))
        else:
            early_stopping += 1

        end_time = time.time()
        epoch_time = end_time - start_time
        epoch_times.append(epoch_time)

        print('Epoch {}/{} | Train Loss: {} | Valid Loss: {} | R2 Score: {} | Epoch Time: {:.2f} seconds'.format(
            epoch + 1, num_epochs, train_losses[-1], valid_losses[-1], r2, epoch_time))


        # if early_stopping > patience:
        #     break
    df_train_log = pd.DataFrame.from_dict(
        {'train_loss': train_losses, 'val_loss': valid_losses, 'val_r2': valid_r2, 'epoch_times': epoch_times})
    df_train_log.to_csv(os.path.join(output_path, 'train_log.csv'), index=False)