# 人员：汤振宇
# 开发时间: 2025/6/16 10:46
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import time
import sklearn.metrics as sm
import torch
import os
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def test_model(model, output_path, test_loader):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    test_labels = []
    test_preds = []
    with torch.no_grad():
        start_time = time.time()  # Record the start time
        for inputs, targets in tqdm(test_loader):
            inputs = inputs.to(torch.device(device))
            targets = targets.type(torch.float).to(torch.device(device))
            outputs = model(inputs)
            test_labels.append(targets)
            test_preds.append(outputs)
        end_time = time.time()  # Record the end time
        test_time = end_time - start_time
    test_labels = torch.cat(test_labels, dim=0).squeeze()
    test_preds = torch.cat(test_preds, dim=0).squeeze()
    r2 = sm.r2_score(test_labels.cpu().numpy(), test_preds.cpu().numpy())
    print('Test R2: {}, Test time: {}'.format(r2, test_time))
    df_test_result = pd.DataFrame.from_dict({'labels': test_labels.cpu().tolist(), 'predicts': test_preds.cpu().tolist()})
    df_test_result.to_csv(os.path.join(output_path, 'test_result.csv'), index=False)