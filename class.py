import numpy as np
import torch
import pandas as pd
import sklearn
import random

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import onnxruntime as rt

batch_size    = 16
learning_rate = 0.003  ## 0.001
N_Epochs      = 4000

epsilon = 0.0001

# Put ilpd.csv in the same directory as this script.
from pathlib import Path
path_data = Path(__file__).resolve().parent / 'ilpd.csv'

expected_columns = ['Age', 'Gender', 'TB', 'DB', 'Alkphos',
                    'Sgpt', 'Sgot', 'TP', 'ALB', 'A/G Ratio', 'Selector']
liver_raw_data = pd.read_csv(path_data, delimiter=',', header=None)
if str(liver_raw_data.iloc[0, 0]).strip().lower() == 'age':
    liver_raw_data = liver_raw_data.iloc[1:].reset_index(drop=True)
liver_raw_data.columns = expected_columns

liver_raw_data['Gender'] = liver_raw_data['Gender'].astype(str).str.strip().map({'Male': 0, 'Female': 1})
for col in expected_columns:
    if col != 'Gender':
        liver_raw_data[col] = pd.to_numeric(liver_raw_data[col], errors='coerce')

# Original labels: 1 = liver patient, 2 = non-liver patient.
liver_raw_data['Selector'] = liver_raw_data['Selector'].map({1: 0, 2: 1})
print('Class counts (0=liver, 1=no liver):')
print(liver_raw_data['Selector'].value_counts(dropna=False))
print('Missing values:')
print(liver_raw_data.isna().sum())
assert liver_raw_data['Selector'].notna().all(), 'Invalid target labels in CSV'
assert liver_raw_data['Gender'].notna().all(), 'Unexpected gender values in CSV'

headers_list = liver_raw_data.columns.values.tolist()
cm = np.corrcoef(liver_raw_data.dropna()[headers_list].values.T)
plt.figure(figsize=(12, 8))
plt.imshow(cm, cmap='coolwarm', vmin=-1, vmax=1)
plt.xticks(range(len(headers_list)), headers_list, rotation=70)
plt.yticks(range(len(headers_list)), headers_list)
plt.colorbar(label='Correlation')
plt.tight_layout()
plt.show()

liver_raw_data_np = liver_raw_data.to_numpy()
print('Dataset shape:', liver_raw_data_np.shape)
X = liver_raw_data_np[:, :-1]
y = liver_raw_data_np[:, 10:11]
y = y.astype(int)
print('X:', X.shape, 'y:', y.shape)
the_set = np.unique(y)
print('Classes:', the_set)
_ = plt.hist(y, bins=[-0.5, 0.5, 1.5])
plt.title('Histogram for Indian Liver Patient Dataset')
plt.xticks([0, 1], ['Liver', 'No liver'])
plt.show()

random_seed = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_seed, stratify=y.ravel())
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)
training_medians = np.nanmedian(X_train, axis=0)
assert np.isfinite(training_medians).all(), 'One or more columns have no valid training data'
X_train = np.where(np.isnan(X_train), training_medians, X_train).astype(np.float32)
X_test = np.where(np.isnan(X_test), training_medians, X_test).astype(np.float32)
y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)

X_train_tr = torch.from_numpy(X_train)
X_test_tr = torch.from_numpy(X_test)
y_train_tr = torch.from_numpy(y_train)
y_test_tr = torch.from_numpy(y_test)

x_means = X_train_tr.mean(0, keepdim=True)
x_deviations = X_train_tr.std(0, keepdim=True) + epsilon
print('Means:', x_means)
print('Deviations:', x_deviations)

liver_train_list = [(X_train_tr[i], y_train_tr[i].item()) for i in range(X_train.shape[0])]
liver_test_list = [(X_test_tr[i], y_test_tr[i].item()) for i in range(X_test.shape[0])]
train_dl = torch.utils.data.DataLoader(liver_train_list, batch_size=batch_size, shuffle=True)
all_test_data = X_test.shape[0]
test_dl = torch.utils.data.DataLoader(liver_test_list, batch_size=all_test_data, shuffle=False)

## MLP
class MLP_Net(nn.Module):
    def __init__(self, x_means, x_deviations):
        super().__init__()
        self.register_buffer('x_means', x_means)
        self.register_buffer('x_deviations', x_deviations)
        self.linear1 = nn.Linear(10, 3)
        self.act1 = nn.Sigmoid()
        self.linear2 = nn.Linear(3, 2)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = (x - self.x_means) / self.x_deviations
        x = self.linear1(x)
        x = self.act1(x)
        # x = self.dropout(x)
        y_pred = self.linear2(x)  # logits
        return y_pred

## Deep Learning with 2 hidden layers
class DL_Net(nn.Module):
    def __init__(self, x_means, x_deviations):
        super().__init__()
        self.register_buffer('x_means', x_means)
        self.register_buffer('x_deviations', x_deviations)
        self.linear1 = nn.Linear(10, 15)
        self.act1 = nn.ReLU()
        self.linear2 = nn.Linear(15, 9)
        self.act2 = nn.ReLU()
        self.linear3 = nn.Linear(9, 2)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = (x - self.x_means) / self.x_deviations
        x = self.linear1(x)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.act2(x)
        x = self.dropout(x)
        y_pred = self.linear3(x)  # logits
        return y_pred

def training_loop(N_Epochs, model, loss_fn, opt):
    model.train()
    for epoch in range(N_Epochs):
        for xb, yb in train_dl:
            y_pred = model(xb)
            loss = loss_fn(y_pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 50 == 0:
            print(epoch, 'loss=', loss)

def print_metrics_function(y_test, y_pred):
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred, labels=[0, 1])
    print('Confusion Matrix:')
    print(confmat)
    print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred, average='weighted', zero_division=0))
    print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred, average='weighted', zero_division=0))
    print('F1-measure: %.3f' % f1_score(y_true=y_test, y_pred=y_pred, average='weighted', zero_division=0))

loss_fn = nn.CrossEntropyLoss()

# ---------------- MLP training and evaluation ----------------
print('\n========== TRAINING MLP ==========')
mlp_model = MLP_Net(x_means, x_deviations)
opt = torch.optim.Adam(mlp_model.parameters(), lr=learning_rate)
training_loop(N_Epochs, mlp_model, loss_fn, opt)
mlp_model.eval()
with torch.no_grad():
    mlp_logits = mlp_model(X_test_tr)
    mlp_preds = torch.argmax(mlp_logits, dim=1).numpy()
print('\n========== MLP TEST RESULTS ==========')
print_metrics_function(y_test.ravel(), mlp_preds)
mlp_f1 = f1_score(y_test.ravel(), mlp_preds, average='weighted', zero_division=0)

# ---------------- DNN training and evaluation ----------------
print('\n========== TRAINING DNN ==========')
dl_model = DL_Net(x_means, x_deviations)
opt = torch.optim.Adam(dl_model.parameters(), lr=learning_rate)
training_loop(N_Epochs, dl_model, loss_fn, opt)
dl_model.eval()
with torch.no_grad():
    dl_logits = dl_model(X_test_tr)
    dl_preds = torch.argmax(dl_logits, dim=1).numpy()
print('\n========== DNN TEST RESULTS ==========')
print_metrics_function(y_test.ravel(), dl_preds)
dl_f1 = f1_score(y_test.ravel(), dl_preds, average='weighted', zero_division=0)

print('\n========== MODEL COMPARISON ==========')
print(f'MLP weighted F1: {mlp_f1:.3f}')
print(f'DNN weighted F1: {dl_f1:.3f}')

# ---------------- Export best model to ONNX ----------------
import onnx
best_model = mlp_model if mlp_f1 >= dl_f1 else dl_model
best_model.eval()
# Save both trained PyTorch models, so exporting again does not require retraining.
torch.save(mlp_model.state_dict(), 'indian_liver_mlp.pth')
torch.save(dl_model.state_dict(), 'indian_liver_dnn.pth')

dummy_input = torch.zeros(1, 10, dtype=torch.float32)
torch.onnx.export(
    best_model, dummy_input, 'indian_liver_best.onnx',
    input_names=['input'], output_names=['logits'],
    opset_version=17,
    dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}},
    dynamo=False
)
onnx.checker.check_model(onnx.load('indian_liver_best.onnx'))
sess = rt.InferenceSession('indian_liver_best.onnx', providers=['CPUExecutionProvider'])
onnx_logits = sess.run(None, {'input': X_test[:3].astype(np.float32)})[0]
with torch.no_grad():
    torch_logits = best_model(torch.from_numpy(X_test[:3])).numpy()
np.testing.assert_allclose(onnx_logits, torch_logits, rtol=1e-4, atol=1e-4)
print('Exported and verified indian_liver_best.onnx')
print('Winning model:', type(best_model).__name__)
print('Output class: 0 = liver patient; 1 = no liver patient')
