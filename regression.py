
from pathlib import Path
from io import BytesIO
from urllib.request import urlopen

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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Professor's original parameters
batch_size    = 16
learning_rate = 0.005  ## 0.001
N_Epochs      = 100
epsilon       = 0.0001

# ---- READ DATA: changed from wine to building energy ----
path_data = Path(__file__).resolve().parent / 'ENB2012_data.xlsx'
if path_data.exists():
    ENERGY_raw_data = pd.read_excel(path_data, engine='openpyxl')
else:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx'
    print('ENB2012_data.xlsx not found. Downloading from the UCI repository...')
    try:
        with urlopen(url, timeout=25) as response:
            raw_excel = response.read()
        path_data.write_bytes(raw_excel)
        ENERGY_raw_data = pd.read_excel(BytesIO(raw_excel), engine='openpyxl')
    except Exception as exc:
        raise FileNotFoundError(
            'Download ENB2012_data.xlsx from the UCI Energy Efficiency page '
            'and put it in the same folder as regression.py. '
            'Official page: https://archive.ics.uci.edu/dataset/242/energy+efficiency'
        ) from exc

ENERGY_raw_data.columns = ENERGY_raw_data.columns.astype(str).str.strip()
required = [f'X{i}' for i in range(1, 9)] + ['Y1']
if not set(required).issubset(ENERGY_raw_data.columns):
    raise ValueError(f'Expected columns {required}; found {ENERGY_raw_data.columns.tolist()}')

# The original workbook includes Y2 (Cooling Load); this assignment predicts Y1 only.
ENERGY_raw_data = ENERGY_raw_data[required].copy()
ENERGY_raw_data = ENERGY_raw_data.apply(pd.to_numeric, errors='coerce')
ENERGY_raw_data = ENERGY_raw_data.dropna(subset=['Y1']).reset_index(drop=True)
print('Dataset shape:', ENERGY_raw_data.shape)
print('Missing values:\n', ENERGY_raw_data.isna().sum())
print('Features:', required[:-1])
print('Regression target: Y1 = Heating Load')

# ---- DATA ANALYSIS: same correlation-matrix concept ----
headers_list = ENERGY_raw_data.columns.values.tolist()
cm = np.corrcoef(ENERGY_raw_data.dropna()[headers_list].values.T)
plt.figure(figsize=(10, 7))
plt.imshow(cm, cmap='coolwarm', vmin=-1, vmax=1)
plt.xticks(range(len(headers_list)), headers_list)
plt.yticks(range(len(headers_list)), headers_list)
plt.colorbar(label='Correlation')
plt.title('Energy Efficiency Dataset — Correlation Matrix')
plt.tight_layout()
plt.savefig('energy_correlation_matrix.png', dpi=160)
plt.close()

# ---- PROCESS DATA: professor's NumPy / PyTorch pipeline ----
ENERGY_raw_data_np = ENERGY_raw_data.to_numpy()
X = ENERGY_raw_data_np[:, :-1]
y = ENERGY_raw_data_np[:, 8:9]
print('X shape:', X.shape)
print('y shape:', y.shape)

# Added validation split: compare models on validation, reserve test for final R2.
random_seed = 42
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_seed
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=random_seed
)

# Training-only median imputation (normally no missing values in this UCI dataset).
training_medians = np.nanmedian(X_train.astype(np.float64), axis=0)
if not np.isfinite(training_medians).all():
    raise ValueError('An input feature has no valid training values.')
X_train = np.where(np.isnan(X_train.astype(float)), training_medians, X_train)
X_val = np.where(np.isnan(X_val.astype(float)), training_medians, X_val)
X_test = np.where(np.isnan(X_test.astype(float)), training_medians, X_test)

X_train = X_train.astype(np.float32)
X_val   = X_val.astype(np.float32)
X_test  = X_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_val   = y_val.astype(np.float32)
y_test  = y_test.astype(np.float32)
print('Train / validation / test:', X_train.shape, X_val.shape, X_test.shape)

X_train_tr = torch.from_numpy(X_train)
X_val_tr   = torch.from_numpy(X_val)
X_test_tr  = torch.from_numpy(X_test)
y_train_tr = torch.from_numpy(y_train)
y_val_tr   = torch.from_numpy(y_val)
y_test_tr  = torch.from_numpy(y_test)

# Professor's training-set standardization
x_means      = X_train_tr.mean(0, keepdim=True)
x_deviations = X_train_tr.std(0, keepdim=True) + epsilon
print('Means:', x_means)
print('Deviations:', x_deviations)

train_ds = TensorDataset(X_train_tr, y_train_tr)
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

# ---- PROFESSOR'S NEURAL NETWORK ARCHITECTURES ----
# Necessary dimension change only: 11 wine features -> 8 energy features.
# register_buffer replaces plain tensor attributes so ONNX includes normalization.
class LinRegNet(nn.Module):
    def __init__(self, x_means, x_deviations):
        super().__init__()
        self.register_buffer('x_means', x_means)
        self.register_buffer('x_deviations', x_deviations)
        self.linear1 = nn.Linear(8, 1)

    def forward(self, x):
        x = (x - self.x_means) / self.x_deviations
        y_pred = self.linear1(x)
        return y_pred

class MLP_Net(nn.Module):
    def __init__(self, x_means, x_deviations):
        super().__init__()
        self.register_buffer('x_means', x_means)
        self.register_buffer('x_deviations', x_deviations)
        self.linear1 = nn.Linear(8, 8)
        self.act1 = nn.Sigmoid()
        self.linear2 = nn.Linear(8, 1)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = (x - self.x_means) / self.x_deviations
        x = self.linear1(x)
        x = self.act1(x)
        x = self.dropout(x)
        y_pred = self.linear2(x)
        return y_pred

class DL_Net(nn.Module):
    def __init__(self, x_means, x_deviations):
        super().__init__()
        self.register_buffer('x_means', x_means)
        self.register_buffer('x_deviations', x_deviations)
        self.linear1 = nn.Linear(8, 10)
        self.act1 = nn.ReLU()
        self.linear2 = nn.Linear(10, 6)
        self.act2 = nn.ReLU()
        self.linear3 = nn.Linear(6, 1)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = (x - self.x_means) / self.x_deviations
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)
        # x = self.dropout(x)
        y_pred = self.linear3(x)
        return y_pred

class LinearPlusNonLinear_Net(nn.Module):
    def __init__(self, x_means, x_deviations):
        super().__init__()
        self.register_buffer('x_means', x_means)
        self.register_buffer('x_deviations', x_deviations)
        self.f1_linear1 = nn.Linear(8, 1)
        self.f2_linear1 = nn.Linear(8, 14)
        self.f2_act1 = nn.Sigmoid()
        self.f2_linear2 = nn.Linear(14, 1)

    def forward(self, x):
        x = (x - self.x_means) / self.x_deviations
        f1 = self.f1_linear1(x)  # corrected professor's typo: f1_linear -> f1_linear1
        f2 = self.f2_linear1(x)
        f2 = self.f2_act1(f2)
        f2 = self.f2_linear2(f2)
        y_pred = f1 + f2
        return y_pred

# ---- PROFESSOR'S TRAINING LOOP: preserved ----
def training_loop(N_Epochs, model, loss_fn, opt):
    for epoch in range(N_Epochs):
        for xb, yb in train_dl:
            y_pred = model(xb)
            loss = loss_fn(y_pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 20 == 0:
            print(epoch, 'loss=', loss)

# ---- ADDED: train each of professor's four PyTorch architectures ----
model_classes = [LinRegNet, MLP_Net, DL_Net, LinearPlusNonLinear_Net]
results = []
models = {}
loss_fn = F.mse_loss

for model_class in model_classes:
    print('\n========== TRAINING', model_class.__name__, '==========')
    model = model_class(x_means, x_deviations)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    training_loop(N_Epochs, model, loss_fn, opt)
    model.eval()  # disables professor's MLP dropout for evaluation
    with torch.no_grad():
        y_pred_val = model(X_val_tr).numpy()
    validation_r2 = r2_score(y_val, y_pred_val)
    print(f'{model_class.__name__} validation R2: {validation_r2:.4f}')
    models[model_class.__name__] = model
    results.append({'Model': model_class.__name__, 'Validation R2': validation_r2})

results_df = pd.DataFrame(results).sort_values('Validation R2', ascending=False)
print('\n========== PYTORCH MODEL COMPARISON ==========' )
print(results_df.to_string(index=False))
results_df.to_csv('energy_model_comparison.csv', index=False)

# Choose best model using validation, not test results
winning_name = str(results_df.iloc[0]['Model'])
best_model = models[winning_name]
best_model.eval()

with torch.no_grad():
    y_pred_test = best_model(X_test_tr).numpy()
print('\n========== FINAL TEST RESULTS ==========' )
print('Winning model:', winning_name)
print('Testing R**2:', r2_score(y_test, y_pred_test))
print('Testing MAE:', mean_absolute_error(y_test, y_pred_test))
print('Testing RMSE:', np.sqrt(mean_squared_error(y_test, y_pred_test)))
print('First 10 predicted vs actual heating loads:')
for predicted, actual in zip(y_pred_test[:10, 0], y_test[:10, 0]):
    print(f'pred={predicted:.2f}, real={actual:.2f}')

torch.save(best_model.state_dict(), 'energy_best_model.pth')

plt.figure(figsize=(7, 5))
plt.scatter(y_test.ravel(), y_pred_test.ravel(), alpha=.65)
lo = min(float(y_test.min()), float(y_pred_test.min()))
hi = max(float(y_test.max()), float(y_pred_test.max()))
plt.plot([lo, hi], [lo, hi], 'k--', label='Perfect prediction')
plt.xlabel('Actual Heating Load')
plt.ylabel('Predicted Heating Load')
plt.title(f'{winning_name}: Actual vs Predicted (Test R² = {r2_score(y_test, y_pred_test):.3f})')
plt.legend()
plt.tight_layout()
plt.savefig('energy_actual_vs_predicted.png', dpi=160)
plt.close()

# ---- ADDED: ONNX export and prediction equivalence check ----
# These packages are needed for ONNX deployment, not to train the models.
try:
    import onnx
    import onnxruntime as rt
    dummy_input = torch.zeros(1, 8, dtype=torch.float32)
    torch.onnx.export(
        best_model, dummy_input, 'energy_best_model.onnx',
        input_names=['input'], output_names=['heating_load'],
        opset_version=17,
        dynamic_axes={'input': {0: 'batch'}, 'heating_load': {0: 'batch'}},
        dynamo=False
    )
    onnx.checker.check_model(onnx.load('energy_best_model.onnx'))
    sess = rt.InferenceSession('energy_best_model.onnx', providers=['CPUExecutionProvider'])
    onnx_pred = sess.run(None, {'input': X_test[:3]})[0]
    with torch.no_grad():
        torch_pred = best_model(torch.from_numpy(X_test[:3])).numpy()
    np.testing.assert_allclose(onnx_pred, torch_pred, rtol=1e-4, atol=1e-4)
    print('Exported and verified energy_best_model.onnx')
except ImportError as exc:
    print('ONNX export unavailable. Install: python -m pip install onnx onnxruntime onnxscript')
    print('Missing dependency:', exc)
