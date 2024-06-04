import os
import torch
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
import zipfile
import pytz

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class NeuNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        width = 2048
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(11, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 1)

        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

data = pd.read_csv('./data/training_set_imputed_RandomForest.csv')



class AirQualityDataset(Dataset):
    def __init__(self, data_param, data_pred):
        self.data_param = data_param
        self.data_pred = data_pred

    def __len__(self):
        return len(self.data_param)

    def __getitem__(self, idx):
        X = self.data_param[idx]
        y = self.data_pred[idx]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)



data_param = data.drop(columns=['PM2.5']).values
data_pred = data['PM2.5'].values


scaler = StandardScaler()
data_param = scaler.fit_transform(data_param)

#print('param:\n',data_param.values, '\n pred:\n', data_pred.values)
#print(torch.tensor(data_param, dtype = (torch.float64))[0])
# Split the data into training and test sets
train_indices, test_indices = train_test_split(range(len(data_param)), test_size=0.2, random_state=42)

# Create dataset and dataloader
dataset = AirQualityDataset(data_param, data_pred)

# Create subsets for training and testing
train_a = Subset(dataset, train_indices)
test_a = Subset(dataset, test_indices)


def get_train_data_loader(batch_size):
  return DataLoader(train_a, batch_size = batch_size)
def get_test_data_loader(batch_size):
  return DataLoader(test_a, batch_size = batch_size)

def plot_loss(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig('loss_plot.png')

# Train and test functions
def train_loop(dataloader, model, loss_fn, optimizer, train_losses):
    model.train()
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X).squeeze()
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch % 100 == 0:
            loss_value, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{len(dataloader.dataset):>5d}][{100*current/len(dataloader.dataset):.3f}%]")

    avg_train_loss = total_loss / len(dataloader)
    train_losses.append(avg_train_loss)

def test_loop(dataloader, model, loss_fn, val_losses):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    total_loss, total_abs_error = 0, 0
    all_dape = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X).squeeze()
            loss = loss_fn(pred, y)
            total_loss += loss.item()
            total_abs_error += (pred - y).abs().sum().item()
            dape = (pred - y).abs() / torch.abs(y)
            all_dape.extend(dape.tolist())

    avg_val_loss = total_loss / num_batches
    mean_absolute_error = total_abs_error / size
    mdape = torch.median(torch.tensor(all_dape)) * 100  # Convert to percentage

    val_losses.append(avg_val_loss)

    print(f"Test Error: \n MAE: {mean_absolute_error:>8f}, MDAPE: {mdape:>8f}%, Avg loss: {avg_val_loss:>8f} \n")
    return avg_val_loss

# Your model, loss function, and optimizer
model = NeuNet().to(device)
loss_fn = nn.L1Loss()
epochs = 15
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)

# Learning rate scheduler
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

same_training_session = False

if not same_training_session:
  train_losses = []
  val_losses = []
batch_sizes = [256]


same_training_session = True
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0

    def step(self, loss):
        if self.best_loss is None or loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

early_stopping = EarlyStopping(patience=15, min_delta=1e-4)

for batch_size in batch_sizes:
  train = get_train_data_loader(batch_size)
  test = get_test_data_loader(batch_size)
  for t in range(epochs):
      print(f"Epoch {t+1}, Batch Size {batch_size}" + '-' * 50)
      train_loop(train, model, loss_fn, optimizer, train_losses)
      print('Testing...')

      validation_loss =test_loop(test, model, loss_fn, val_losses)


      scheduler.step(validation_loss)

      if early_stopping.step(validation_loss):
        print("Early stopping")
        break
print("Done!")

plot_loss(train_losses, val_losses)

class AirQualitySubmitDataset(Dataset):
    def __init__(self, data_param):
        self.data_param = data_param

    def __len__(self):
        return len(self.data_param)

    def __getitem__(self, idx):
        X = self.data_param[idx]
        return torch.tensor(X, dtype=torch.float32)



# Load and preprocess the new dataset
new_data = pd.read_csv('./data/public_test.csv')
new_data_param = new_data.drop(columns=['StationId', 'Datetime']).values
scaler = StandardScaler()
new_data_param = scaler.fit_transform(new_data_param)

# Create a DataLoader for the new dataset
batch_size = 64
new_dataset = AirQualitySubmitDataset(new_data_param)  # Replace YourDataset with your dataset class
new_loader = DataLoader(new_dataset, batch_size=batch_size, shuffle=False)

# Make predictions using the model
model.eval()
predictions = []
with torch.no_grad():
    for X in new_loader:
        X = X.to(device)
        pred = model(X).squeeze()
        predictions.extend(pred.tolist())

# Save predictions to CSV
df = pd.DataFrame({'res': predictions})
print(df)
save_path = 'res.csv'
df.to_csv(save_path, index=False)

# Check if the file was saved successfully
if os.path.exists(save_path):
    print(f"File saved successfully at: {os.path.abspath(save_path)}")
else:
    print("Error: File was not saved.")

# Zip File
with zipfile.ZipFile('prediction.zip', 'w') as zipf:
    zipf.write('res.csv', 'res.csv')

# Get current time in UTC+7 (Hanoi time zone)
hanoi_tz = pytz.timezone('Asia/Ho_Chi_Minh')
current_time = datetime.now(hanoi_tz).strftime("%d-%m_%H-%M-%S")

# Create the final zip file with the timestamp in UTC+7
zip_filename = f'K_{current_time}.zip'
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    zipf.write('prediction.zip', 'prediction.zip')

print(f"Zip file created: {zip_filename}")
