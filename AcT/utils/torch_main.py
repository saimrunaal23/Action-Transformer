import argparse
import os

import tensorflow as tf

import torch
import torch.nn.functional as F
from torch import nn
import torch.utils
from torchsummary import summary
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

from sklearn.model_selection import train_test_split

from utils.tools import CustomSchedule, CosineSchedule, read_yaml
from utils.tools import Logger
from utils.data import load_mpose, random_flip, random_noise, one_hot
from utils.transformer import TransformerEncoder, PatchClassEmbedding, Patches

parser = argparse.ArgumentParser(description='Process some input')
parser.add_argument('--config', default='config.yaml', type=str, help='Config path', required=False)
args = parser.parse_args()
config = read_yaml(args.config)

X_train, y_train, X_test, y_test = load_mpose(config['DATASET'], 1, verbose=False)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size=config['VAL_SIZE'],
                                                  random_state=config['SEEDS'][0],
                                                  stratify=y_train)
X_train = torch.Tensor(X_train)
y_train = torch.Tensor(y_train)
X_val = torch.Tensor(X_val)
y_val = torch.Tensor(y_val)

ds_train = torch.utils.data.TensorDataset(X_train, y_train)

ds_val = torch.utils.data.TensorDataset(X_val, y_val)

print(len(ds_train))

split = 1
fold = 0
trial = None
bin_path = config['MODEL_DIR']

model_size = config['MODEL_SIZE']
n_heads = config[model_size]['N_HEADS']
n_layers = config[model_size]['N_LAYERS']
embed_dim = config[model_size]['EMBED_DIM']
dropout = config[model_size]['DROPOUT']
mlp_head_size = config[model_size]['MLP']
d_model = 64 * n_heads
d_ff = d_model * 4
epochs = config['N_EPOCHS']
#learning_rate = CustomSchedule(d_model, warmup_steps=len(ds_train) * config['N_EPOCHS'] * config['WARMUP_PERC'], decay_step=len(ds_train) * config['N_EPOCHS'] * config['STEP_PERC'])
learning_rate = 0.1
print(learning_rate)

trainloader = torch.utils.data.DataLoader(ds_train, batch_size=config['BATCH_SIZE'],
                                          shuffle=True)

testloader = torch.utils.data.DataLoader(ds_val, batch_size=config['BATCH_SIZE'],
                                         shuffle=False)

print(trainloader)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(52, 64)
        self.pce = PatchClassEmbedding(d_model, config['FRAMES'])
        self.transformer = TransformerEncoder(d_model, n_heads, d_ff, dropout, nn.GELU(), n_layers)
        self.ll = LambdaLayer(lambda x: x[:, 0, :])
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 20)

    def forward(self, x):
        x = self.fc1(x)
        x = self.pce(x)
        x = self.transformer(x)
        x = self.ll(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


net = Net()

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

epoch_log = []
loss_log = []
accuracy_log = []
for epoch in range(epochs):
    print(f'Starting Epoch: {epoch + 1}...')
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            correct = 0
            total = 0

            with torch.no_grad():
                for data in testloader:
                    images, labels = data

                    outputs = net(images)

                    _, predicted = torch.max(outputs.data, 1)

                    total += labels.size(0)

                    correct += (predicted == labels).sum().item()

                accuracy = 100 * correct / total
                epoch_num = epoch + 1
                actual_loss = running_loss / 50
                print(
                    f'Epoch: {epoch_num}, Mini-Batches Completed: {(i + 1)}, Loss: {actual_loss:.3f}, Test Accuracy = {accuracy:.3f}%')
                running_loss = 0.0

    epoch_log.append(epoch_num)
    loss_log.append(actual_loss)
    accuracy_log.append(accuracy)

print('Finished Training')


