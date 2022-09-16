import argparse
import os
import torch
import torchmetrics
import torch.nn.functional as F
from torch import nn
import torch.utils

from utils.tools import read_yaml
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split

from utils import data
from utils.tools import CustomSchedule, CosineSchedule
from utils.data import load_mpose, random_flip, random_noise, one_hot
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
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


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class LitModel(pl.LightningModule):

    def __init__(self, config):
        self.config = config
        self.split = 1
        self.fold = 0
        self.trial = None
        self.bin_path = self.config['MODEL_DIR']

        self.model_size = self.config['MODEL_SIZE']
        self.n_heads = self.config[self.model_size]['N_HEADS']
        self.n_layers = self.config[self.model_size]['N_LAYERS']
        self.embed_dim = self.config[self.model_size]['EMBED_DIM']
        self.dropout = self.config[self.model_size]['DROPOUT']
        self.mlp_head_size = self.config[self.model_size]['MLP']
        self.d_model = 64 * self.n_heads
        self.d_ff = self.d_model * 4
        self.learning_rate = CustomSchedule(self.d_model,
                                            warmup_steps=len(ds_train) * self.config['N_EPOCHS'] * self.config[
                                                'WARMUP_PERC'],
                                            decay_step=len(ds_train) * self.config['N_EPOCHS'] * self.config[
                                                'STEP_PERC'])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.ds_train, batch_size=self.config['BATCH_SIZE'], shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.ds_val, batch_size=self.config['BATCH_SIZE'], shuffle=False)

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, batch, batch_idx):
        data, label = batch
        output = self.forward(data)
        loss = nn.CrossEntropyLoss()(output, label)
        self.log('train_loss', loss)
        self.log('train_acc_step', self.accuracy(output, label))
        return {'loss': loss, 'log': self.log}

    def training_epoch_end(self, outs):
        self.log('train_acc_epoch', self.accuracy.compute())

    def validation_step(self, batch, batch_idx):
        val_data, val_label = batch
        val_output = self.forward(val_data)
        val_loss = nn.CrossEntropyLoss()(val_output, val_label)
        self.log('val_acc_step', self.accuracy(val_output, val_label))
        self.log('val_loss', val_loss)

    def validation_epoch_end(self, outs):
        self.log('val_acc_epoch', self.accuracy.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, x):
        transformer = nn.Transformer(self.d_model, self.n_heads, self.n_layers,self.n_layers, self.d_ff, self.dropout, nn.GELU())
      #  inputs = tf.keras.layers.Input(shape=(self.config['FRAMES'],
       #                                       self.config[self.config['DATASET']]['KEYPOINTS'] * self.config[
       #                                           'CHANNELS']))
        x = nn.Linear(self.d_model)(x)
        x = PatchClassEmbedding(self.d_model, self.config['FRAMES'])(x)
        x = transformer(x)
        x = LambdaLayer(lambda x: x[:, 0, :])(x)
        x = nn.Linear(self.mlp_head_size)(x)
        outputs = nn.Linear(self.config['CLASSES'])(x)

        return outputs


model = LitModel(config)

trainer = pl.Trainer(accelerator='cpu', max_epochs=2)

trainer.fit(model)
