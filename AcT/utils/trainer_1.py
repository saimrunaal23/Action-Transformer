import os
import torch
import torchmetrics
import torch.nn.functional as F

from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from utils.transformer import TransformerEncoder, PatchClassEmbedding, Patches


from PIL import Image

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

class LitModel(pl.LightningModule):

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
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
        self.activation = tf.nn.gelu
        self.d_model = 64 * self.n_heads
        self.d_ff = self.d_model * 4
        self.learning_rate = CustomSchedule(self.d_model, 
                                warmup_steps=len(self.ds_train)*self.config['N_EPOCHS']*self.config['WARMUP_PERC'],
                                decay_step=len(self.ds_train)*self.config['N_EPOCHS']*self.config['STEP_PERC'])


    def get_data(self):
        X_train, y_train, X_test, y_test = load_mpose(self.config['DATASET'], self.split, verbose=False)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                          test_size=self.config['VAL_SIZE'],
                                                          random_state=self.config['SEEDS'][self.fold],
                                                          stratify=y_train)
                
        ds_train =  data.TensorDataset((X_train, y_train))
        ds_train = ds_train.map(lambda x,y : one_hot(x,y,self.config['CLASSES']))
        ds_train = ds_train.map(random_flip)
        ds_train = ds_train.map(random_noise)
        self.ds_train = ds_train.shuffle(X_train.shape[0])

        ds_val = data.TensorDataset((X_val, y_val))
        self.ds_val = ds_val.map(lambda x,y : one_hot(x,y,self.config['CLASSES']))


        ds_test = data.TensorDataset((X_test, y_test))
        self.ds_test = ds_test.map(lambda x,y : one_hot(x,y,self.config['CLASSES']))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset = self.ds_train, batch_size = self.config['BATCH_SIZE'], shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset = self.ds_val, batch_size = self.config['BATCH_SIZE'], shuffle=False)

    def cross_entropy_loss(self, logits, labels):
     return F.nll_loss(logits, labels)

    def training_step(self, batch, batch_idx):
        data, label = batch
        output = self.forward(data)
        loss = nn.CrossEntropyLoss()(output,label)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=(self.learning_rate))
        return optimizer

    def forward(self, x):
       transformer = TransformerEncoder(self.d_model, self.n_heads, self.d_ff, self.dropout, self.activation, self.n_layers)
       inputs = tf.keras.layers.Input(shape=(self.config['FRAMES'], 
                                              self.config[self.config['DATASET']]['KEYPOINTS']*self.config['CHANNELS']))
        x = nn.Linear(self.d_model)(inputs)
        x = PatchClassEmbedding(self.d_model, self.config['FRAMES'])(x)
        x = transformer(x)
        x = LambdaLayer(lambda x: x[:,0,:])(x)
        x = nn.Linear(self.mlp_head_size)(x)
        outputs = nn.Linear(self.config['CLASSES'])(x)

        return outputs


model = LitModel(batch_size = 32, learning_rate=0.001)

trainer = pl.Trainer(gpus=1, self.config['N_EPOCHS'])

trainer.fit(model)