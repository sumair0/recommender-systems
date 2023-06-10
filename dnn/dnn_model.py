
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dnn_dataset import TrainDataset

class DNN(pl.LightningModule):

    def __init__(self, num_users, num_items, ratings):
        
        super().__init__()
        self.ratings = ratings
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=64)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=64)
        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=16)
        self.fc4 = nn.Linear(in_features=16, out_features=8)
        self.fc5 = nn.Linear(in_features=8, out_features=4)
        self.fc6 = nn.Linear(in_features=4, out_features=2)
        self.output = nn.Linear(in_features=2, out_features=1)
        self.drop = nn.Dropout(0.01)
        self.max_rating = ratings['rating'].max()
        self.min_rating = ratings['rating'].min()
       

    def forward(self, user, item):
        
        item_embed = self.item_embedding(item)
        user_embed = self.user_embedding(user)
        x = torch.cat([user_embed, item_embed], dim=-1)

        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = nn.ReLU()(self.fc3(x))
        x = nn.ReLU()(self.fc4(x))
        x = nn.ReLU()(self.fc5(x))
        x = nn.ReLU()(self.fc6(x))
        
        x = self.drop(x)
        out = self.output(x)
        
        return out
    

    def training_step(self, batch, batch_idx):
        user, item, labels = batch
        preds = self(user, item)
        loss = nn.MSELoss()(preds, labels.view(-1, 1).float())
        
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return DataLoader(TrainDataset(self.ratings), batch_size=512)