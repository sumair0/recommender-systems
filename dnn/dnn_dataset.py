from torch.utils.data import Dataset
import torch 

class TrainDataset(Dataset):
    def __init__(self, ratings):
        self.users, self.items, self.labels = self.get_dataset(ratings)
        
    def __len__(self):
        return len(self.users)
    
    def __getitem__ (self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]
    
    def get_dataset(self, ratings):
        labels = ratings['rating'].tolist()
        items = ratings['movieId'].tolist()
        users = ratings['userId'].tolist()
        return torch.tensor(users), torch.tensor(items), torch.tensor(labels)