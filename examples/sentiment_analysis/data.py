from torch.utils.data import DataLoader, Dataset

class SentimentDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.data = df
        
    def __getitem__(self, idx):
        return self.data.values[idx][1], self.data.values[idx][2]
    
    def __len__(self):
        return len(self.data)