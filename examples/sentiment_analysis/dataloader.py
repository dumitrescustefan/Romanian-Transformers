import pandas as pd
from dataset import SentimentDataset
from imbalanced_sampler import ImbalancedDatasetSampler
from torch.utils.data import DataLoader

class SentimentDataLoaders:
    def __init__(self, hparams):
        df = pd.read_csv("./ro/train.csv")
        df.dropna(inplace=True)
        # TRAIN/VAL/TEST DATA
        self.train_df = df.iloc[:15000]
        self.validation_df = df.iloc[15000:]
        self.test_df = pd.read_csv("./ro/test.csv")
        self.test_df.dropna(inplace=True)
        self.hparams = hparams
        self.train, self.valid, self.test = self.generate_bert_dataloaders()

    def generate_bert_dataloaders(self):

        # train dataloader
        train_sampler = ImbalancedDatasetSampler(self.train_df)
        train_dataset = SentimentDataset(self.train_df)
        train_dataloader = DataLoader(
                                train_dataset,
                                batch_size=self.hparams.batch_size,
                                shuffle=False,
                                sampler=train_sampler
                                )

        # valid dataloader
        valid_dataset = SentimentDataset(self.validation_df)
        valid_dataloader = DataLoader(
                                valid_dataset,
                                batch_size=self.hparams.batch_size,
                                shuffle=False
                                )

        # test dataloader
        test_dataset = SentimentDataset(self.test_df)
        test_dataloader = DataLoader(
                            test_dataset,
                            batch_size=self.hparams.batch_size,
                            shuffle=False
                            )

        return train_dataloader, valid_dataloader, test_dataloader
