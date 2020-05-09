import torch
from torch import nn
from transformers import AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl
from model import SentimentModel
from dataloader import SentimentDataLoaders
from transformers import *
from functools import lru_cache
from argparse import Namespace

class TrainingModule(pl.LightningModule):
    def __init__(self, bert_model=None, tokenizer=None, hparams=None):
        super().__init__()
        self.model = SentimentModel(bert_model, tokenizer, output_size=2)
        self.loss = nn.CrossEntropyLoss()
        self.hparams = hparams
        self.dataloader = SentimentDataLoaders(hparams)

    def step(self, batch, step_name="train"):
        X, y = batch
        loss = self.loss(self.forward(X), y)
        loss_key = f"{step_name}_loss"
        tensorboard_logs = {loss_key: loss}
        print({("loss" if step_name == "train" else loss_key): loss, 'log': tensorboard_logs,
               "progress_bar": {loss_key: loss}})
        return {("loss" if step_name == "train" else loss_key): loss, 'log': tensorboard_logs,
                "progress_bar": {loss_key: loss}}

    def forward(self, X):
        return self.model(X)

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "valid")

    def validation_end(self, outputs):
        loss = torch.stack([x["valid_loss"] for x in outputs]).mean()
        return {"valid_loss": loss}

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def train_dataloader(self):
        return self.dataloader.train

    def val_dataloader(self):
        return self.dataloader.valid

    def test_dataloader(self):
        return self.dataloader.test

    @lru_cache()
    def total_steps(self):
        return len(self.dataloader.train) // self.hparams.accumulate_grad_batches * self.hparams.epochs

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.lr)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps(),
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]
