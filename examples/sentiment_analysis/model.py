import torch
from torch import nn
from utils import Mish
import pytorch_lightning as pl


class SentimentModel(pl.LightningModule):

    def __init__(self, model,
                 tokenizer,
                 output_size,
                 embedding_size=768,
                 dropout=0.1,
                 device=None
                 ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embedding_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, output_size)
        self.activation = Mish()
        self.device = device

    def forward(self, X):
        ids_sentences = [self.tokenizer.encode(text, add_special_tokens=True,
                                               pad_to_max_length=True,
                                               max_length=256) for text in X]
        input_tensor = torch.tensor(ids_sentences).to(self.device)

        with torch.no_grad():
            last_hidden_state, _, hidden_states = self.model(input_tensor)

        sent_emb = last_hidden_state[:, 0, :]

        output = self.activation(self.dropout(self.fc1(sent_emb)))
        output = self.dropout(self.fc2(output))

        return output