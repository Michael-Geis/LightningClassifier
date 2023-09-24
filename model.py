## IMPORTS
import os
import regex
import pandas as pd
import numpy as np
import cleantext

from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import lightning as L
from lightning.pytorch.loggers import CSVLogger

import torch
import torchmetrics
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
import config


class FineTuneHeadForMLC(L.LightningModule):
    def __init__(self, learning_rate):
        super().__init__()

        self.lr = learning_rate
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.MODEL_NAME_OR_PATH,
            problem_type="multi_label_classification",
            num_labels=config.NUM_LABELS,
        )
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.pre_classifier.parameters():
            param.requires_grad = True

        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def forward(self, tokenizer_output, labels):
        output = self.model(**tokenizer_output, labels=labels)
        return output

    def _shared_step(self, batch, batch_index):
        output = self(**batch)
        return output["loss"], output["logits"]

    def training_step(self, batch, batch_index):
        loss, logits = self._shared_step(batch, batch_index)
        return loss

    def validation_step(self, batch, batch_index):
        loss, logits = self._shared_step(batch, batch_index)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
