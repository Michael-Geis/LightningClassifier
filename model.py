## IMPORTS
from transformers import AutoModelForSequenceClassification

import lightning as L
from lightning.pytorch.loggers import CSVLogger

import torch

from metrics import micro_metrics, macro_metrics
import config


class FineTuneHeadForMLC(L.LightningModule):
    def __init__(self, learning_rate):
        super().__init__()

        self.lr = learning_rate
        self.save_hyperparameters(ignore=["model"])

        ## Initialize train metrics
        self.micro_train_metrics = micro_metrics.clone(prefix="train_micro_")
        self.macro_train_metrics = macro_metrics.clone(prefix="train_macro_")

        ## Initialize val metrics
        self.micro_val_metrics = micro_metrics.clone(prefix="val_micro_")
        self.macro_val_metrics = macro_metrics.clone(prefix="val_macro_")

        ## Initialize model and freeze pre-trained layers
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
        features, labels = batch
        output = self.forward(tokenizer_output=features, labels=labels)
        return output["loss"], output["logits"]

    def training_step(self, batch, batch_index):
        loss, logits = self._shared_step(batch, batch_index)
        features, labels = batch

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        micro_mets = self.micro_train_metrics(logits, labels)
        self.log_dict(micro_mets, on_step=False, on_epoch=True, sync_dist=True)

        macro_mets = self.macro_train_metrics(logits, labels)
        self.log_dict(macro_mets, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_index):
        loss, logits = self._shared_step(batch, batch_index)
        features, labels = batch
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        self.micro_val_metrics.update(logits, labels)
        self.macro_val_metrics.update(logits, labels)

    def on_validation_epoch_end(self) -> None:
        micro_mets = self.micro_val_metrics.compute()
        self.log_dict(micro_mets, sync_dist=True)
        self.micro_val_metrics.reset()

        macro_mets = self.macro_val_metrics.compute()
        self.log_dict(macro_mets, sync_dist=True)
        self.macro_val_metrics.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
