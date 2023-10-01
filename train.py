import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.tuner import Tuner
from dataset import ArXivDataModule
from model import FineTuneHeadForMLC
import config
import os

## Config variables
MAX_EPOCHS = config.MAX_EPOCHS
ACCELERATOR = config.ACCELERATOR
DEVICES = config.DEVICES
DETERMINISTIC = config.DETERMINISTIC
LEARNING_RATE = config.LEARNING_RATE

CHECKPOINT_PATH = config.CHECKPOINT_PATH
DATA_DIR = config.DATA_DIR


def main(learning_rate=LEARNING_RATE):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.manual_seed(93696)

    ## Define callbacks
    # The path of the best model is stored as trainer.checkpoint_callback.best_model_path
    min_val_loss_ckpt = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
        save_top_k=1,
        mode="min",
        monitor="val_loss",
        save_last=True,  # The params at the end of training can be loaded with ckpt_path="last"
    )

    ## Define Trainer
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=ACCELERATOR,
        devices=DEVICES,
        deterministic=DETERMINISTIC,
        callbacks=[min_val_loss_ckpt],
    )

    ## Set up Data Module and Load model
    dm = ArXivDataModule(data_dir=DATA_DIR)

    if learning_rate == "auto":
        model = FineTuneHeadForMLC(learning_rate=1.0)
        ## Use automatic learning rate finder to find a candidate lr
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(model=model, datamodule=dm)
        model.lr = lr_finder.suggestion()
    else:
        model = FineTuneHeadForMLC(learning_rate=learning_rate)

    trainer.fit(model=model, datamodule=dm)


if __name__ == "__main__":
    main()
