import torch
import lightning as L
from dataset import ArXivDataModule
from model import FineTuneHeadForMLC
import config

MAX_EPOCHS = config.MAX_EPOCHS
ACCELERATOR = config.ACCELERATOR
DEVICES = config.DEVICES
DETERMINISTIC = config.DETERMINISTIC
LEARNING_RATE = config.LEARNING_RATE


def main(learning_rate=LEARNING_RATE):
    torch.manual_seed(93696)
    dm = ArXivDataModule(path_to_data_dir="./data/")
    model = FineTuneHeadForMLC(learning_rate=learning_rate)

    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=ACCELERATOR,
        devices=DEVICES,
        deterministic=DETERMINISTIC,
    )

    trainer.fit(model=model, datamodule=dm)


if __name__ == "__main__":
    main()
