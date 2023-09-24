import torch
from lightning import Trainer
from modelclasses.mlclassifier import MLClassifier
from lightningmodules.lightningmlclassifier import LightningMLClassifier
from datamodules.arxiv_datamodule import ArXivDataModule

torch.manual_seed(93696)
dm = ArXivDataModule(data_dir="./data/HF-bert-base-uncased-splits")

pytorch_clf = MLClassifier(num_features=768, num_labels=18)
lightning_model = LightningMLClassifier(model=pytorch_clf, learning_rate=0.05)

trainer = Trainer(
    max_epochs=10,
    accelerator="cpu",
    devices="auto",
    deterministic=True,
)

trainer.fit(model=lightning_model, datamodule=dm)

train_acc = trainer.validate(dataloaders=dm.train_dataloader())[0]["val_acc"]
dev_acc = trainer.validate(datamodule=dm)[0]["val_acc"]
test_acc = trainer.validate(datamodule=dm)[0]["test_acc"]

print(
    f"Train Accuracy: {train_acc*100:.2f}%",
    f"Dev Accuracy: {dev_acc*100:.2f}%",
    f"Test Accuracy: {test_acc*100:.2f}%",
)
