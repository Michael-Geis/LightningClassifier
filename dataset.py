import os
import os
import regex
import cleantext
import numpy as np
import pandas as pd

from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

import lightning as L
import config

## CONFIG VARIABLES
DATA_DIR = config.DATA_DIR

#### TODO ----- PUT THE TOKENIZATION OF THE DATASET BACK INTO PREPARE DATA? DOES THAT
####            NEED TO BE CALLED ON EVERY GPU?


class ArXivDataset(TorchDataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.feature_columns = [
            col for col in self.dataset.column_names if not col == "labels"
        ]
        self.labels = dataset["labels"].float()

        for col in self.feature_columns:
            setattr(self, col, dataset[col])

    def __getitem__(self, index):
        labels = self.labels[index]
        features = {}
        for col in self.feature_columns:
            features[col] = getattr(self, col)[index]
        return features, labels

    def __len__(self):
        return len(self.dataset)


class ArXivDataModule(L.LightningDataModule):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = config.DATALOADER_BATCH_SIZE
        self.num_workers = config.DATALOADER_NUM_WORKERS

        ## Pre-process and tokenize dataset to create dataloaders if it does not exist on disk.
        self.path_to_tokenized_dataset_dict = os.path.join(
            self.data_dir, "tokenized_dataset/"
        )

        if not os.path.exists(self.path_to_tokenized_dataset_dict):
            self.source_dataset_dict = self._dataset_dict_from_file(data_dir)
            self.preprocessed_dataset_dict = self._preprocess_dataset_dict(
                self.source_dataset_dict
            )
            self.preprocessed_dataset_dict.set_format("torch")
            self.preprocessed_dataset_dict.save_to_disk(
                self.path_to_tokenized_dataset_dict
            )

        ## Otherwise load the pre-processed DatasetDict from disk.
        else:
            self.preprocessed_dataset_dict = DatasetDict.load_from_disk(
                self.path_to_tokenized_dataset_dict
            )

    def setup(self, stage=None):
        self.train_dataset = ArXivDataset(self.preprocessed_dataset_dict["train"])
        self.val_dataset = ArXivDataset(self.preprocessed_dataset_dict["val"])
        self.test_dataset = ArXivDataset(self.preprocessed_dataset_dict["test"])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    ## Data preparation helper functions

    def _dataset_dict_from_file(self, data_dir):
        """Returns HF DatasetDict instance with train/val/test splits. Each dataset
        contains a field called 'title' containing the title of the article and one field for
        each of the possible labels e.g. 'math.CO'. Each subject field has a value of either 0
        or 1 which denotes if the label applies.

        Args:
            path_to_data_dir: path to directory containing the train, val, test data. Each of these

            datatype: File type the data is stored in, e.g. "csv" or "parquet"

        Returns:
            DatasetDict object containing a column for the input text to be classifier, and columns for the labels. The column name containing
            the text can be configured in the config file.
        """
        split_dict = {}
        for split in ["train", "val", "test"]:
            source_data_df = pd.read_parquet(os.path.join(data_dir, f"{split}.parquet"))

            label_columns = config.LABEL_COLUMNS
            input_text_column = config.INPUT_TEXT_COLUMN

            data_df_columns = ["input_text"].extend(label_columns)
            data_df = pd.DataFrame(columns=data_df_columns)

            data_df["input_text"] = source_data_df[input_text_column]
            for label in label_columns:
                data_df[label] = source_data_df[label]

            split_dataset = Dataset.from_pandas(data_df)
            split_dict[split] = split_dataset
        return DatasetDict(split_dict)

    def _preprocess_dataset_dict(self, dataset_dict):
        def _vectorize_labels(dataset_dict):
            dsd_vectorized_labels = dataset_dict.map(
                self._batch_vectorize_labels,
                batched=True,
                batch_size=config.DATA_PREPROCESS_BATCH_SIZE,
            )
            dataset_dict_columns = dataset_dict["train"].column_names
            dsd_vectorized_labels = dsd_vectorized_labels.remove_columns(
                [
                    col
                    for col in dataset_dict_columns
                    if not col in ["input_text", "labels"]
                ]
            )
            return dsd_vectorized_labels

        def _sanitize_input_text(dataset_dict):
            dsd_sanitized_input_text = dataset_dict.map(
                self._batch_clean_input_text,
                batched=True,
                batch_size=config.DATA_PREPROCESS_BATCH_SIZE,
            )
            return dsd_sanitized_input_text

        def _tokenize_dataset_dict(dataset_dict):
            dsd_tokenized = dataset_dict.map(
                self._batch_tokenize,
                batched=True,
                batch_size=config.DATA_PREPROCESS_BATCH_SIZE,
            )
            dsd_tokenized = dsd_tokenized.remove_columns(["input_text"])
            return dsd_tokenized

        dataset_dict = _vectorize_labels(dataset_dict)
        dataset_dict = _sanitize_input_text(dataset_dict)
        dataset_dict = _tokenize_dataset_dict(dataset_dict)

        return dataset_dict

    def _batch_tokenize(self, batch):
        tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME_OR_PATH)

        input_text = batch["input_text"]
        encoded_batch = tokenizer(
            input_text,
            max_length=config.TOKENIZER_MAX_LENGTH,
            truncation=config.TOKENIZER_TRUNCATION,
            padding=config.TOKENIZER_PADDING,
        )

        return encoded_batch

    def _batch_vectorize_labels(self, batch):
        labels = config.LABEL_COLUMNS
        num_labels = config.NUM_LABELS

        batch_size = len(batch["input_text"])
        label_array = np.zeros((batch_size, num_labels))
        for k, label in enumerate(labels):
            label_array[:, k] = batch[label]

        return {"labels": label_array}

    def _batch_clean_input_text(self, batch):
        cleaned_input_text = [
            self._clean_input_text(text) for text in batch["input_text"]
        ]
        return {"input_text": cleaned_input_text}

    def _clean_input_text(self, input_text):
        #### STRING CLEANING UTILITIES

        ## 1. Latin-ize latex accents enclosed in brackets
        def remove_latex_accents(string):
            accent = r"\\[\'\"\^\`H\~ckl=bdruvtoi]\{([a-z])\}"
            replacement = r"\1"

            string = regex.sub(accent, replacement, string)
            return string

        ## 2. Remove latex environments
        def remove_env(string):
            env = r"\\[a-z]{2,}{[^{}]+?}"

            string = regex.sub(env, "", string)
            return string

        ## 3. Latin-ize non-{} enclosed latex accents:
        def remove_accents(string):
            accent = r"\\[\'\"\^\`H\~ckl=bdruvtoi]([a-z])"
            replacement = r"\1"

            string = regex.sub(accent, replacement, string)
            return string

        ## 4. ONLY remove latex'd math that is separated as a 'word' i.e. has space characters on either side of it.

        def remove_latex(string):
            latex = r"\s(\$\$?)[^\$]*?\1\S*"
            string = regex.sub(latex, " LATEX ", string)
            return string

        input_text = input_text.replace("\n", " ")
        input_text = remove_latex_accents(input_text)
        input_text = remove_env(input_text)
        input_text = remove_accents(input_text)
        input_text = remove_latex(input_text)
        input_text = cleantext.clean(
            input_text,
            fix_unicode=True,
            lower=True,
            to_ascii=True,
            normalize_whitespace=True,
            no_currency_symbols=True,
        )
        return input_text
