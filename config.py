## HF model
MODEL_NAME_OR_PATH = "distilbert-base-uncased"

## HF Tokenizer
TOKENIZER_NAME_OR_PATH = "distilbert-base-uncased"
TOKENIZER_MAX_LENGTH = 64
TOKENIZER_TRUNCATION = True
TOKENIZER_PADDING = "max_length"

## Data
NUM_LABELS = 18
INPUT_TEXT_COLUMN = "title"
LABEL_COLUMNS = []

## Data Preprocessing
DATA_PREPROCESS_BATCH_SIZE = 256

## DataLoaders
DATALOADER_BATCH_SIZE = 12
DATALOADER_NUM_WORKERS = 2
