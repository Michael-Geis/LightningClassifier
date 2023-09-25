## HF model
MODEL_NAME_OR_PATH = "distilbert-base-uncased"

## HF Tokenizer
TOKENIZER_NAME_OR_PATH = "distilbert-base-uncased"
TOKENIZER_MAX_LENGTH = 64
TOKENIZER_TRUNCATION = True
TOKENIZER_PADDING = "max_length"

## Data
DATA_DIR = "./data/"
NUM_LABELS = 18
INPUT_TEXT_COLUMN = "title"
LABEL_COLUMNS = [
    "math.AG",
    "math.AP",
    "math.CA",
    "math.CO",
    "math.DG",
    "math.DS",
    "math.FA",
    "math.GR",
    "math.GT",
    "math.IT",
    "math.MP",
    "math.NA",
    "math.NT",
    "math.OC",
    "math.PR",
    "math.QA",
    "math.RT",
    "math.ST",
]

## Data Preprocessing
DATA_PREPROCESS_BATCH_SIZE = 256

## DataLoaders
DATALOADER_BATCH_SIZE = 12
DATALOADER_NUM_WORKERS = 8

## Training
MAX_EPOCHS = 1
ACCELERATOR = "auto"
DEVICES = "auto"
DETERMINISTIC = True
LEARNING_RATE = 2e-5
