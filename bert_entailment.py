import sys, os
nlp_path = os.path.abspath('../../')
if nlp_path not in sys.path:
    sys.path.insert(0, nlp_path)

import scrapbook as sb

from tempfile import TemporaryDirectory

import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

import torch

from utils_nlp.models.transformers.sequence_classification import Processor, SequenceClassifier
from utils_nlp.dataset.multinli import load_pandas_df
from utils_nlp.common.timer import Timer

QUICK_RUN = True
CACHE_DIR = './cache'

print(SequenceClassifier.list_supported_models())

### CONFIGURATION
MODEL_NAME = "distilbert-base-uncased"
TO_LOWER = False
BATCH_SIZE = 48

TRAIN_DATA_USED_FRACTION = 1
DEV_DATA_USED_FRACTION = 1
NUM_EPOCHS = 2
WARMUP_STEPS= 2500

if QUICK_RUN:
    TRAIN_DATA_USED_FRACTION = 1
    DEV_DATA_USED_FRACTION = 1
    NUM_EPOCHS = 5
    WARMUP_STEPS= 10

if not torch.cuda.is_available():
    BATCH_SIZE = BATCH_SIZE/2

RANDOM_SEED = 42

# model configurations
MAX_SEQ_LENGTH = 128

# optimizer configurations
LEARNING_RATE= 5e-5

# data configurations
TEXT_COL_1 = "sentence1"
TEXT_COL_2 = "sentence2"
LABEL_COL = "gold_label"
LABEL_COL_NUM = "gold_label_num"

### Load Data
train_df = load_pandas_df(local_cache_path=CACHE_DIR, file_split="train")
dev_df_matched = load_pandas_df(local_cache_path=CACHE_DIR, file_split="dev_matched")
dev_df_mismatched = load_pandas_df(local_cache_path=CACHE_DIR, file_split="dev_mismatched")

dev_df_matched = dev_df_matched.loc[dev_df_matched['gold_label'] != '-']
dev_df_mismatched = dev_df_mismatched.loc[dev_df_mismatched['gold_label'] != '-']

print("Training dataset size: {}".format(train_df.shape[0]))
print("Development (matched) dataset size: {}".format(dev_df_matched.shape[0]))
print("Development (mismatched) dataset size: {}".format(dev_df_mismatched.shape[0]))
print()
print(train_df[['gold_label', 'sentence1', 'sentence2']].head())

train_df = train_df.sample(frac=TRAIN_DATA_USED_FRACTION).reset_index(drop=True)
dev_df_matched = dev_df_matched.sample(frac=DEV_DATA_USED_FRACTION).reset_index(drop=True)
dev_df_mismatched = dev_df_mismatched.sample(frac=DEV_DATA_USED_FRACTION).reset_index(drop=True)

label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_df[LABEL_COL])
train_df[LABEL_COL_NUM] = train_labels 
num_labels = len(np.unique(train_labels))

### TOKENIZATION
processor = Processor(model_name=MODEL_NAME, cache_dir=CACHE_DIR, to_lower=TO_LOWER)
train_dataloader = processor.create_dataloader_from_df(
    df=train_df,
    text_col=TEXT_COL_1,
    label_col=LABEL_COL_NUM,
    shuffle=True,
    text2_col=TEXT_COL_2,
    max_len=MAX_SEQ_LENGTH,
    batch_size=BATCH_SIZE,
)
dev_dataloader_matched = processor.create_dataloader_from_df(
    df=dev_df_matched,
    text_col=TEXT_COL_1,
    shuffle=False,
    text2_col=TEXT_COL_2,
    max_len=MAX_SEQ_LENGTH,
    batch_size=BATCH_SIZE,
)
dev_dataloader_mismatched = processor.create_dataloader_from_df(
    df=dev_df_mismatched,
    text_col=TEXT_COL_1,
    shuffle=False,
    text2_col=TEXT_COL_2,
    max_len=MAX_SEQ_LENGTH,
    batch_size=BATCH_SIZE,
)


### TRAINING
classifier = SequenceClassifier(
    model_name=MODEL_NAME, num_labels=num_labels, cache_dir=CACHE_DIR
)
torch.save(classifier.model.state_dict(), "./pretrained_{0}.pth".format(round(time.time()))))

with Timer() as t:
    classifier.fit(
            train_dataloader,
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            warmup_steps=WARMUP_STEPS,
        )
print("Training time : {:.3f} hrs".format(t.interval / 3600))
torch.save(classifier.model.state_dict(), "./trained_{0}.pth".format(round(time.time()))))

### PREDICTION
with Timer() as t:
    predictions_matched = classifier.predict(dev_dataloader_matched)
print("Prediction time : {:.3f} hrs".format(t.interval / 3600))

with Timer() as t:
    predictions_mismatched = classifier.predict(dev_dataloader_mismatched)
print("Prediction time : {:.3f} hrs".format(t.interval / 3600))


### EVALUATION
predictions_matched = label_encoder.inverse_transform(predictions_matched)
print(classification_report(dev_df_matched[LABEL_COL], predictions_matched, digits=3))

predictions_mismatched = label_encoder.inverse_transform(predictions_mismatched)
print(classification_report(dev_df_mismatched[LABEL_COL], predictions_mismatched, digits=3))

result_matched_dict = classification_report(dev_df_matched[LABEL_COL], predictions_matched, digits=3, output_dict=True)
result_mismatched_dict = classification_report(dev_df_mismatched[LABEL_COL], predictions_mismatched, digits=3, output_dict=True)

# sb.glue("matched_precision", result_matched_dict["weighted avg"]["precision"])
# sb.glue("matched_recall", result_matched_dict["weighted avg"]["recall"])
# sb.glue("matched_f1", result_matched_dict["weighted avg"]["f1-score"])
# sb.glue("mismatched_precision", result_mismatched_dict["weighted avg"]["precision"])
# sb.glue("mismatched_recall", result_mismatched_dict["weighted avg"]["recall"])
# sb.glue("mismatched_f1", result_mismatched_dict["weighted avg"]["f1-score"])
