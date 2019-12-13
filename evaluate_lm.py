import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers import DistilBertTokenizer
from transformers.optimization import AdamW
from transformers import DistilBertModel, DistilBertForMaskedLM
from mask_debias_dataset import MaskDebiasingDataset
import argparse 
from tqdm import tqdm
from helpers import get_input
import random

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='model to evaluate')
parser.add_argument('--evaluation_data', type=str, help='newline separated text', default='../realworldnlp/data/tatoeba/sentences.eng.200k.txt')
args = parser.parse_args()

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
if args.model:
    model.load_state_dict(torch.load(args.model))
model.eval()

eval_data = []
with open(args.evaluation_data) as f:
    for line in f:
        eval_data.append(line.strip())

total = 0.0
correct = 0.0

for data in tqdm(eval_data[:1000]):
    original = data
    orig_input = get_input(tokenizer, [original])
    mask_idx = random.randint(0, orig_input.shape[1]-1)
    masked_input = orig_input.clone()
    masked_input[0][mask_idx] = tokenizer.mask_token_id
    
    outputs = model(masked_input)
    predictions = outputs[0]

    mask_prediction = torch.argmax(predictions[0][mask_idx])
    actual_val = orig_input[0][mask_idx]

    if (mask_prediction.item() == actual_val.item()):
        correct += 1
    total += 1

accuracy = correct / total

print("ACC: {0}".format(accuracy))