import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers import DistilBertTokenizer
from transformers.optimization import AdamW
from transformers import DistilBertModel, DistilBertForMaskedLM
from mask_debias_dataset import MaskDebiasingDataset
from torch.utils.data import DataLoader, Dataset, random_split
import argparse 
from tqdm import tqdm
from helpers import get_input

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='model to evaluate')

args = parser.parse_args()

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
dataset = MaskDebiasingDataset('masked_evaluation_data.jsonl', 'definitional_pairs.txt', tokenizer.mask_token)
model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
if args.model:
    model.load_state_dict(torch.load(args.model))
model.eval()

train_data = DataLoader(dataset, batch_size=1)

def_score = torch.tensor([0.0])
bias_score = torch.tensor([0.0])

for data in tqdm(train_data):
    tup1, tup2, original = data 
    def_masked, def_masked_words, def_idx = tup1
    bias_masked, bias_pair_masked, bias_masked_token, bias_idx = tup2 

    def_input = get_input(tokenizer, def_masked)
    bias_input = get_input(tokenizer, bias_masked)
    bias_pair_input = get_input(tokenizer, bias_pair_masked)
    original_input = get_input(tokenizer, original)

    def_mask_indices = torch.where(def_input == tokenizer.mask_token_id)
    bias_mask_indices = torch.where(bias_input == tokenizer.mask_token_id)
    bias_pair_mask_indices = torch.where(bias_pair_input == tokenizer.mask_token_id)

    # assume the beginning is always batch index, and assume batch size is always 1
    def_mask_indices = def_mask_indices[1]
    def_masked_words = def_masked_words[0]

    # Get the definitional word pair for this example
    word_pairs = [(w, dataset.get_pair(w)) for w in def_masked_words] # this assumes batch_size of 1
    word_pair_indices = [tokenizer.convert_tokens_to_ids(w) for w in word_pairs] 

    # Get the predictions when masking definitional words, and compute loss:
    outputs = model(def_input)
    predictions = outputs[0]

    for def_mask_idx in def_mask_indices:
        pair_scores = predictions[0][def_mask_idx][word_pair_indices]
        # Now add this to loss
        def_score += torch.flatten(torch.abs(pair_scores[0] - pair_scores[1])) / abs(torch.max(pair_scores, dim=0)[0])
    
    # Get the predictions when masking potentially biased words, and compute loss:
    biased_word_id = tokenizer.convert_tokens_to_ids(bias_masked_token)

    outputs = model(bias_input)
    predictions = outputs[0]
    # assume the beginning is always batch index, and assume batch size is always 1
    bias_mask_index = bias_mask_indices[1][0]
    true_score = predictions[0][bias_mask_index][biased_word_id]

    bias_pair_mask_index = bias_pair_mask_indices[1][0]
    outputs = model(bias_pair_input)
    predictions = outputs[0]
    pair_score = predictions[0][bias_pair_mask_index][biased_word_id]
    bias_score += torch.abs(true_score - pair_score) / abs(max([true_score, pair_score]))

print("Definitional words had on average a {0} percent difference between genders".format(torch.mean(def_score) * 100))
print("Potentially biased words had on average a {0} percent difference between genders".format(torch.mean(bias_score) * 100))
