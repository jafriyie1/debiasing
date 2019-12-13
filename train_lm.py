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
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--save', type=str, default='trained_lm.pth')

args = parser.parse_args()


# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
dataset = MaskDebiasingDataset('masked_training_data.jsonl', 'definitional_pairs.txt', tokenizer.mask_token)
model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
print(len(dataset))

batch_size = 8
train_data = DataLoader(dataset, batch_size=args.batch_size)

'''
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters,
                     lr=2e-5,
                     warmup=.1)
'''

optimizer = AdamW(model.parameters(), 
                    lr=args.lr, 
                    correct_bias=False)

c_loss = torch.nn.CrossEntropyLoss()

for epoch in range(args.epochs):
    tr_loss = 0
    epoch_loss = 0
    for data in tqdm(train_data):
        model.train()
        optimizer.zero_grad()
        tup1, tup2, original = data 
        def_masked, def_masked_words, def_idx = tup1
        bias_masked, bias_pair_masked, bias_masked_token, bias_idx = tup2 

        segments_ids_def = [1 for _ in range(len(def_masked[0]))]
        segments_ids_bias = [1 for _ in range(len(bias_masked[0]))]

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

        loss = torch.tensor([0.0])

        for def_mask_idx in def_mask_indices:
            pair_scores = predictions[0][def_mask_idx][word_pair_indices]
            # Now add this to loss
            loss += torch.flatten(torch.abs(pair_scores[0] - pair_scores[1])) / abs(torch.max(pair_scores, dim=0)[0])

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

        loss += torch.abs(true_score - pair_score) / abs(max([true_score, pair_score]))

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print("The loss at epoch {} is: {}".format(epoch+1, epoch_loss))

torch.save(model.state_dict(), args.save)
'''
# Tokenize input
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = tokenizer.tokenize(text)

# Mask a token that we will try to predict back with `BertForMaskedLM`
masked_index = 8
tokenized_text[masked_index] = '[MASK]'
assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer', '[SEP]']

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# Load pre-trained model (weights)
model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
model.eval()

# If you have a GPU, put everything on cuda
tokens_tensor = tokens_tensor.to('cuda')
segments_tensors = segments_tensors.to('cuda')
model.to('cuda')

# Predict all tokens
with torch.no_grad():
    outputs = model(tokens_tensor, segments_tensors)
    predictions = outputs[0]

# confirm we were able to predict 'henson'
predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]

predicted_tokens = tokenizer.convert_ids_to_tokens(365)

print(predicted_tokens)

print(predictions[0, masked_index])
print(predicted_token)

assert predicted_token == 'henson'
'''