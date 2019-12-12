import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers import DistilBertTokenizer
from transformers.optimization import AdamW
from transformers import DistilBertModel, DistilBertForMaskedLM
from mask_debias_dataset import MaskDebiasingDataset
from torch.utils.data import DataLoader, Dataset, random_split
import argparse 
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--lr', type=float, default=2e-5)

args = parser.parse_args()


# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
dataset = MaskDebiasingDataset('masked_training_data.jsonl')
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
    for data in tqdm(train_data):
        model.train()
        optimizer.zero_grad()
        tup1, tup2, original = data 
        def_masked, def_masked_tokens, def_idx = tup1
        def_masked_tokens = list(def_masked_tokens)
        bias_masked, bias_masked_tokens, bias_idx = tup2 
        bias_masked_tokens = list(bias_masked_tokens)

        def_masked = [tokenizer.tokenize(i) for i in def_masked]
        print(def_masked_tokens)
        def_masked_tokens = [tokenizer.tokenize(i) for i in def_masked_tokens[0]]
        bias_masked = [tokenizer.tokenize(i) for i in bias_masked]
        bias_masked_tokens = [tokenizer.tokenize(i) for i in bias_masked_tokens[0]]
        original = [tokenizer.tokenize(i) for i in original]

        print(len(def_masked[0]))
        print(len(bias_masked[0]))
        print(len(original[0]))

        def_masked = [tokenizer.convert_tokens_to_ids(i) for i in def_masked]
        def_masked_tokens = [tokenizer.convert_tokens_to_ids(i) for i in def_masked_tokens]
        bias_masked = [tokenizer.convert_tokens_to_ids(i) for i in bias_masked]
        bias_masked_tokens = [tokenizer.convert_tokens_to_ids(i) for i in bias_masked_tokens]
        original = [tokenizer.convert_tokens_to_ids(i) for i in original]

        #print(def_masked)
        #print(original)

    

        #print('yup', def_masked[0])

        #print('Transform', tokenizer.convert_tokens_to_ids(def_masked[0]))
        #def_masked_tokens = tokenizer.convert_tokens_to_ids(def_masked_tokens)
        #bias_masked = tokenizer.convert_tokens_to_ids(bias_masked)
        #bias_masked_tokens = tokenizer.convert_tokens_to_ids(bias_masked_tokens)

        segments_ids_def  =  [1 for _ in range(len(def_masked[0]))]
        segments_ids_bias = [1 for _ in range(len(bias_masked[0]))]

        print(len(segments_ids_def))

        def_masked = torch.tensor(def_masked)
        def_masked_tokens = torch.tensor(def_masked_tokens)
        bias_masked = torch.tensor(bias_masked)
        bias_masked_tokens = torch.tensor(bias_masked_tokens)
        original = torch.tensor(original)

        segments_tensors_def = torch.tensor([segments_ids_def])
        
        segments_tensors_bias = torch.tensor(segments_ids_bias)
        print(original.size())
        print(bias_masked.size())
        #print(segments_ids_bias.size())
        if len(original[0]) == len(def_masked[0]):
            outputs = model(original, masked_lm_labels = def_masked)
            predictions = outputs[1]
            loss = outputs[0]
            tr_loss += loss


        #loss += c_loss(predictions, def_masked_tokens)
        if len(original[0]) == len(bias_masked[0]):
            outputs = model(original, masked_lm_labels = bias_masked)
            predictions = outputs[1]
            tr_loss += outputs[0]
            #loss += c_loss(predictions, def_masked_tokens)

        tr_loss.backward(retain_graph=True)
        optimizer.step()


        #print(data)
    print("The loss at epoch {} is: {}".format(epoch+1, loss))

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