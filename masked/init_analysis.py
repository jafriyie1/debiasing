from pytorch_pretrained_bert import BertTokenizer,BertForMaskedLM
import torch
import pandas as pd
import math
import numpy as np
#from transformers import DistilBertTokenizer, DistilBertModel, DistilBertForSequenceClassification

bertMaskedLM = BertForMaskedLM.from_pretrained('bert-base-uncased')
bertMaskedLM.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def get_score(sentence):
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    predictions=bertMaskedLM(tensor_input)
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(predictions.squeeze(),tensor_input.squeeze()).data 
    return math.exp(loss)

arr = np.array([
        ['All author have seen the manuscript and approved to submit to your journal .',
       'All authors have seen the manuscript and approved to submit to your journal .',
       'All authors have seen the manuscript and have been approved to submit to your journal.',
       'All authors have seen the manuscript and approved it to be submitted to your journal .'],
       ['All author have seen the manuscript and approved to submit to your journal .',
       'All authors have seen the manuscript and approved to submit to your journal .',
       'All authors have seen the manuscript and have been approved to submit to your journal.',
       'All authors have seen the manuscript and approved it to be submitted to your journal .'],
]
)

for j in range(1):
    print(j, [get_score(i) for i in arr[j]])