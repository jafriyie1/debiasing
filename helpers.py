import torch

def get_input(tokenizer, sentence):
    sentence_tokenized = [tokenizer.tokenize(i) for i in sentence]
    sentence_ids = [tokenizer.convert_tokens_to_ids(i) for i in sentence_tokenized]
    sentence_input = torch.tensor(sentence_ids)
    return sentence_input