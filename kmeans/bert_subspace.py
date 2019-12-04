import torch
from transformers import DistilBertTokenizer, DistilBertModel
import numpy as np
import pickle 
import argparse


LANG_SPECIFIC_DATA = {
    'en': {
        'getEmbedding': lambda elmo, sent: elmo.embed_sentence(sent)[2][0],
        'getEmbeddingProf': lambda elmo, sent: elmo.embed_sentence(sent)[2][1],
        'pairs_two': [
            ["woman", "man"],
            ["girl", "boy"],
            ["mother", "father"],
            ["daughter", "son"],
            ["gal", "guy"],
            ["female", "male"]
        ],
        'pairs': [['she', 'he']],
        'tok_sent': lambda x: [x, "ate", "an", "apple", "for", "breakfast"],
        'tok_sent_prof': lambda x: ['The', x, 'ate', 'an', 'apple', 'for', 'breakfast']
    }
}

def get_tokens(tokenizer, sent):
    text = " ".join(sent)
    marked_text = "[CLS] " + text + " [SEP]" 

    tokenized_text = tokenizer.tokenize(marked_text)
    print(tokenized_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    return tokens_tensor, segments_tensors

def get_embedding_vec(tokenizer, tok_sent, indx, model):
    tokens_tensor, segments_tensors = \
        get_tokens(tokenizer, tok_sent)

    with torch.no_grad():
        encoded_layers = model(tokens_tensor, segments_tensors)

    return encoded_layers[0][0][indx]


def create_subspace(lang, tr_model=None):
    bias_subspace = []

    if tr_model is None:
        model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    else:
        model = tr_model

    model.eval()

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    pairs = LANG_SPECIFIC_DATA[lang]['pairs']
    pairs_two = LANG_SPECIFIC_DATA[lang]['pairs_two']
    female, male = pairs[0]

    tok_sent_f = LANG_SPECIFIC_DATA[lang]['tok_sent'](female)
    tok_sent_m = LANG_SPECIFIC_DATA[lang]['tok_sent'](male)

    bias_em = get_embedding_vec(tokenizer, tok_sent_f, 0, model).numpy()
    bias_em = list(bias_em)

    bias_subspace.append(bias_em)

    bias_em = get_embedding_vec(tokenizer, tok_sent_m, 0, model).numpy()
    bias_em = list(bias_em)

    bias_subspace.append(bias_em)

    for pair in pairs_two:
        female, male = pair

        tok_sent_f = LANG_SPECIFIC_DATA[lang]['tok_sent_prof'](female)
        tok_sent_m = LANG_SPECIFIC_DATA[lang]['tok_sent_prof'](male)

        vec = get_embedding_vec(tokenizer, tok_sent_f, 1, model).numpy()
        vec = list(vec)

        bias_subspace.append(vec)

        vec2 =  get_embedding_vec(tokenizer, tok_sent_m, 1, model).numpy()
        vec2 = list(vec2)

        bias_subspace.append(vec2)

    bias_subspace = np.array(bias_subspace)
    basis = bias_subspace

    print('Saving subspace....')
    with open('distilbert_subspace.pkl', 'wb') as f:
        pickle.dump(basis,f)

    #print('basis shape: {}'.format(basis.shape))
    return basis

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
    '--lang', type=str, default='en', help='language to use')
    args = parser.parse_args()


    basis = create_subspace(args.lang)

if __name__ == '__main__':
    main()

