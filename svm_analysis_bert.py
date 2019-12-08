from time import sleep
import os
import pandas as pd
import numpy as np
import argparse
from numpy.linalg import norm
from scipy.linalg import orth
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import matplotlib

import matplotlib.pyplot as plt
plt.switch_backend('tkagg')
import random
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle 
import sys
sys.path.append(os.path.join('utils_nlp', 'models', 'transformers'))

import torch
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertForSequenceClassification
from utils_nlp.models.transformers.sequence_classification import Processor, SequenceClassifier

import numpy as np
import pickle 
import argparse
import scipy


def generate_example_from_occupation(x):
    choices = [
        (['They', 'have', 'been', 'an', x, 'for', 'many', 'years'], 4),
        (['Today', ',', 'the', x, 'walked', 'down', 'the', 'street'], 3),
        (['"', 'Yesterday', 'was', 'Tuesday', '"', ',', 'said', 'the', x], 8),
        (['There', 'are', 'many', 'types', 'of', x], 5),
        (['The', x, 'ate', 'an', 'apple', 'for', 'dinner'], 1)
    ]
    idx = np.random.choice(len(choices))
    return choices[idx]

def generate_example(x):
    return ([x, "ate", "an", "apple", "for", "breakfast"], 0)

def generate_trivial_example(x):
    return ([x], 0)

LANG_SPECIFIC_DATA = {
    'en': {
        'pairs_two': [  ],
        #     ["woman", "man"],
        #     ["girl", "boy"],
        #     ["mother", "father"],
        #     ["daughter", "son"],
        #     ["gal", "guy"],
        #     ["female", "male"],
        #     ["lord", "lady"]
        # ],
        'pairs': [['she', 'he']],
        'tok_sent': generate_trivial_example,
        'tok_sent_prof': generate_trivial_example
    }
}

def get_tokens(tokenizer, sent):
    text = " ".join(sent[0])
    marked_text = "[CLS] " + text + " [SEP]" 
    tokenized_text = tokenizer.tokenize(marked_text)
    print(tokenized_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    return tokens_tensor, segments_tensors

def get_embedding_vec(tokenizer, tok_sent, model, singular=False):
    tokens_tensor, segments_tensors = \
        get_tokens(tokenizer, tok_sent)

    with torch.no_grad():
        encoded_layers = model(tokens_tensor, segments_tensors)

    if singular:
        indx = tok_sent[1] + 1

    return encoded_layers[0][0][indx] if singular else encoded_layers[0][0][:]

def create_subspace(lang, model, tokenizer):
    bias_subspace = []

    model.eval()
    
    pairs = LANG_SPECIFIC_DATA[lang]['pairs']
    pairs_two = LANG_SPECIFIC_DATA[lang]['pairs_two']
    female, male = pairs[0]

    tok_sent_f = LANG_SPECIFIC_DATA[lang]['tok_sent'](female)
    tok_sent_m = LANG_SPECIFIC_DATA[lang]['tok_sent'](male)

    bias_em = get_embedding_vec(tokenizer, tok_sent_f, model, singular=True).numpy()
    bias_em = list(bias_em)

    bias_subspace.append(bias_em)

    bias_em = get_embedding_vec(tokenizer, tok_sent_m, model, singular=True).numpy()
    bias_em = list(bias_em)

    bias_subspace.append(bias_em)

    for pair in pairs_two:
        female, male = pair

        tok_sent_f = LANG_SPECIFIC_DATA[lang]['tok_sent_prof'](female)
        tok_sent_m = LANG_SPECIFIC_DATA[lang]['tok_sent_prof'](male)

        vec = get_embedding_vec(tokenizer, tok_sent_f, model, singular=True).numpy()
        vec = list(vec)

        bias_subspace.append(vec)

        vec2 =  get_embedding_vec(tokenizer, tok_sent_m, model, singular=True).numpy()
        vec2 = list(vec2)

        bias_subspace.append(vec2)

    bias_subspace = np.array(bias_subspace)
    basis = bias_subspace
    basis = scipy.linalg.orth(basis.transpose()).transpose()

    print('Saving subspace....')
    
    with open('kmeans/distilbert_subspace.pkl', 'wb') as f:
        pickle.dump(basis,f)
    

    #print('basis shape: {}'.format(basis.shape))
    return basis

def load_subspace():
    with open('kmeans/distilbert_subspace.pkl', 'rb') as f:
        subspace = pickle.load(f)
    print('original subspace was loaded')
    return subspace

def get_stereotype_words(path):
    df = pd.read_csv(path, sep='\t')
    adj_df_list = list(df.values.flatten())

    return adj_df_list


def get_reg_words(path):
    df = pd.read_csv(path, sep='\t')
    adj_df_list = list(df.values.flatten())

    return adj_df_list

def proj_gen_space(tokenizer, model, word_list, basis, lang):
    proj_vectors = []
    tok_sentences = []
    for word in word_list:
        tok_sent = LANG_SPECIFIC_DATA[lang]['tok_sent_prof'](word)
        vecs = np.array(get_embedding_vec(tokenizer, tok_sent, model))

        score = 0
        new_vecs = []
        for v in vecs:
            val = np.zeros(v.shape)
            for b in basis:
                # print(b.size)
                val += np.dot(b, v) / norm(b) * b
            new_vecs.append(val)

        proj_vectors.append(np.array(new_vecs))
        tok_sentences.append(['[CLS]'] + tok_sent[0] + ['[SEP]'])

    return np.array(proj_vectors), tok_sentences

def pca_viz(X, words_labels, n_colors=2):
    from mpl_toolkits.mplot3d import Axes3D

    pca = PCA(n_components=3)
    pca_results = pca.fit_transform(X)
    labels = np.asarray([x[1] for x in words_labels])
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.scatter(pca_results[:, 0], pca_results[:, 1], pca_results[:, 2],
               c=labels, edgecolor='none', alpha=0.5,
               cmap=plt.cm.get_cmap('Accent', n_colors))

    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.ylabel('component 3')
    # plt.colorbar()
    plt.show()
    # sleep(100)
    # plt.savefig('test.png')

def train_kmeans(X, words, n_clus=2):
    sent2bert = KMeans(n_clusters=n_clus).fit(X)
    labels = list(sent2bert.labels_)
    corr_words_and_labels = list(zip(words, labels))
    words_labels_vecs = list(zip(words, X, labels))
    #corr_words_and_labels = sorted(corr_words_and_labels, key= lambda x: x[1])

    return sent2bert, corr_words_and_labels, words_labels_vecs

def gen_df(label_list, score_list):
    df_list = []
    label_list = sorted(label_list, key=lambda x: x[0])
    score_list = sorted(score_list, key=lambda x: x[0])

    for i, tups in enumerate(label_list):
        word, label = tups
        word_two, score = score_list[i]
        assert word_two == word
        temp_dict = {'word': word, 'score': score, 'label': label}
        df_list.append(temp_dict)

    df = pd.DataFrame(df_list)
    df.to_csv('analysis.csv', index=False)

def score_vectors(tokenizer, model, word_list, basis, lang):
    score_list = []

    for word in word_list:
        tok_sent = LANG_SPECIFIC_DATA[lang]['tok_sent_prof'](word)
        vec = get_embedding_vec(tokenizer, tok_sent, model, singular=True).numpy()
        #vec = LANG_SPECIFIC_DATA[lang]['getEmbedding'](elmo, tok_sent)

        score = 0
        for b in basis:
            score += np.dot(b, vec) / (norm(b) * norm(vec))

        #print(score)

        score_list.append((word, score))

    return score_list

def train_svm(data_list):
    random.shuffle(data_list)

    train_data = data_list[:int(len(data_list) * .8)]
    test_data = data_list[int(len(data_list) * .8):]

    def genX(sep_list):
        X = [x[1] for x in sep_list]
        y = [x[2] for x in sep_list]

        return X, y 

    train_X, train_y = genX(train_data)
    test_X, test_y = genX(test_data)

    clf = SVC()
    clf.fit(train_X, train_y)

    preds = clf.predict(test_X)
    acc = accuracy_score(test_y, preds)

    print('This is the accuracy score using the current basis: {}'.format(acc))

parser = argparse.ArgumentParser()
parser.add_argument(
    '--lang', type=str, default='en', help='language to use')
parser.add_argument('--model', type=str, default='n', help='use trained model (y or n)')
parser.add_argument('--load', type=str, default='n', help='use original subspace (y or n)')


def main():
    n_colors = 2
    opt = parser.parse_args()
    
    data_path = os.path.join(
        os.getcwd(), 'gp_debias', 'wordlist', opt.lang, 'occupation_stereotype_list.tsv')

    if opt.model == 'y':
        device = torch.device('cpu')
        n_model = SequenceClassifier(
            model_name='distilbert-base-uncased', num_labels=3, cache_dir='./cache'
        )

        con = DistilBertModel

        state_dict = torch.load("trained_1575511705.pth", map_location=device)
        # create new OrderedDict that does not contain `module. (To deal with pytorch bug)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        # load params
        n_model.model.load_state_dict(state_dict)
        model = n_model.model.distilbert

        print('loaded model')
    else:
        model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    model.eval()
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    stereo_list = get_stereotype_words(data_path)

    if opt.load == 'n':
        basis = create_subspace(opt.lang, model, tokenizer)
    else:
        basis = load_subspace()

    X_vecs, sentences = proj_gen_space(tokenizer, model, stereo_list, basis, opt.lang)
    # import pdb; pdb.set_trace()
    import pprint
    pp = pprint.PrettyPrinter(indent=2)
    import pdb; pdb.set_trace()        
    norms = norm(X_vecs, axis=2)
    for i in range(len(sentences)):
        pp.pprint(sorted(list(zip(norms[i], sentences[i])), key=lambda x: x[0]))
    # pp.pprint(sorted(list(zip([norms[i][2] for i in range(norms.shape[0])], stereo_list)), key=lambda x: x[0]))
    sent2bert, labeled_words, vecs_labels = train_kmeans(X_vecs, stereo_list, n_colors)

    if opt.load == 'n':
        train_svm(vecs_labels)

    print(sorted(labeled_words, key=lambda x: x[1]))
    pca_viz(X_vecs, labeled_words, n_colors)
    scores = score_vectors(tokenizer, model, stereo_list, basis, opt.lang)
    #print(scores)
    stereo_scores = list(reversed(sorted(scores, key=lambda x: x[1])))

    print()
    print()
    # print(list(stereo_scores))

    gen_df(labeled_words, stereo_scores)

    print('Done')
    # sleep(100)


if __name__ == '__main__':
    main()
