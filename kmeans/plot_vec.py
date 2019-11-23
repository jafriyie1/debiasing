from time import sleep
from allennlp.commands.elmo import ElmoEmbedder
import os
import pandas as pd
import numpy as np
import argparse
from numpy.linalg import norm
from scipy.linalg import orth
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
#import seaborn as sns
import matplotlib
#matplotlib.use( 'tkagg' , force=True)
# print(matplotlib.get_backend())
import matplotlib.pyplot as plt
# plt.switch_backend('tkagg')

LANG_SPECIFIC_DATA = {
    'en': {
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
    },
    'hi': {
        'pairs': [['लड़का', 'लड़की']],
        'pairs_two': [
            ['आदमी', 'औरत'],
            ['बेटा', 'बेटी'],
            ['बाप', 'माँ'],
        ],
        'tok_sent': lambda x: [x, 'ने', 'नाश्ते', 'के', 'लिए', 'एक', 'सेब', 'खाया'],
        'tok_sent_prof': lambda x: [x, 'ने', 'नाश्ते', 'के', 'लिए', 'एक', 'सेब', 'खाया']
    }
}

def get_gender_basis(elmo, lang):
    bias_subspace = []
    pairs = LANG_SPECIFIC_DATA[lang]['pairs']
    pairs_two = LANG_SPECIFIC_DATA[lang]['pairs_two']
    female, male = pairs[0]

    tok_sent_f = LANG_SPECIFIC_DATA[lang]['tok_sent'](female)
    tok_sent_m = LANG_SPECIFIC_DATA[lang]['tok_sent'](male)

    vectors = elmo.embed_sentence(tok_sent_f)
    bias_subspace.append(vectors[2][1])

    vectors2 = elmo.embed_sentence(tok_sent_m)
    bias_subspace.append(vectors2[2][1])

    for pair in pairs_two:
        female, male = pair

        tok_sent_f = LANG_SPECIFIC_DATA[lang]['tok_sent_prof'](female)
        tok_sent_m = LANG_SPECIFIC_DATA[lang]['tok_sent_prof'](male)

        vectors = elmo.embed_sentence(tok_sent_f)
        bias_subspace.append(vectors[2][1])

        vectors2 = elmo.embed_sentence(tok_sent_m)
        bias_subspace.append(vectors2[2][1])

    bias_subspace = np.array(bias_subspace)
    basis = bias_subspace

    return basis


def get_stereotype_words(path):
    df = pd.read_csv(path, sep='\t')
    adj_df_list = list(df.values.flatten())

    return adj_df_list


def get_reg_words(path):
    df = pd.read_csv(path, sep='\t')
    adj_df_list = list(df.values.flatten())

    return adj_df_list


def score_vectors(elmo, word_list, basis, lang):
    score_list = []

    for word in word_list:
        tok_sent = LANG_SPECIFIC_DATA[lang]['tok_sent_prof'](word)
        vec = elmo.embed_sentence(tok_sent)
        vec = vec[2][1]

        score = 0
        for b in basis:
            score += np.dot(b, vec) / (norm(b) * norm(vec))

        score_list.append((word, score))

    return score_list

def evaluate_projected_vectors(word_list, vectors):
    words_with_scores = [
        (word_list[i], np.linalg.norm(vectors[i])) for i in range(len(word_list))
    ]
    words_with_scores.sort(key = lambda w: w[1], reverse=True)
    print(words_with_scores)
    return words_with_scores

def proj_gen_space(elmo, word_list, basis, lang):
    proj_vectors = []
    for word in word_list:
        tok_sent = LANG_SPECIFIC_DATA[lang]['tok_sent_prof'](word)
        vec = elmo.embed_sentence(tok_sent)
        vec = vec[2][1]

        score = 0
        new_vec = 0
        for b in basis:
            # print(b.size)
            new_vec = (np.dot(b, vec) / norm(b)) * b

        proj_vectors.append(new_vec)

    return proj_vectors


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


def get_vectors(elmo, word_list, lang):
    vec_list = []
    new_word_list = word_list

    for word in word_list:
        tok_sent = LANG_SPECIFIC_DATA[lang]['tok_sent_prof'](word)
        vec = elmo.embed_sentence(tok_sent)
        vec = vec[2][1]
        vec_list.append(vec)

    return np.array(vec_list), new_word_list


def train_kmeans(X, words, n_clus=2):
    model = KMeans(n_clusters=n_clus).fit(X)
    labels = list(model.labels_)
    corr_words_and_labels = list(zip(words, labels))
    #corr_words_and_labels = sorted(corr_words_and_labels, key= lambda x: x[1])

    return model, corr_words_and_labels


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

parser = argparse.ArgumentParser()
parser.add_argument(
    '--lang', type=str, default='en', help='language to use')

def main():
    n_colors = 2
    opt = parser.parse_args()
    elmo = ElmoEmbedder()
    data_path = os.path.join(
        os.getcwd(), 'gp_debias', 'wordlist', opt.lang, 'stereotype_list.tsv')
    data_path2 = os.path.join(
        os.getcwd(), 'gp_debias', 'wordlist', opt.lang, 'no_gender_list.tsv')

    stereo_list = get_stereotype_words(data_path)
    no_gen_list = get_reg_words(data_path2)
    basis = get_gender_basis(elmo, opt.lang)

    #X_vecs, corr_word_list = get_vectors(elmo, stereo_list)
    X_vecs = proj_gen_space(elmo, stereo_list, basis, opt.lang)
    evaluate_projected_vectors(stereo_list, X_vecs)
    exit(0)
    model, labeled_words = train_kmeans(X_vecs, stereo_list, n_colors)

    print(sorted(labeled_words, key=lambda x: x[1]))
    pca_viz(X_vecs, labeled_words, n_colors)
    scores = score_vectors(elmo, stereo_list, basis, opt.lang)
    stereo_scores = list(reversed(sorted(scores, key=lambda x: x[1])))

    print()
    print()
    # print(list(stereo_scores))

    gen_df(labeled_words, stereo_scores)

    print('Done')
    # sleep(100)


if __name__ == '__main__':
    main()
