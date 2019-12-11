# Script for tagging training data for debiasing masked LM task.
import os
import sys
import select
import argparse
import json
from mask_debias_dataset import STRIP_STR

# Sample run command:
# python create_debiasing_mask_data.py --output_file masked_training_data.jsonl --input_file ../realworldnlp/data/tatoeba/sentences.eng.200k.txt --definitional_words definitional_words.txt --start_idx 718

parser = argparse.ArgumentParser()

parser.add_argument('--output_file', type=str, help='file to which to write training examples', required=True)
parser.add_argument('--append', type=bool, help='whether or not to append to existing example file', default=True)
parser.add_argument('--input_file', type=str, help='file with newline separated input sentences to process', required=True)
parser.add_argument('--start_idx', type=int, help='index in input file to start it, in case some has already been processed')
parser.add_argument('--definitional_words', type=str, help='path to a file containing definitionally gendered words we should look for', required=False)

opt = parser.parse_args()

outf = open(opt.output_file, 'a' if opt.append else 'w')
inputf = open(opt.input_file, 'r')

def clean_word(word):
    return word.lower().strip(STRIP_STR)

def save_example(sentence, definitional_indices, bias_idx):
    line_info = json.dumps({"sentence": sentence, "definitional": definitional_indices, "bias": bias_idx})
    outf.write(line_info + '\n')

# Handles input as index or text
def process_input(tokens, value, description, possible_words=None):
    try:
        idx = int(value)
        if idx >= 0 and idx < len(tokens):
            print(description + " gendered word: " + tokens[idx])
            if possible_words and clean_word(tokens[idx]) not in possible_words:
                print("invalid word, please try again")
                return None
            return idx
        else:
            print("invalid index provided, please try again")
            return None
    except ValueError as e:
        indices = [idx for idx in filter(lambda i: clean_word(tokens[i]) == value, range(len(tokens)))]
        if len(indices) == 1:
            print(description + " index: " + str(indices[0]))
            return indices[0]
        else:
            print("word not found or was ambiguous, please try again")
            return None

#Handles a single sentence. If it returns False, something about the processing failed and the user should be prompted with this sentence again
def handle_sentence(sentence, index, definitional_words=None):
    print("------------sentence {0}------------------".format(index))
    print(sentence)
    tokens = [tok.strip(STRIP_STR) for tok in sentence.split()]
    biased = input("does this sentence contain potentially biased data? ")
    if not biased or str.lower(biased) == 'n':
        return True
    if biased == 'exit':
        finish()
    
    definitional_indices = []

    definitional = input("indices of definitional gender words, or the words themselves: ")
    definitional_list = definitional.split(', ')
    for d in definitional_list:
        definitional_idx = process_input(tokens, d, 'definitional', definitional_words)
        if (definitional_idx == None):
            return False
        definitional_indices.append(definitional_idx)

    bias = input("index of the potentially biased word, or the word itself: ")
    bias_idx = process_input(tokens, bias, 'biased')
    if (bias_idx == None):
        return False

    print("saving information for sentence...")
    save_example(sentence, definitional_indices, bias_idx)

    return True

def sentence_getter():
    line = inputf.readline()
    return line.strip()

def contains_words(sentence, words):
    return any([clean_word(w) in words for w in sentence.split()])

def finish():
    inputf.close()
    outf.close()
    exit(0)

definitional_words = []
if opt.definitional_words:
    with open(opt.definitional_words) as f:
        for w in f:
            definitional_words.append(clean_word(w))

index = 0
while(inputf.readable() and index < 10000):
    sentence = sentence_getter()
    if sentence == '':
        break
    
    #respect start_idx
    if index < opt.start_idx - 1:
        index += 1
        continue

    # respect definitional_words
    if len(definitional_words) > 0:
        if not contains_words(sentence, definitional_words):
            index += 1
            continue

    handled = False
    # keep showing each sentence until it is processed correctly
    while not handled:
        handled = handle_sentence(sentence, index, definitional_words=definitional_words)
        print()
    index += 1

finish()