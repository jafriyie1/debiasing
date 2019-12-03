# NLP Final Project
## debiasing


### Dependencies:
 - allennlp
 - scipy
 - numpy
 - sklearn
 - ElmoForManyLangs (https://github.com/HIT-SCIR/ELMoForManyLangs)


## (x)BERT STUFF

#### bert_debiasing.py
Contains code for fine-tuning a bert model on the NLI task. Depends on:
https://github.com/microsoft/nlp-recipes/tree/master/utils_nlp

Also, depends on huggingface/transformers: https://github.com/huggingface/transformers

### Steps
We need to get bert_entailment.py up and running on a real GPU for one of the simplified bert models (alberta or distilbert).

Once that's done, we need to get edit the loss function in the corresponding "model" in the `transformers` repo.
    - save (pickle) our basis info to incorporate it into the loss function

To make ^ Possible, we want to install `transformers` using the "from source" method, with a `pip install --user -e .` or something similar.

Our folder structure should look like
`root`
    - `debiasing` (this repo)
        - `.../bert_entailment.py` training file
    - `nlp-recipes/utils_nlp` <- Can be a direct clone of the actual repo, or some sort of local install via PIP? So this folder may be optional
    - `transformers` <- This should be *our* fork of `huggingface/transformers`, which we will edit to change the loss function, etc.
