import json
STRIP_STR = ' ,\'":./\n`'

class MaskDebiasingDataset():
    def __init__(self, data_file, definitional_pairs_file, mask_token):
        self.data_file = data_file
        self.definitional_pairs_file = definitional_pairs_file
        self.mask_token = mask_token
        self.data = []
        with open(self.data_file) as f:
            for line in f:
                self.data.append(json.loads(line.strip(STRIP_STR)))
        self.pairs = {}
        with open(self.definitional_pairs_file) as f:
            for line in f:
                w1, w2 = [w.strip(STRIP_STR) for w in line.split(',')]
                self.pairs[w1] = w2
                self.pairs[w2] = w1
                
    def __getitem__(self, index):
        datum = self.data[index]

        original = datum['sentence']
        tokens = [w for w in original.split()] # not stripping here because we want original punctuation

        definitional_masked_tokens = []
        definitional_tokens = [t for t in tokens]
        for idx in datum['definitional']:
            def_idx = int(idx)
            definitional_masked_tokens.append(definitional_tokens[def_idx].lower().strip(STRIP_STR))
            definitional_tokens[def_idx] = self.mask_token
        definitional_masked = ' '.join(definitional_tokens)

        idx = datum['bias']
        bias_idx = int(idx)
        bias_tokens = [t for t in tokens]
        bias_masked_token = bias_tokens[bias_idx].lower().strip(STRIP_STR)
        bias_tokens[bias_idx] = self.mask_token
        bias_masked = ' '.join(bias_tokens)

        # This represents masking the biased word, and swapping the definitional words
        bias_pair_tokens = [t for t in tokens]
        bias_pair_tokens[bias_idx] = self.mask_token
        for idx in datum['definitional']:
            def_idx = int(idx)
            bias_pair_tokens[def_idx] = self.get_pair(bias_pair_tokens[def_idx].lower().strip(STRIP_STR))
        bias_pair_masked = ' '.join(bias_pair_tokens)

        return ((definitional_masked, definitional_masked_tokens, def_idx), 
                (bias_masked, bias_pair_masked, bias_idx),
                     original)

    def get_pair(self, word):
        return self.pairs[word]

    def __len__(self):
        return len(self.data)