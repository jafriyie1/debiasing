import json
STRIP_STR = ' ,\'":./\n`'
MASK_TOKEN = '[MASK]' # TODO Find real mask token, pass it in?

class MaskDebiasingDataset():
    def __init__(self, data_file):
        self.data_file = data_file
        self.data = []
        with open(self.data_file) as f:
            for line in f:
                self.data.append(json.loads(line.strip(STRIP_STR)))

    def __getitem__(self, index):
        datum = self.data[index]

        original = datum['sentence']
        tokens = [w for w in original.split()] # not stripping here because we want original punctuation

        definitional_masked_tokens = []
        def_idx = -1 
        bias_idx = -1
        for idx in datum['definitional']:
            definitional_tokens = tokens
            def_idx = int(idx)
            definitional_masked_tokens.append(definitional_tokens[idx].lower().strip(STRIP_STR))
            definitional_tokens[int(idx)] = MASK_TOKEN
        definitional_masked = ' '.join(definitional_tokens)

        idx = datum['bias']
        bias_tokens = tokens
        bias_masked_token = bias_tokens[idx].lower().strip(STRIP_STR)
        bias_tokens[int(idx)] = MASK_TOKEN
        bias_idx = int(idx)

        bias_masked = ' '.join(bias_tokens)

        return ((definitional_masked, definitional_masked_tokens, def_idx), 
                (bias_masked, [bias_masked_token], bias_idx),
                     original)

    def __len__(self):
        return len(self.data)