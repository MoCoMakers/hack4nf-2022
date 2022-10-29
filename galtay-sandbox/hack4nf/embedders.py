from collections import Counter
import itertools

import numpy as np
from scipy import sparse
from tqdm import tqdm


def filter_counter(counter, min_val=None, max_val=None):
    tokens_to_drop = []
    if min_val is not None:
        for token, val in counter.items():
            if val < min_val:
                tokens_to_drop.append(token)

    if max_val is not None:
        for token, val in counter.items():
            if val > max_val:
                tokens_to_drop.append(token)

    for token in tokens_to_drop:
        del counter[token]
    return counter


class TokenEmbeddings:

    def __init__(self, ser_tokens, min_unigram_count):
        self.ser_tokens = ser_tokens
        self.min_unigram_count = min_unigram_count
        self.calculate_grams()
        self.calculate_skipgram_matrix()
        self.calculate_ppmi_matrix()


    def calculate_grams(self):

        unigram_counts = Counter()
        for row in tqdm(self.ser_tokens, desc='calculating unigrams'):
            for token in row:
                unigram_counts[token] += 1

        unigram_counts = filter_counter(unigram_counts, min_val=self.min_unigram_count)
        index_to_token = {ii: token for ii, token in enumerate(unigram_counts.keys())}
        token_to_index = {token: ii for ii, token in index_to_token.items()}

        skipgram_weights = Counter()
        for row in tqdm(self.ser_tokens, desc='calculating skipgrams'):
            # filter out tokens that were filtered from unigrams
            row = [el for el in row if el in unigram_counts]
            perms = list(itertools.permutations(row, 2))
            for token_left, token_right in perms:
                # normalize for the fact that we take all permuations instead of a sliding window
                length_norm = (len(row) - 1) * 2
                weight = 1 / length_norm
                skipgram = (token_left, token_right)
                skipgram_weights[skipgram] += weight

        self.unigram_counts = unigram_counts
        self.skipgram_weights = skipgram_weights
        self.index_to_token = index_to_token
        self.token_to_index = token_to_index


    def calculate_skipgram_matrix(self):
        row_indexs = []
        col_indexs = []
        dat_values = []
        for (tok1, tok2), weight in tqdm(self.skipgram_weights.items(), desc="calculating skipgrams matrix"):
            row_indexs.append(self.token_to_index[tok1])
            col_indexs.append(self.token_to_index[tok2])
            dat_values.append(weight)
        mat = sparse.csr_matrix((dat_values, (row_indexs, col_indexs)))
        self.skipgram_mat = mat


    def calculate_ppmi_matrix(self, alpha=0.75):

        # for standard PPMI
        DD = self.skipgram_mat.sum()
        sum_over_words = np.array(self.skipgram_mat.sum(axis=0)).flatten()
        sum_over_contexts = np.array(self.skipgram_mat.sum(axis=1)).flatten()

        # for context distribution smoothing (cds)
        sum_over_words_alpha = sum_over_words**alpha
        Pc_alpha_denom = np.sum(sum_over_words_alpha)

        row_indxs = []
        col_indxs = []
        ppmi_dat_values = []   # positive pointwise mutual information

        for skipgram in tqdm(
            self.skipgram_weights.items(),
            total=len(self.skipgram_weights),
            desc='calculating ppmi matrix',
        ):

            (word, context), pound_wc = skipgram
            word_indx = self.token_to_index[word]
            context_indx = self.token_to_index[context]

            pound_w = sum_over_contexts[word_indx]
            pound_c = sum_over_words[context_indx]
            pound_c_alpha = sum_over_words_alpha[context_indx]

            Pwc = pound_wc / DD
            Pw = pound_w / DD
            Pc = pound_c / DD
            Pc_alpha = pound_c_alpha / Pc_alpha_denom

            pmi = np.log2(Pwc / (Pw * Pc_alpha))
            ppmi = max(pmi, 0)

            row_indxs.append(word_indx)
            col_indxs.append(context_indx)
            ppmi_dat_values.append(ppmi)

        mat = sparse.csr_matrix((ppmi_dat_values, (row_indxs, col_indxs)))
        self.ppmi_mat = mat
