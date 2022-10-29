from collections import Counter
import itertools
import os

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import linalg
from tqdm import tqdm

from hack4nf import genie


def filter_counter(counter_orig, min_val=None, max_val=None):
    counter = Counter(counter_orig)
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

    def __init__(self, ser_tokens, min_unigram_count=10, embedding_size=200, ppmi_alpha=0.75, svd_p=1.0):
        self.ser_tokens = ser_tokens
        self.min_unigram_count = min_unigram_count
        self.embedding_size = embedding_size
        self.ppmi_alpha = ppmi_alpha
        self.svd_p = svd_p

        self.calculate_grams()
        self.create_skipgram_matrix()
        self.calculate_ppmi_matrix()
        self.calculate_svd_matrix()
        self.calculate_sample_vectors()


    def calculate_grams(self):

        all_unigram_counts = Counter()
        for row in tqdm(self.ser_tokens, desc='calculating unigrams'):
            for token in row:
                all_unigram_counts[token] += 1

        unigram_counts = filter_counter(all_unigram_counts, min_val=self.min_unigram_count)
        index_to_token = {ii: token for ii, token in enumerate(unigram_counts.keys())}
        token_to_index = {token: ii for ii, token in index_to_token.items()}

        skipgram_weights = Counter()
        for full_row in tqdm(self.ser_tokens, desc='calculating skipgrams'):
            # filter out tokens that were filtered from unigrams
            row = [token for token in full_row if token in unigram_counts]
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


    def create_skipgram_matrix(self):
        row_indexs = []
        col_indexs = []
        dat_values = []
        for (tok1, tok2), weight in tqdm(self.skipgram_weights.items(), desc="calculating skipgrams matrix"):
            row_indexs.append(self.token_to_index[tok1])
            col_indexs.append(self.token_to_index[tok2])
            dat_values.append(weight)
        mat = sparse.csr_matrix((dat_values, (row_indexs, col_indexs)))
        self.skipgram_mat = mat


    def calculate_ppmi_matrix(self):

        # for standard PPMI
        DD = self.skipgram_mat.sum()
        sum_over_words = np.array(self.skipgram_mat.sum(axis=0)).flatten()
        sum_over_contexts = np.array(self.skipgram_mat.sum(axis=1)).flatten()

        # for context distribution smoothing (cds)
        sum_over_words_alpha = sum_over_words**self.ppmi_alpha
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


    def calculate_svd_matrix(self):

        uu, ss, vv = linalg.svds(self.ppmi_mat, self.embedding_size)
        svd_mat = uu.dot(np.diag(ss**self.svd_p))
        self.svd_mat = svd_mat


    def calculate_sample_vectors(self):

        self.index_to_sample = {
            ii: sample_id for ii, sample_id in enumerate(self.ser_tokens.index)}
        self.sample_to_index = {
            sample_id: ii for ii, sample_id in self.index_to_sample.items()}

        # create num_samples X num_tokens zeros matrix then fill it
        sample_vecs = np.zeros((
            len(self.index_to_sample),
            self.svd_mat.shape[1],
        ))
        for sample_id, full_row in tqdm(self.ser_tokens.items(), desc='making sample vectors'):
            row = [token for token in full_row if token in self.unigram_counts]
            vec = np.zeros(self.svd_mat.shape[1])
            for token in row:
                token_index = self.token_to_index[token]
                token_vec = self.svd_mat[token_index]
                vec += token_vec
            vec = vec / len(row)
            sample_index = self.sample_to_index[sample_id]
            sample_vecs[sample_index,:] = vec

        self.sample_vecs = sample_vecs


    def write_projector_files(self, path, tag, token_name):

        # write out token level embeddings
        #====================================================================
        df_vecs = pd.DataFrame(self.svd_mat)
        df_vecs.to_csv(
            os.path.join(path, f'{tag}_{token_name}_svd_vecs.tsv'),
            sep='\t',
            index=False,
            header=False,
        )

        # record token names -> index
        df_meta = pd.DataFrame(
            [self.index_to_token[ii] for ii in range(len(self.index_to_token))],
            columns=[token_name],
        )

        # record token unigram counts
        df_ucnt = pd.DataFrame(self.unigram_counts.items(), columns=[token_name, 'unigram_count'])
        df_meta = pd.merge(
            df_meta,
            df_ucnt,
            on=token_name)

        df_meta.to_csv(f'{tag}_{token_name}_svd_meta.tsv', sep='\t', index=False)


        # write out sample level embeddings
        #====================================================================

        df_vecs = pd.DataFrame(self.sample_vecs)
        df_vecs.to_csv(
            os.path.join(path, f'{tag}_sample_vecs.tsv'),
            sep='\t',
            index=False,
            header=False,
        )
