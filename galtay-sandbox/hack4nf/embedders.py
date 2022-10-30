from collections import Counter
import itertools
import math
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


def create_skipgram_matrix(skipgram_weights, unigram_to_index):
    row_indexs = []
    col_indexs = []
    dat_values = []
    for (unigram_a, unigram_b), weight in tqdm(
        skipgram_weights.items(),
        desc="calculating skipgrams matrix"
    ):
        row_indexs.append(unigram_to_index[unigram_a])
        col_indexs.append(unigram_to_index[unigram_b])
        dat_values.append(weight)
    return sparse.csr_matrix((dat_values, (row_indexs, col_indexs)))


def calculate_ppmi_matrix(skipgram_matrix, skipgram_weights, token_to_index, ppmi_alpha):

    """This function uses the notation from LGD15
    https://aclanthology.org/Q15-1016/
    """

    # for standard PPMI
    DD = skipgram_matrix.sum()

    # #(w) is a sum over contexts and #(c) is a sum over words
    pound_w_arr = np.array(skipgram_matrix.sum(axis=1)).flatten()
    pound_c_arr = np.array(skipgram_matrix.sum(axis=0)).flatten()

    # for context distribution smoothing (cds)
    pound_c_alpha_arr = pound_c_arr**ppmi_alpha
    pound_c_alpha_norm = np.sum(pound_c_alpha_arr)

    row_indxs = []
    col_indxs = []
    ppmi_dat_values = []   # positive pointwise mutual information

    for skipgram in tqdm(
        skipgram_weights.items(),
        total=len(skipgram_weights),
        desc='calculating ppmi matrix',
    ):

        (word, context), pound_wc = skipgram
        word_indx = token_to_index[word]
        context_indx = token_to_index[context]

        pound_w = pound_w_arr[word_indx]
        pound_c = pound_c_arr[context_indx]
        pound_c_alpha = pound_c_alpha_arr[context_indx]

        # this is how you would write the probabilities
        # Pwc = pound_wc / DD
        # Pw = pound_w / DD
        # Pc = pound_c / DD
        # Pc_alpha = pound_c_alpha / pound_c_alpha_norm

        # its more computationally effecient to use the counts directly
        pmi = np.log2((pound_wc * DD) / (pound_w * pound_c))
        pmia = np.log2((pound_wc * pound_c_alpha_norm) / (pound_w * pound_c_alpha))

        ppmi = max(pmi, 0)
        ppmia = max(pmia, 0)

        row_indxs.append(word_indx)
        col_indxs.append(context_indx)
        ppmi_dat_values.append(ppmia)

    return sparse.csr_matrix((ppmi_dat_values, (row_indxs, col_indxs)))


def calculate_svd_matrix(high_dim_matrix, embedding_size, svd_p):
    uu, ss, vv = linalg.svds(high_dim_matrix, embedding_size)
    svd_matrix = uu.dot(np.diag(ss**svd_p))
    return svd_matrix


class GeneMutationEmbeddings:

    def __init__(
        self,
        ser_genes_in_samples,
        subset_name,
        min_unigram_count=10,
        embedding_size=200,
        ppmi_alpha=0.75,
        svd_p=1.0,
    ):

        self.ser_genes_in_samples = ser_genes_in_samples
        self.subset_name = subset_name
        self.min_unigram_count = min_unigram_count
        self.embedding_size = embedding_size
        self.ppmi_alpha = ppmi_alpha
        self.svd_p = svd_p


    def create_embeddings(self):

        unigram_counts, skipgram_weights = self.calculate_grams(
            self.ser_genes_in_samples,
            self.min_unigram_count,
        )
        index_to_gene = {ii: gene for ii, gene in enumerate(unigram_counts.keys())}
        gene_to_index = {gene: ii for ii, gene in index_to_gene.items()}
        skipgram_matrix = create_skipgram_matrix(skipgram_weights, gene_to_index)

        ppmi_matrix = calculate_ppmi_matrix(
            skipgram_matrix,
            skipgram_weights,
            gene_to_index,
            self.ppmi_alpha,
        )

        svd_matrix = calculate_svd_matrix(
            ppmi_matrix,
            self.embedding_size,
            self.svd_p,
        )

        index_to_sample = {
            ii: sample_id for ii, sample_id in enumerate(self.ser_genes_in_samples.index)}
        sample_to_index = {
            sample_id: ii for ii, sample_id in index_to_sample.items()}

        sample_vecs = self.calculate_sample_vectors(
            self.ser_genes_in_samples,
            svd_matrix,
            unigram_counts,
            gene_to_index,
            sample_to_index,
        )


        self.unigram_counts = unigram_counts
        self.skipgram_weights = skipgram_weights
        self.skipgram_matrix = skipgram_matrix
        self.ppmi_matrix = ppmi_matrix
        self.svd_matrix = svd_matrix
        self.index_to_sample = index_to_sample
        self.sample_to_index = sample_to_index
        self.index_to_gene = index_to_gene
        self.gene_to_index = gene_to_index
        self.sample_vecs = sample_vecs


    def calculate_grams(self, ser_genes_in_samples, min_unigram_count):

        all_unigram_counts = Counter()
        for row in tqdm(ser_genes_in_samples, desc='calculating unigrams'):
            for gene in row:
                all_unigram_counts[gene] += 1
        unigram_counts = filter_counter(all_unigram_counts, min_val=min_unigram_count)

        skipgram_weights = Counter()
        for full_row in tqdm(ser_genes_in_samples, desc='calculating skipgrams'):
            # filter out genes that were filtered from unigrams
            row = [gene for gene in full_row if gene in unigram_counts]
            # normalize for the fact that we take all permuations instead of a sliding window
            # in other words we have to correct for the fact that a gene will appear in more
            # skipgrams if its in a longer sentence (which wouldn't happen for a sliding window)
            # this norm is the number of permuations each element will appear in
            length_norm = max(1, (len(row) - 1) * 2)

            perms = list(itertools.permutations(row, 2))
            for gene_a, gene_b in perms:
                weight = 1 / length_norm
                skipgram = (gene_a, gene_b)
                skipgram_weights[skipgram] += weight

        return unigram_counts, skipgram_weights


    def calculate_sample_vectors(
        self,
        ser_genes_in_samples,
        gene_vecs,
        unigram_counts,
        gene_to_index,
        sample_to_index,
    ):

        num_samples = ser_genes_in_samples.size
        embedding_size = gene_vecs.shape[1]
        sample_vecs = np.zeros((num_samples, embedding_size))
        for sample_id, full_row in tqdm(ser_genes_in_samples.items(), desc='making sample vectors'):
            row = [gene for gene in full_row if gene in unigram_counts]
            vec = np.zeros(embedding_size)
            norm = len(row) if len(row) > 0 else 1
            for gene in row:
                gene_index = gene_to_index[gene]
                gene_vec = gene_vecs[gene_index]
                vec += gene_vec
            vec = vec / norm
            sample_index = sample_to_index[sample_id]
            sample_vecs[sample_index,:] = vec
        return sample_vecs


    def write_projector_files(self, df_dcs, path, tag):

        # write out gene level embeddings
        #====================================================================
        df_vecs = pd.DataFrame(self.ppmi_matrix.todense())
        df_vecs.to_csv(
            os.path.join(path, f'{tag}_gene_ppmi_vecs.tsv'),
            sep='\t',
            index=False,
            header=False,
        )

        df_vecs = pd.DataFrame(self.svd_matrix)
        df_vecs.to_csv(
            os.path.join(path, f'{tag}_gene_svd_{self.embedding_size}_vecs.tsv'),
            sep='\t',
            index=False,
            header=False,
        )


        # write out gene level metadata
        #====================================================================

        # record gene names -> index
        df_meta = pd.DataFrame(
            [self.index_to_gene[ii] for ii in range(len(self.index_to_gene))],
            columns=["gene"],
        )

        # record gene unigram counts
        df_ucnt = pd.DataFrame(self.unigram_counts.items(), columns=["gene", 'unigram_count'])
        df_meta = pd.merge(
            df_meta,
            df_ucnt,
            on="gene")

        df_meta.to_csv(os.path.join(path, f'{tag}_gene_meta.tsv'), sep='\t', index=False)


        # write out sample level embeddings
        #====================================================================
        df_vecs = pd.DataFrame(self.sample_vecs)
        df_vecs.to_csv(
            os.path.join(path, f'{tag}_sample_vecs.tsv'),
            sep='\t',
            index=False,
            header=False,
        )

        # write out sample level metadata
        #====================================================================

        df_meta = pd.DataFrame(
            [self.index_to_sample[ii] for ii in range(len(self.index_to_sample))],
            columns=["SAMPLE_ID"],
        )

        # reocrd sample metadata from data clinical sample
        df_meta = pd.merge(
            df_meta,
            df_dcs,
            on='SAMPLE_ID',
            how='left',
        )

        # record sample metadata from data mutations extended
        df_tmp = self.ser_genes_in_samples.to_frame().reset_index().rename(columns={"Hugo_Symbol": "Mutated_Hugo"})
        df_tmp['Mutated_Hugo'] = df_tmp['Mutated_Hugo'].apply(set)

        df_meta = pd.merge(
            df_meta,
            df_tmp,
            on='SAMPLE_ID',
        )

        df_meta['CENTER'] = df_meta['SAMPLE_ID'].apply(lambda x: x.split('-')[1])
        CENTER_CODES = ['MSK', 'DFCI']
        for center in CENTER_CODES:
            df_meta[f'{center}_flag'] = (df_meta['CENTER']==center).astype(int)


        HUGO_CODES = ['NF1', 'NF2', 'SMARCB1', 'LZTR1']
        for hugo in HUGO_CODES:
            df_meta[f'{hugo}_mut'] = df_meta['Mutated_Hugo'].apply(lambda x: hugo in x).astype(int)

        ONCOTREE_CODES = ['NST', 'MPNST', 'NFIB', 'SCHW', 'CSCHW', 'MSCHW']
        for oncotree in ONCOTREE_CODES:
            df_meta[f'{oncotree}_flag'] = (df_meta['ONCOTREE_CODE']==oncotree).astype(int)


        df_meta = df_meta.drop(['Mutated_Hugo'], axis=1)
        df_meta.to_csv(os.path.join(path, f'{tag}_sample_meta.tsv'), sep='\t', index=False)





class GeneCnaEmbeddings:

    def __init__(
        self,
        ser_genes_in_samples,
        subset_name,
        min_unigram_count=10,
        embedding_size=200,
        ppmi_alpha=0.75,
        svd_p=1.0,
    ):

        self.ser_genes_in_samples = ser_genes_in_samples
        self.subset_name = subset_name
        self.min_unigram_count = min_unigram_count
        self.embedding_size = embedding_size
        self.ppmi_alpha = ppmi_alpha
        self.svd_p = svd_p


    def create_embeddings(self):

        unigram_counts, skipgram_weights = self.calculate_grams(
            self.ser_genes_in_samples,
            self.min_unigram_count,
        )
        index_to_gene = {ii: gene for ii, gene in enumerate(unigram_counts.keys())}
        gene_to_index = {gene: ii for ii, gene in index_to_gene.items()}
        skipgram_matrix = create_skipgram_matrix(skipgram_weights, gene_to_index)

        ppmi_matrix = calculate_ppmi_matrix(
            skipgram_matrix,
            skipgram_weights,
            gene_to_index,
            self.ppmi_alpha,
        )

        svd_matrix = calculate_svd_matrix(
            ppmi_matrix,
            self.embedding_size,
            self.svd_p,
        )

        index_to_sample = {
            ii: sample_id for ii, sample_id in enumerate(self.ser_genes_in_samples.index)}
        sample_to_index = {
            sample_id: ii for ii, sample_id in index_to_sample.items()}

        sample_vecs = self.calculate_sample_vectors(
            self.ser_genes_in_samples,
            svd_matrix,
            unigram_counts,
            gene_to_index,
            sample_to_index,
        )


        self.unigram_counts = unigram_counts
        self.skipgram_weights = skipgram_weights
        self.skipgram_matrix = skipgram_matrix
        self.ppmi_matrix = ppmi_matrix
        self.svd_matrix = svd_matrix
        self.index_to_sample = index_to_sample
        self.sample_to_index = sample_to_index
        self.index_to_gene = index_to_gene
        self.gene_to_index = gene_to_index
        self.sample_vecs = sample_vecs


    def calculate_grams(self, ser_genes_in_samples, min_unigram_count):

        all_unigram_counts = Counter()
        for row in tqdm(ser_genes_in_samples, desc='calculating unigrams'):
            for gene, weight in row:
                all_unigram_counts[gene] += abs(weight)
        unigram_counts = filter_counter(all_unigram_counts, min_val=min_unigram_count)

        skipgram_weights = Counter()
        for full_row in tqdm(ser_genes_in_samples, desc='calculating skipgrams'):
            # filter out genes that were filtered from unigrams
            row = [(gene, weight) for (gene, weight) in full_row if gene in unigram_counts]
            # normalize for the fact that we take all permuations instead of a sliding window
            # in other words we have to correct for the fact that a gene will appear in more
            # skipgrams if its in a longer sentence (which wouldn't happen for a sliding window)
            # this norm is the number of permuations each element will appear in
            length_norm = max(1, (len(row) - 1) * 2)

            perms = list(itertools.permutations(row, 2))
            for (gene_a, weight_a), (gene_b, weight_b) in perms:
                weight = math.sqrt(weight_a**2 + weight_b**2) / length_norm
                skipgram = (gene_a, gene_b)
                skipgram_weights[skipgram] += weight

        return unigram_counts, skipgram_weights


    def calculate_sample_vectors(
        self,
        ser_genes_in_samples,
        gene_vecs,
        unigram_counts,
        gene_to_index,
        sample_to_index,
    ):

        num_samples = ser_genes_in_samples.size
        embedding_size = gene_vecs.shape[1]
        sample_vecs = np.zeros((num_samples, embedding_size))
        for sample_id, full_row in tqdm(ser_genes_in_samples.items(), desc='making sample vectors'):
            row = [(gene, weight) for (gene, weight) in full_row if gene in unigram_counts]
            vec = np.zeros(embedding_size)
            norm = len(row) if len(row) > 0 else 1
            for gene, weight in row:
                gene_index = gene_to_index[gene]
                gene_vec = gene_vecs[gene_index]
                vec += gene_vec
            vec = vec / norm
            sample_index = sample_to_index[sample_id]
            sample_vecs[sample_index,:] = vec
        return sample_vecs


    def write_projector_files(self, df_dcs, path, tag):

        # write out gene level embeddings
        #====================================================================
        df_vecs = pd.DataFrame(self.ppmi_matrix.todense())
        df_vecs.to_csv(
            os.path.join(path, f'{tag}_gene_ppmi_vecs.tsv'),
            sep='\t',
            index=False,
            header=False,
        )

        df_vecs = pd.DataFrame(self.svd_matrix)
        df_vecs.to_csv(
            os.path.join(path, f'{tag}_gene_svd_{self.embedding_size}_vecs.tsv'),
            sep='\t',
            index=False,
            header=False,
        )


        # write out gene level metadata
        #====================================================================

        # record gene names -> index
        df_meta = pd.DataFrame(
            [self.index_to_gene[ii] for ii in range(len(self.index_to_gene))],
            columns=["gene"],
        )

        # record gene unigram counts
        df_ucnt = pd.DataFrame(self.unigram_counts.items(), columns=["gene", 'unigram_count'])
        df_meta = pd.merge(
            df_meta,
            df_ucnt,
            on="gene")

        df_meta.to_csv(os.path.join(path, f'{tag}_gene_meta.tsv'), sep='\t', index=False)


        # write out sample level embeddings
        #====================================================================
        df_vecs = pd.DataFrame(self.sample_vecs)
        df_vecs.to_csv(
            os.path.join(path, f'{tag}_sample_vecs.tsv'),
            sep='\t',
            index=False,
            header=False,
        )

        # write out sample level metadata
        #====================================================================

        df_meta = pd.DataFrame(
            [self.index_to_sample[ii] for ii in range(len(self.index_to_sample))],
            columns=["SAMPLE_ID"],
        )

        # reocrd sample metadata from data clinical sample
        df_meta = pd.merge(
            df_meta,
            df_dcs,
            on='SAMPLE_ID',
            how='left',
        )

        # record sample metadata from data mutations extended
        ser_tmp = self.ser_genes_in_samples.apply(lambda x: [el[0] for el in x])
        df_tmp = ser_tmp.to_frame('CNA_Hugo').reset_index()
        df_tmp['CNA_Hugo'] = df_tmp['CNA_Hugo'].apply(set)

        df_meta = pd.merge(
            df_meta,
            df_tmp,
            on='SAMPLE_ID',
        )

        df_meta['CENTER'] = df_meta['SAMPLE_ID'].apply(lambda x: x.split('-')[1])
        CENTER_CODES = ['MSK', 'DFCI']
        for center in CENTER_CODES:
            df_meta[f'{center}_flag'] = (df_meta['CENTER']==center).astype(int)

        HUGO_CODES = ['NF1', 'NF2', 'SMARCB1', 'LZTR1']
        for hugo in HUGO_CODES:
            df_meta[f'{hugo}_cna'] = df_meta['CNA_Hugo'].apply(lambda x: hugo in x).astype(int)

        ONCOTREE_CODES = ['NST', 'MPNST', 'NFIB', 'SCHW', 'CSCHW', 'MSCHW']
        for oncotree in ONCOTREE_CODES:
            df_meta[f'{oncotree}_flag'] = (df_meta['ONCOTREE_CODE']==oncotree).astype(int)

        df_meta = df_meta.drop(['CNA_Hugo'], axis=1)
        df_meta.to_csv(os.path.join(path, f'{tag}_sample_meta.tsv'), sep='\t', index=False)
