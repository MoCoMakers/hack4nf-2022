from collections import Counter
import itertools
import math
import os
import random
from typing import Callable, Dict, List, Optional, Tuple

from loguru import logger
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import linalg
from tqdm import tqdm

from nextgenlp import genie


SEED = 29712836
random.seed(SEED)
MAX_PPMI_NDIM_WRITE = 10_000

def nPr(n, r=2):
    """Number of permuations of size 2"""
    return int(math.factorial(n)/math.factorial(n-r))

SAMP_MULT = 2
MAX_LEN = 100
NUM_PERMS = {ii: nPr(ii) for ii in range(2,MAX_LEN+1)}
PERM_RATIO = {ii: (SAMP_MULT * ii) / NUM_PERMS[ii] for ii in range(2,MAX_LEN+1)}




def filter_counter(
    counter_orig: Counter,
    min_weight: Optional[int] = None,
    max_weight: Optional[int] = None,
) -> Counter:
    """Filter a counter removing values below `min_weight` and values above `max_weight`"""
    counter = Counter(counter_orig)
    to_drop = []
    if min_weight is not None:
        for thing, weight in counter.items():
            if weight < min_weight:
                to_drop.append(thing)

    if max_weight is not None:
        for thing, weight in counter.items():
            if weight > max_weight:
                to_drop.append(thing)

    for thing in to_drop:
        del counter[thing]

    return counter


def unigram_weighter_one(weight: float) -> float:
    return 1.0


def unigram_weighter_identity(weight: float) -> float:
    return weight


def unigram_weighter_abs(weight: float) -> float:
    return abs(weight)


class UnigramWeighter:
    def __init__(self, method:str, shift:float=0.0):
        self.method = method
        self.shift = shift

        if method == "one":
            self.call_me = unigram_weighter_one
        elif method == "identity":
            self.call_me = unigram_weighter_identity
        elif method == "abs":
            self.call_me = unigram_weighter_abs
        else:
            raise ValueError("method not recognized")

    def __call__(self, weight: float) -> float:
        return self.call_me(weight + self.shift)



def skipgram_weighter_one(weight_a: float, weight_b: float) -> float:
    return 1.0


def skipgram_weighter_product(weight_a: float, weight_b: float) -> float:
    return weight_a * weight_b


def skipgram_weighter_norm(weight_a: float, weight_b: float) -> float:
    return math.sqrt(weight_a ** 2 + weight_b ** 2)


class SkipgramWeighter:
    def __init__(self, method:str, shift:float=0.0):
        self.method = method
        self.shift = shift

        if method == "one":
            self.call_me = skipgram_weighter_one
        elif method == "product":
            self.call_me = skipgram_weighter_product
        elif method == "norm":
            self.call_me = skipgram_weighter_norm
        else:
            raise ValueError("method not recognized")

    def __call__(self, weight_a: float, weight_b: float) -> float:
        return self.call_me(weight_a + self.shift, weight_b + self.shift)


def random_int_except(a, b, no):
    """Generate a random integer between a and b (inclusive) avoiding no"""
    x = random.randint(a,b)
    while x == no:
        x = random.randint(a,b)
    return x


def calculate_grams(
    sentences: pd.Series,
    min_unigram_weight: int,
    unigram_weighter: Callable[[float], float],
    skipgram_weighter: Callable[[float, float], float],
) -> Tuple[Counter, Counter]:
    """Caclulate unigrams and skipgrams from sentences.

    Input:
      * sentences: a pd.Series with lists of (unigram, weight) tuples.
      * min_unigram_count: discard unigrams with weights below this
      * unigram_weighter: transform unigram weight in sentence to weight in counter
      * skipgram_weighter: transform skipgram weights in sentence to weight in counter

    """

    logger.info(
        f"calculating unigrams and skipgrams with min_unigram_weight={min_unigram_weight}"
    )

    all_unigram_weights = Counter()
    for sentence in tqdm(sentences, desc="calculating unigrams"):
        for unigram, weight in sentence:
            all_unigram_weights[unigram] += unigram_weighter(weight)
    logger.info("found {} unique unigrams".format(len(all_unigram_weights)))

    unigram_weights = filter_counter(all_unigram_weights, min_weight=min_unigram_weight)
    logger.info(
        "found {} unique unigrams after filtering by min_unigram_weight={}".format(
            len(unigram_weights), min_unigram_weight
        )
    )

    skipgram_weights = Counter()
    for full_sentence in tqdm(sentences, desc="calculating skipgrams"):
        # filter out unigrams
        sentence = [
            (unigram, weight)
            for (unigram, weight) in full_sentence
            if unigram in unigram_weights
        ]
        num_toks = len(sentence)

        if num_toks < 2:
            continue

        # if sentence length is <= MAX_LEN take all permutations and normalize
        # if sentence length > MAX_LEN take SAMP_MULT random samples for each entry

        if num_toks <= MAX_LEN:
            perms = list(itertools.permutations(sentence, 2))
            length_norm = PERM_RATIO[num_toks]
            for (unigram_a, weight_a), (unigram_b, weight_b) in perms:
                skipgram = (unigram_a, unigram_b)
                skipgram_weights[skipgram] += (
                    skipgram_weighter(weight_a, weight_b) * length_norm
                )
        else:
            for ii in range(num_toks):
                unigram_a, weight_a = sentence[ii]
                for nn in range(SAMP_MULT):
                    jj = random_int_except(0, num_toks-1, ii)
                    unigram_b, weight_b = sentence[jj]
                    skipgram = (unigram_a, unigram_b)
                    skipgram_weights[skipgram] += (
                        skipgram_weighter(weight_a, weight_b)
                    )

    logger.info("found {} unique unigrams".format(len(unigram_weights)))
    logger.info("found {} unique skipgrams".format(len(skipgram_weights)))

    return unigram_weights, skipgram_weights


def create_skipgram_matrix(
    skipgram_weights: Counter, unigram_to_index: Dict[str, int]
) -> sparse.csr_matrix:
    """Create a sparse skipgram matrix from a skipgram_weights counter.

    Input:
      * skipgram_weights: co-occurrence wrights between unigrams (counter)
      * unigram_to_index: maps unigram names to matrix indices
    """
    row_indexs = []
    col_indexs = []
    dat_values = []
    for (unigram_a, unigram_b), weight in tqdm(
        skipgram_weights.items(), desc="calculating skipgrams matrix"
    ):
        row_indexs.append(unigram_to_index[unigram_a])
        col_indexs.append(unigram_to_index[unigram_b])
        dat_values.append(weight)
    nrows = ncols = len(unigram_to_index)
    return sparse.csr_matrix(
        (dat_values, (row_indexs, col_indexs)), shape=(nrows, ncols)
    )


def calculate_ppmi_matrix(
    skipgram_matrix: sparse.csr_matrix,
    skipgram_weights: Counter,
    unigram_to_index: Dict[str, int],
    ppmi_alpha: float,
) -> sparse.csr_matrix:

    """Calculates positive pointwise mutual information from skipgrams.

    Input:
      * skipgram_matrix: co-occurrence weights between unigrams (matrix)
      * skipgram_weights: co-occurrence wrights between unigrams (counter)
      * unigram_to_index: maps unigram names to matrix indices
      * ppmi_alpha: context distribution smoothing factor (1.0 = None)

    This function uses the notation from LGD15
    https://aclanthology.org/Q15-1016/

    See https://github.com/MocoMakers/hack4nf-2022/blob/main/challenge-1/notebooks/genie_pmi_vectors.ipynb
    for further details.
    """

    # for normalizing counts to probabilities
    DD = skipgram_matrix.sum()

    # #(w) is a sum over contexts and #(c) is a sum over words
    pound_w_arr = np.array(skipgram_matrix.sum(axis=1)).flatten()
    pound_c_arr = np.array(skipgram_matrix.sum(axis=0)).flatten()

    # for context distribution smoothing (cds)
    pound_c_alpha_arr = pound_c_arr ** ppmi_alpha
    pound_c_alpha_norm = np.sum(pound_c_alpha_arr)

    row_indxs = []
    col_indxs = []
    dat_values = []

    for skipgram in tqdm(
        skipgram_weights.items(),
        total=len(skipgram_weights),
        desc="calculating ppmi matrix",
    ):

        (word, context), pound_wc = skipgram
        word_indx = unigram_to_index[word]
        context_indx = unigram_to_index[context]

        pound_w = pound_w_arr[word_indx]
        pound_c = pound_c_arr[context_indx]
        pound_c_alpha = pound_c_alpha_arr[context_indx]

        # this is how you would write the probabilities
        # Pwc = pound_wc / DD
        # Pw = pound_w / DD
        # Pc = pound_c / DD
        # Pc_alpha = pound_c_alpha / pound_c_alpha_norm

        # its more computationally effecient to use the counts directly
        # its less computationally efficient to also calculate the unsmoothed pmi
        # but we don't want it to feel left out
        pmi = np.log2((pound_wc * DD) / (pound_w * pound_c))
        pmi_alpha = np.log2((pound_wc * pound_c_alpha_norm) / (pound_w * pound_c_alpha))

        # turn pointwise mutual information into positive pointwise mutual information
        ppmi = max(pmi, 0)
        ppmi_alpha = max(pmi_alpha, 0)

        row_indxs.append(word_indx)
        col_indxs.append(context_indx)
        dat_values.append(ppmi_alpha)

    nrows = ncols = len(unigram_to_index)
    return sparse.csr_matrix((dat_values, (row_indxs, col_indxs)), shape=(nrows, ncols))


def calculate_svd_matrix(
    high_dim_matrix: sparse.csr_matrix, embedding_size: int, svd_p: float
) -> np.ndarray:
    """Singular Value Decomposition with eigenvalue weighting.

    See 3.3 of LGD15 https://aclanthology.org/Q15-1016/
    """
    uu, ss, vv = linalg.svds(high_dim_matrix, k=embedding_size)
    svd_matrix = uu.dot(np.diag(ss ** svd_p))
    return svd_matrix


def calculate_sample_vectors(
    sentences: pd.Series, unigram_vecs, unigram_to_index, sample_to_index,
):

    """
    Input:
      * sentences: a pd.Series with lists of (unigram, weight) tuples.
      * unigram_vecs: unigram vecs to combine into sample vecs
      * unigram_to_index: unigram string -> matrix index
      * sample_to_index: sample ID -> matrix index
    """

    num_samples = sentences.size
    embedding_size = unigram_vecs.shape[1]
    sample_vecs = np.zeros((num_samples, embedding_size))
    for sample_id, full_sentence in tqdm(
        sentences.items(), desc="making sample vectors"
    ):
        sentence = [
            (unigram, weight)
            for (unigram, weight) in full_sentence
            if unigram in unigram_to_index
        ]
        sample_vec = np.zeros(embedding_size)
        norm = len(sentence) if len(sentence) > 0 else 1
        for unigram, weight in sentence:
            unigram_index = unigram_to_index[unigram]
            unigram_vec = unigram_vecs[unigram_index]
            sample_vec += unigram_vec
        sample_vec = sample_vec / norm
        sample_index = sample_to_index[sample_id]
        sample_vecs[sample_index, :] = sample_vec
    return sample_vecs


class PpmiEmbeddings:
    def __init__(
        self,
        df_dcs,
        min_unigram_weight,
        unigram_weighter,
        skipgram_weighter,
        embedding_size=100,
        ppmi_alpha=0.75,
        svd_p=1.0,
    ):

        self.df_dcs = df_dcs
        self.min_unigram_weight = min_unigram_weight
        self.unigram_weighter = unigram_weighter
        self.skipgram_weighter = skipgram_weighter
        self.embedding_size = embedding_size
        self.ppmi_alpha = ppmi_alpha
        self.svd_p = svd_p

    def create_embeddings(self, sent_col):

        unigram_weights, skipgram_weights = calculate_grams(
            self.df_dcs[sent_col],
            self.min_unigram_weight,
            self.unigram_weighter,
            self.skipgram_weighter,
        )

        index_to_unigram = dict(enumerate(unigram_weights.keys()))
        unigram_to_index = {unigram: ii for ii, unigram in index_to_unigram.items()}
        skipgram_matrix = create_skipgram_matrix(skipgram_weights, unigram_to_index)

        ppmi_matrix = calculate_ppmi_matrix(
            skipgram_matrix, skipgram_weights, unigram_to_index, self.ppmi_alpha,
        )

        lo_dim = min(self.embedding_size, ppmi_matrix.shape[0] - 1)
        svd_matrix = calculate_svd_matrix(ppmi_matrix, lo_dim, self.svd_p,)

        index_to_sample = dict(enumerate(self.df_dcs.index))
        sample_to_index = {sample_id: ii for ii, sample_id in index_to_sample.items()}

        sample_vecs = calculate_sample_vectors(
            self.df_dcs[sent_col], svd_matrix, unigram_to_index, sample_to_index,
        )

        self.unigram_weights = unigram_weights
        self.skipgram_weights = skipgram_weights
        self.index_to_unigram = index_to_unigram
        self.unigram_to_index = unigram_to_index
        self.skipgram_matrix = skipgram_matrix
        self.ppmi_matrix = ppmi_matrix
        self.svd_matrix = svd_matrix
        self.index_to_sample = index_to_sample
        self.sample_to_index = sample_to_index
        self.sample_vecs = sample_vecs

    def write_unigram_projector_files(self, path, tag, df_meta_extra=None):

        os.makedirs(path, exist_ok=True)
        files_written = []

        # write out gene level embeddings
        # ====================================================================
        if self.ppmi_matrix.shape[0] < MAX_PPMI_NDIM_WRITE:
            fpath = os.path.join(path, f"{tag}_unigram_ppmi_vecs.tsv")
            files_written.append(fpath)
            df_vecs = pd.DataFrame(self.ppmi_matrix.todense())
            df_vecs.to_csv(fpath, sep="\t", index=False, header=False)
        else:
            logger.info(f"skipping PPMI vector write for shape {self.ppmi_matrix.shape}")

        fpath = os.path.join(path, f"{tag}_unigram_svd_{self.embedding_size}_vecs.tsv")
        files_written.append(fpath)
        df_vecs = pd.DataFrame(self.svd_matrix)
        df_vecs.to_csv(fpath, sep="\t", index=False, header=False)

        # write out gene level metadata
        # ====================================================================

        # record unigram names -> index
        df_meta = pd.DataFrame(
            [self.index_to_unigram[ii] for ii in range(len(self.index_to_unigram))],
            columns=["unigram"],
        )

        # record unigram weights
        df_uweight = pd.DataFrame(
            self.unigram_weights.items(), columns=["unigram", "unigram_weight"]
        )
        df_meta = pd.merge(df_meta, df_uweight, on="unigram")

        df_meta["unigram_weight_norm"] = (
            df_meta["unigram_weight"] / df_meta["unigram_weight"].sum()
        )

        if df_meta_extra is not None:
            # add in extra metadata
            df_meta = pd.merge(df_meta, df_meta_extra, on="unigram", how="left")

        fpath = os.path.join(path, f"{tag}_unigram_meta.tsv")
        files_written.append(fpath)
        df_meta.to_csv(fpath, sep="\t", index=False)

        return files_written

    def write_sample_projector_files(self, path, tag, df_dcs):

        files_written = []

        # write out sample level embeddings
        # ====================================================================
        fpath = os.path.join(path, f"{tag}_sample_{self.embedding_size}_vecs.tsv")
        files_written.append(fpath)
        df_vecs = pd.DataFrame(self.sample_vecs)
        df_vecs.to_csv(
            fpath, sep="\t", index=False, header=False,
        )

        # write out sample level metadata
        # ====================================================================

        df_meta = pd.DataFrame(
            [self.index_to_sample[ii] for ii in range(len(self.index_to_sample))],
            columns=["SAMPLE_ID"],
        )

        # reocrd sample metadata from data clinical sample
        df_meta = pd.merge(df_meta, df_dcs, on="SAMPLE_ID", how="left",)

        # record sample metadata from data mutations extended
        df_tmp = df_dcs["mut_sent"].apply(set).to_frame("Hugos").reset_index()

        df_meta = pd.merge(df_meta, df_tmp, on="SAMPLE_ID",)

        df_meta["CENTER"] = df_meta["SAMPLE_ID"].apply(lambda x: x.split("-")[1])
        CENTER_CODES = ["DFCI", "MSK", "UCSF"]
        for center in CENTER_CODES:
            df_meta[f"{center}_flag"] = (df_meta["CENTER"] == center).astype(int)

        HUGO_CODES = ["NF1", "NF2", "SMARCB1", "LZTR1"]
        for hugo in HUGO_CODES:
            df_meta[f"{hugo}_mut"] = (
                df_meta["Hugos"].apply(lambda x: hugo in x).astype(int)
            )

        EXTRA_HUGO_CODES = ["KIT"]
        for hugo in EXTRA_HUGO_CODES:
            df_meta[f"{hugo}_mut"] = (
                df_meta["Hugos"].apply(lambda x: hugo in x).astype(int)
            )

        ONCOTREE_CODES = ["NST", "MPNST", "NFIB", "SCHW", "CSCHW", "MSCHW"]
        for oncotree in ONCOTREE_CODES:
            df_meta[f"{oncotree}_flag"] = (df_meta["ONCOTREE_CODE"] == oncotree).astype(
                int
            )

        df_meta["NST_CANCER_TYPE_FLAG"] = (
            df_meta["ONCOTREE_CODE"].isin(ONCOTREE_CODES)
        ).astype(int)

        EXTRA_ONCOTREE_CODES = ["GIST"]
        for oncotree in EXTRA_ONCOTREE_CODES:
            df_meta[f"{oncotree}_flag"] = (df_meta["ONCOTREE_CODE"] == oncotree).astype(
                int
            )

        to_drop = [
            "Hugos",
            "mut_sent",
            "mut_sent_flat",
            "var_sent_flat",
            "score_sent",
            "mut_sent_score",
            "var_sent_score",
        ]
        for col in to_drop:
            if col in df_meta.columns:
                df_meta = df_meta.drop(col, axis=1)

        fpath = os.path.join(path, f"{tag}_sample_meta.tsv")
        files_written.append(fpath)
        df_meta.to_csv(fpath, sep="\t", index=False)

        return files_written
