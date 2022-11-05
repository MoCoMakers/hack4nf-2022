from glob import glob
import os
import pandas as pd

from loguru import logger

from nextgenlp import genie_constants


class GenieData:

    def __init__(self, file_paths):
        self.file_paths = file_paths

    def load(self):
        self.df_gp_tall = read_gene_panels(self.file_paths["gene_panels"], style="tall")
        self.df_gp_wide = read_gene_panels(self.file_paths["gene_panels"], style="wide")
        # this has one row pre mutation
        self.df_mut_all = read_pat_sam_mut(
            self.file_paths["data_clinical_patient"],
            self.file_paths["data_clinical_sample"],
            self.file_paths["data_mutations_extended"],
        )
        # remove rows that don't have mutations
        self.df_dcs_all = read_clinical_sample(self.file_paths["data_clinical_sample"])
        self.df_dcs_all = self.df_dcs_all.loc[set(self.df_mut_all["SAMPLE_ID"])]

        logger.info("all variant rows: {}".format(self.df_mut_all.shape[0]))
        logger.info("all sample rows: {}".format(self.df_dcs_all.shape[0]))

    def impute(self):
        self.df_mut_all["HGVSp_Short"] = self.df_mut_all["HGVSp_Short"].fillna("")
        for col in ["Polyphen_Score", "SIFT_Score"]:
            self.df_mut_all[col] = self.df_mut_all[col].fillna(
                self.df_mut_all[col].mean()
            )
        self.df_mut_all["score"] = (
            self.df_mut_all["Polyphen_Score"] + self.df_mut_all["SIFT_Score"]
        ) / 2
        self.df_mut_all["var_token"] = (
            self.df_mut_all["Hugo_Symbol"] + "<>" + self.df_mut_all["HGVSp_Short"]
        )

    def subset(self, subset, y_col=None, y_min_count=None):

        self.subset = subset
        self.panels = genie_constants.SEQ_ASSAY_ID_GROUPS[subset]

        if subset == "ALL":
            self.df_dcs = self.df_dcs_all.copy()
            self.subset_sample_ids = set(self.df_dcs.index)
            self.subset_genes = set(self.df_gp_wide.columns)
        else:
            (
                self.subset_sample_ids,
                self.subset_genes,
            ) = get_genes_and_samples_from_seq_assay_ids(
                self.df_gp_wide, self.df_dcs_all, self.panels
            )
            self.df_dcs = self.df_dcs_all.loc[self.subset_sample_ids].copy()

        self.df_mut = self.df_mut_all[
            self.df_mut_all["SAMPLE_ID"].isin(self.df_dcs.index)
        ].copy()
        self.df_mut = self.df_mut[self.df_mut["Hugo_Symbol"].isin(self.subset_genes)]
        self.df_dcs = self.df_dcs.loc[self.df_mut["SAMPLE_ID"].unique()]

        logger.info(
            "subset {} has {} genes {} variant rows and {} sample rows".format(
                subset,
                len(self.subset_genes),
                self.df_mut.shape[0],
                self.df_dcs.shape[0],
            )
        )

        if y_col is not None:
            y_counts = self.df_dcs[y_col].value_counts()
            y_keep = y_counts[y_counts >= y_min_count].index
            self.df_dcs = self.df_dcs[self.df_dcs[y_col].isin(y_keep)]
            self.df_mut = self.df_mut[self.df_mut["SAMPLE_ID"].isin(self.df_dcs.index)]

            logger.info(
                "after filtering {} with min count {}, subset has {} variant rows and {} sample rows".format(
                    y_col, y_min_count, self.df_mut.shape[0], self.df_dcs.shape[0]
                )
            )

        self.df_dcs["mut_sent"] = get_mut_gene_sentences(self.df_mut, with_weights=False)
        self.df_dcs["var_sent"] = self.df_mut.groupby("SAMPLE_ID")["var_token"].apply(list)
        self.df_dcs["score_sent"] = self.df_mut.groupby("SAMPLE_ID")["score"].apply(list)


def read_gene_panel(gp_path: str) -> pd.DataFrame:
    """Read one data_gene_panel_<SEQ_ASSAY_ID>.txt file"""
    with open(gp_path, "r") as fp:
        lines = fp.readlines()
    gene_panel = lines[0].strip().split("stable_id:")[-1].strip()
    num_genes = int(lines[1].strip().split("Number of Genes - ")[-1])
    genes = lines[2].strip().split("\t")[1:]
    assert num_genes == len(genes)
    df = pd.DataFrame(genes, columns=["Hugo_Symbol"])
    df["SEQ_ASSAY_ID"] = gene_panel
    return df


def read_gene_panels(gp_path: str, style="wide") -> pd.DataFrame:
    """Read all data_gene_panel_<SEQ_ASSAY_ID>.txt files"""
    fpaths = glob(os.path.join(gp_path, "data_gene_panel*.txt"))
    dfs = [read_gene_panel(fpath) for fpath in fpaths]
    df = pd.concat(dfs).reset_index(drop=True)
    if style == "tall":
        return df
    elif style == "wide":
        df["value"] = 1
        df = (
            df.pivot(index="SEQ_ASSAY_ID", columns="Hugo_Symbol")["value"]
            .fillna(0)
            .astype(int)
        )
        return df
    else:
        raise ValueError(f"style must be 'wide' or 'tall', got {style}")


def read_data_gene_matrix(fpath: str) -> pd.DataFrame:
    """Read data_gene_matrix.txt file"""
    df = pd.read_csv(fpath, sep="\t")
    df = df.set_index("SAMPLE_ID", verify_integrity=True)
    return df


def read_genomic_information(fpath: str) -> pd.DataFrame:
    """Read genomic_information.txt file"""
    df = pd.read_csv(fpath, sep="\t")
    return df


def read_assay_information(fpath: str) -> pd.DataFrame:
    """Read assay_information.txt file"""
    df = pd.read_csv(fpath, sep="\t")
    df = df.set_index("SEQ_ASSAY_ID", verify_integrity=True)
    return df


def read_data_fusions(fpath: str) -> pd.DataFrame:
    """Read data_fusions.txt file"""
    df = pd.read_csv(fpath, sep="\t")
    return df


def read_clinical_patient(fpath: str) -> pd.DataFrame:
    """Read data_clinical_patient.txt file"""
    df = pd.read_csv(fpath, sep="\t", comment="#")
    df = df.set_index("PATIENT_ID", verify_integrity=True)
    return df


def read_clinical_sample(fpath: str) -> pd.DataFrame:
    """Read data_clinical_sample.txt file"""
    df = pd.read_csv(fpath, sep="\t", comment="#")
    df["CENTER"] = df["SAMPLE_ID"].apply(lambda x: x.split("-")[1])
    df = df.set_index("SAMPLE_ID", verify_integrity=True)
    return df


def read_cna_seg(fpath: str) -> pd.DataFrame:
    """Read genie_data_cna_hf19.seg file"""
    return pd.read_csv(fpath, sep="\t")


def read_cna(fpath: str) -> pd.DataFrame:
    """Read data_CNA.txt file

    This is discrete copy number data
    This rearranges so that we have
    * N-sample rows
    * N-gene columns
    """
    df = pd.read_csv(fpath, sep="\t").set_index("Hugo_Symbol").T.sort_index()
    df.index.name = "SAMPLE_ID"
    return df


def get_cna_norms(df_cna: pd.DataFrame, axis: int, k: int = 1, p: int = 2) -> pd.Series:
    """Vector Norms [sum(|c_i,j|^p)]**(k/p)

    k=1,p=2 is L2 norm
    axis=0 will do gene vectors
    axis=1 will do sample vectors

    this was the mapper lens used in
    https://www.pnas.org/doi/10.1073/pnas.1102826108

    TODO: do better imputation than just setting to 0
    """
    ser = (df_cna.fillna(0).abs() ** p).sum(axis=axis) ** (k / p)
    return ser


def get_melted_cna(
    df_cna: pd.DataFrame, drop_nan: bool = True, drop_zero: bool = True
) -> pd.DataFrame:
    """Melt discrete copy number data

    This transforms an N-sample by N-gene CNA dataframe into a dataframe that has
    one row for each cell in the original matrix.
    """
    df = df_cna.copy()
    df["SAMPLE_ID"] = df.index
    df = df.reset_index(drop=True)
    df_melted = pd.melt(df, id_vars="SAMPLE_ID", var_name="hugo", value_name="dcna")
    if drop_nan:
        df_melted = df_melted[~df_melted["dcna"].isnull()]
    if drop_zero:
        df_melted = df_melted[df_melted["dcna"] != 0]
    return df_melted


def read_mutations_extended(fpath: str) -> pd.DataFrame:
    """Read a data_mutations_extended.txt file"""
    return pd.read_csv(
        fpath,
        dtype={
            "Entrez_Gene_Id": pd.Int64Dtype(),
            "Chromosome": str,
            "Reference_Allele": str,
            "Tumor_Seq_Allele1": str,
            "Tumor_Seq_Allele2": str,
            "Match_Norm_Seq_Allele1": str,
            "Match_Norm_Seq_Allele2": str,
            "Matched_Norm_Sample_Barcode": str,
            "FILTER": str,
        },
        sep="\t",
    )


def dme_to_cravat(df: pd.DataFrame) -> pd.DataFrame:
    """Create Open Cravat dataframe from data_mutations_extended dataframe

    NOTE: some inspiration from civicpy
    https://github.com/griffithlab/civicpy/blob/master/examples/Project%20GENIE.ipynb
    """
    df_cravat = pd.DataFrame()
    df_cravat["CHROM"] = df["Chromosome"].apply(lambda x: f"chr{x}")
    df_cravat["POS"] = df["Start_Position"]
    df_cravat["STRAND"] = df["Strand"]
    df_cravat["REF"] = df["Reference_Allele"]
    df_cravat["ALT"] = df["Tumor_Seq_Allele2"]
    df_cravat["INDIVIDUAL"] = df["Tumor_Sample_Barcode"]

    # decide if we should use Allele1 or Allele2 for ALT
    # turns out bmask is never true in the data i have looked at
    bmask = (
        (df["Reference_Allele"] != df["Tumor_Seq_Allele1"])
        & (~df["Tumor_Seq_Allele1"].isnull())
        & (df["Tumor_Seq_Allele1"] != df["Tumor_Seq_Allele2"])
    )
    df_cravat["ALT"] = df["Tumor_Seq_Allele2"]
    df_cravat.loc[bmask, "ALT"] = df["Tumor_Seq_Allele1"]

    return df_cravat


def read_pat_sam_mut(
    patient_fpath: str, sample_fpath: str, mutations_fpath: str
) -> pd.DataFrame:
    """Read and join the,
    * data_clinical_patient
    * data_clinical_sample
    * data_mutations_extended
    """
    df_dcp = read_clinical_patient(patient_fpath)
    # reset index so that "SAMPLE_ID" will be included in final columns
    # drop "CENTER" b/c it's in df_dme
    df_dcs = read_clinical_sample(sample_fpath).reset_index().drop(columns=["CENTER"])
    df_dme = read_mutations_extended(mutations_fpath)

    df_psm = pd.merge(
        df_dme,
        df_dcs,
        left_on="Tumor_Sample_Barcode",
        right_on="SAMPLE_ID",
    )

    df_psm = pd.merge(
        df_psm,
        df_dcp,
        on="PATIENT_ID",
    )

    return df_psm


def get_mut_gene_sentences(df_mut: pd.DataFrame, with_weights=True) -> pd.Series:
    sentences = df_mut.groupby("SAMPLE_ID")["Hugo_Symbol"].apply(list)
    if with_weights:
        sentences.apply(lambda x: [(el, 1.0) for el in x])
    return sentences


def get_cna_sentences(df_cna: pd.DataFrame, drop_nan=True, drop_zero=True) -> pd.Series:
    df_cna_melted = get_melted_cna(df_cna, drop_nan=drop_nan, drop_zero=drop_zero)
    cna_sentences = df_cna_melted.groupby("SAMPLE_ID").apply(
        lambda x: list(zip(x["hugo"], x["dcna"]))
    )
    return cna_sentences


def filter_sentences_by_gene(sentences, keep_genes):
    return sentences.apply(
        lambda x: [(gene, weight) for gene, weight in x if gene in keep_genes]
    )


def get_genes_and_samples_from_seq_assay_ids(df_gp_wide, df_dcs, seq_assay_ids):
    sample_ids = set()
    genes = set(df_gp_wide.columns)
    for seq_assay_id in seq_assay_ids:
        seq_assay_id_genes = set(
            [gene for (gene, flag) in df_gp_wide.loc[seq_assay_id].items() if flag == 1]
        )
        seq_assay_id_sample_ids = set(
            df_dcs[df_dcs["SEQ_ASSAY_ID"] == seq_assay_id].index
        )
        genes = genes & seq_assay_id_genes
        sample_ids.update(seq_assay_id_sample_ids)
    return sample_ids, genes
