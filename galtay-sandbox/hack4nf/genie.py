import pandas as pd


NF_HUGO_SYMBOLS = [
    "NF1",
    "NF2",
]

NF_EXTRA_HUGO_SYMBOLS = [
    "SMARCB1",
    "LZTR1",
]

NF_ONCOTREE_CODES = [
    "NST",
    "MPNST",
    "NFIB",
    "SCHW",
    "CSCHW",
    "MSCHW",
]


def read_clinical_patient(fpath: str) -> pd.DataFrame:
    """Read a data_clinical_patient.txt file"""
    return pd.read_csv(fpath, sep="\t", comment="#")


def read_clinical_sample(fpath: str) -> pd.DataFrame:
    """Read a data_clinical_sample.txt file"""
    return pd.read_csv(fpath, sep="\t", comment="#")


def read_cna_seg(fpath: str) -> pd.DataFrame:
    """Read a genie_data_cna_hf19.seg file"""
    return pd.read_csv(fpath, sep="\t")


def read_cna(fpath: str) -> pd.DataFrame:
    """Read a data_CNA.txt file

    This is discrete copy number data
    This rearranges so that we have
    * N-sample rows
    * N-gene columns
    """
    return pd.read_csv(fpath, sep="\t").set_index("Hugo_Symbol").T.sort_index()


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


def read_pat_sam_mut(pat_fpath: str, sam_fpath: str, mut_fpath: str) -> pd.DataFrame:
    """Read and join the,
    * data_clinical_patient
    * data_clinical_sample
    * data_mutations_extended
    """
    df_pat = read_clinical_patient(pat_fpath)
    df_sam = read_clinical_sample(sam_fpath)
    df_mut = read_mutations_extended(mut_fpath)

    df_mut = pd.merge(
        df_mut,
        df_sam,
        left_on="Tumor_Sample_Barcode",
        right_on="SAMPLE_ID",
    )
    df_mut = pd.merge(
        df_mut,
        df_pat,
        on="PATIENT_ID",
    )

    return df_mut
