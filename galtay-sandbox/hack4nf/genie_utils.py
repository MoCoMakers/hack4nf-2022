import pandas as pd


SYNIDS = {
    "genie_12.0": {
        "data_clinical_patient": "syn32689054",
        "data_clinical_sample": "syn32689057",
        "data_mutations_extended": "syn32689317",
        "data_CNA": "syn32689019",
        "data_cna_hg19_seg": "syn32689379",
    },
    "genie_13.3": {
        "data_clinical_patient": "syn36710136",
        "data_clinical_sample": "syn36710137",
        "data_mutations_extended": "syn36710142",
        "data_CNA": "syn36710134",
        "data_cna_hg19_seg": "syn36710143",
    },
}


def read_clinical_patient(fpath):
    return pd.read_csv(fpath, sep="\t", comment="#")

def read_clinical_sample(fpath):
    return pd.read_csv(fpath, sep="\t", comment="#")

def read_cna_seg(fpath):
    return pd.read_csv(fpath, sep="\t")

def read_cna(fpath):
    """Read discrete copy number data

    This rearranges so that we have
    * Nsamp rows
    * Ngene columns
    """
    return pd.read_csv(fpath, sep="\t").set_index('Hugo_Symbol').T.sort_index()


def get_cna_norms(df_cna, axis, k=1, p=2):
    """Vector Norms [sum(|c_i,j|^p)]**(k/p)

    k=1,p=2 is L2 norm
    axis=0 will do gene vectors
    axis=1 will do sample vectors

    TODO: do better imputation than just setting to 0
    """
    ser = ((df_cna.fillna(0).abs()**p).sum(axis=axis)**(k/p))
    return ser


def get_melted_cna(df_cna, drop_nan=True, drop_zero=True):
    df = df_cna.copy()
    df['SAMPLE_ID'] = df.index
    df = df.reset_index(drop=True)
    mdf = pd.melt(df, id_vars='SAMPLE_ID', var_name='hugo', value_name='dcna')
    if drop_nan:
        mdf = mdf[~mdf['dcna'].isnull()]
    if drop_zero:
        mdf = mdf[mdf['dcna']!=0]
    return mdf


def read_mutations_extended(fpath):
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

def dme_to_cravat(df):
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
        (df['Reference_Allele'] != df['Tumor_Seq_Allele1'])
        & (~df['Tumor_Seq_Allele1'].isnull())
        & (df['Tumor_Seq_Allele1'] != df['Tumor_Seq_Allele2'])
    )
    df_cravat["ALT"] = df['Tumor_Seq_Allele2']
    df.loc[bmask, "ALT"] = df['Tumor_Seq_Allele1']

    return df_cravat
