import pandas as pd


def read_clinical_patient(fpath):
    return pd.read_csv(fpath, sep="\t", comment="#")

def read_clinical_sample(fpath):
    return pd.read_csv(fpath, sep="\t", comment="#")

def read_cna(fpath):
    return pd.read_csv(fpath, sep="\t")

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
