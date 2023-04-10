"""
Tools for analyzing "Drug Screenging of pNF Cell Lines"
Single Agent Screens
https://www.synapse.org/#!Synapse:syn5522627

More background can be found here,
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5996849


files included in the dataset above are,

.
├── aucRecalculatedFromNCATSscreens_nplr.txt
├── ipNF95_11bC_after_correction.png
├── ipNF95_11bC_before_correction.png
├── matrix portal raw data
│   ├── s-ntap-1PNO-1
│   ├── s-ntap-HFF-1
│   ├── s-ntap-ipn02.8-1
│   ├── s-ntap-IPNF05_5_P30-1
│   ├── s-ntap-IPNF05_5_P31-1
│   ├── s-ntap-IPNF06_2A_P27-1
│   ├── s-ntap-IPNF95_11b_P53-1
│   ├── s-ntap-IPNF95-1bc-1
│   ├── s-ntap-IPNF95_6_P53-1
│   ├── s-ntap-ipnNF95_11C_P31-1
│   ├── SYNAPSE_METADATA_MANIFEST.tsv
│   └── synapse_storage_manifest.csv
├── NTAP ipNF02.3 2l MIPE qHTS.csv
├── NTAP ipNF02.8 MIPE qHTS.csv
├── NTAP ipNF05.5 MC MIPE qHTS.csv
├── NTAP ipNF05.5 SC MIPE qHTS.csv
├── NTAP ipNF06.2A MIPE qHTS.csv
├── NTAP_ipNF95.11bC_MIPE_qHTS.csv
├── NTAP ipNF95.11b C_T MIPE qHTS.csv
├── NTAP ipNF95.6 MIPE qHTS.csv
├── NTAP ipnNF95.11C MIPE qHTS.csv
├── qhts-protocol-dump-headers.txt
├── s-ntap-HFF-1.csv
├── s-ntap-MTC-1.csv
└── SYNAPSE_METADATA_MANIFEST.tsv

"""
from pathlib import Path
import re
from typing import Dict, List, Iterable

import numpy as np
import pandas as pd


def read_manifest(data_path, filter=True):
    """Read data manifest for all files.

    if filter then
      * keep only dose-response curve files
    """

    df = pd.read_csv(
        data_path / "SYNAPSE_METADATA_MANIFEST.tsv",
        sep="\t",
    )
    if filter:
        df = df[df["name"].apply(lambda x: "ntap" in x.lower())].reset_index(drop=True)

    return df.sort_values("specimenID")


COL_RENAMES = {
    "Hill": "HILL",
    "Cell Line": "Cell line",
    "protocol": "NCGC protocol",
    "NCCGC protocol": "NCGC protocol",
    "SID": "NCGC SID",
    "Infinity": "INF",
    "Zero": "ZERO",
    "NCGCID": "NCGC SID",
    "Name": "name",
    "Target": "target",
    "smi": "SMILES",
}
for ii in range(0, 11):
    COL_RENAMES[f"C{ii}"] = f"CONC{ii}"

KEEP_COLS = (
    [
        "NCGC protocol",
        "NCGC SID",
        "Cell line",
        "LAC50",
        "AC50",
        "HILL",
        "INF",
        "ZERO",
        "MAXR",
        "FAUC",
        "TAUC",
        "R2",
        "PHILL",
        "name",
        "target",
        "SMILES",
        "MoA",
        "source_file",
        "source_dir",
    ]
    + [f"DATA{i}" for i in range(0, 11)]
    + [f"CONC{i}" for i in range(0, 11)]
)


def get_concentration_multipliers(file_base):
    """Set multipliers to turn concentrations into micromolars"""

    # AC50 is in molar units
    # concentration column is in micromolars
    group_1 = [
        "NTAP ipNF02.3 2l MIPE qHTS.csv",
        "NTAP ipNF05.5 MC MIPE qHTS.csv",
        "NTAP ipNF06.2A MIPE qHTS.csv",
        "NTAP ipnNF95.11C MIPE qHTS.csv",
        "NTAP ipNF02.8 MIPE qHTS.csv",
        "NTAP ipNF95.11b C_T MIPE qHTS.csv",
        "NTAP ipNF05.5 SC MIPE qHTS.csv",
        "NTAP ipNF95.6 MIPE qHTS.csv",
        "s-ntap-HFF-1.csv",
        "s-ntap-MTC-1.csv",
    ]

    # AC50 is in micromolars
    # concentration is in molar units
    group_2 = [
        "NTAP_ipNF95.11bC_MIPE_qHTS.csv",
        "s-ntap-IPNF95_11b_P53-1",
        "s-ntap-HFF-1",
        "s-ntap-IPNF06_2A_P27-1",
        "s-ntap-IPNF95-1bc-1",
        "s-ntap-1PNO-1",
        "s-ntap-IPNF95_6_P53-1",
        "s-ntap-IPNF05_5_P30-1",
        "s-ntap-IPNF05_5_P31-1",
        "s-ntap-ipn02.8-1",
        "s-ntap-ipnNF95_11C_P31-1",
    ]

    if file_base in group_1:
        ac50_mult = 1e6
        c_mult = 1.0
    elif file_base in group_2:
        ac50_mult = 1.0
        c_mult = 1e6
    else:
        raise ValueError()

    return ac50_mult, c_mult


def set_conc(file_base, df):
    ac50_mult, c_mult = get_concentration_multipliers(file_base)

    if "AC50" in df.columns:
        df["AC50"] = df["AC50"].values * ac50_mult

    if "LAC50" in df.columns:
        df["LAC50"] = df["LAC50"].values + np.log10(ac50_mult)

    if "AC50" in df.columns and "LAC50" not in df.columns:
        df["LAC50"] = np.log10(df["AC50"])

    if "LAC50" in df.columns and "AC50" not in df.columns:
        df["AC50"] = 10 ** df["LAC50"]

    conc_cols = [f"CONC{i}" for i in range(0, 11)]
    for ii, col in enumerate(conc_cols):
        df[col] = df[col].values * c_mult

    return df


def read_matrix_portal_raw_hts(data_path):
    """Read matrix portal raw data dose response curve files"""

    file_bases = [
        "s-ntap-1PNO-1",
        "s-ntap-HFF-1",
        "s-ntap-ipn02.8-1",
        "s-ntap-IPNF05_5_P30-1",
        "s-ntap-IPNF05_5_P31-1",
        "s-ntap-IPNF06_2A_P27-1",
        "s-ntap-IPNF95_11b_P53-1",
        "s-ntap-IPNF95-1bc-1",
        "s-ntap-IPNF95_6_P53-1",
        "s-ntap-ipnNF95_11C_P31-1",
    ]

    dfs = {}
    for file_base in file_bases:
        file_path = data_path / "matrix portal raw data" / file_base
        df = pd.read_csv(file_path)
        df = df.rename(columns=COL_RENAMES)
        df["source_file"] = file_base
        df["source_dir"] = "matrix_portal_raw_data"
        df = set_conc(file_base, df)

        if "NCGC protocol" not in df.columns:
            df["NCGC protocol"] = file_base

        for col in ["Cell line", "FAUC", "TAUC", "MoA", "R2", "PHILL"]:
            if col not in df.columns:
                df[col] = pd.NA

        for keep_col in KEEP_COLS:
            if keep_col not in df.columns:
                print(file_base)
                print(keep_col)
                print(df.columns)
                raise ValueError()

        df = df[KEEP_COLS]
        dfs[file_base] = df

    return dfs


def read_top_level_hts(data_path):
    """Read top level dose response curve files"""

    file_bases = [
        "NTAP ipNF02.3 2l MIPE qHTS.csv",
        "NTAP ipNF02.8 MIPE qHTS.csv",
        "NTAP ipNF05.5 MC MIPE qHTS.csv",
        "NTAP ipNF05.5 SC MIPE qHTS.csv",
        "NTAP ipNF06.2A MIPE qHTS.csv",
        "NTAP_ipNF95.11bC_MIPE_qHTS.csv",
        "NTAP ipNF95.11b C_T MIPE qHTS.csv",
        "NTAP ipNF95.6 MIPE qHTS.csv",
        "NTAP ipnNF95.11C MIPE qHTS.csv",
        "s-ntap-HFF-1.csv",
        #        's-ntap-MTC-1.csv',
    ]

    dfs = {}
    for file_base in file_bases:
        file_path = data_path / file_base
        df = pd.read_csv(file_path)
        df = df.rename(columns=COL_RENAMES)
        df["source_file"] = file_base
        df["source_dir"] = "top"
        df = set_conc(file_base, df)

        if file_base == "s-ntap-HFF-1.csv":
            df["Cell line"] = "HFF"

        if file_base == "NTAP_ipNF95.11bC_MIPE_qHTS.csv":
            df["NCGC protocol"] = "s-ntap-IPNF95-1bc-1"
            df["Cell line"] = "ipNF95.11bC"

        for col in ["MoA", "TAUC", "FAUC", "R2", "PHILL"]:
            if col not in df.columns:
                df[col] = pd.NA

        for keep_col in KEEP_COLS:
            if keep_col not in df.columns:
                print(file_base)
                print(keep_col)
                print(df.columns)
                raise ValueError()

        df = df[KEEP_COLS]
        dfs[file_base] = df

    return dfs


if __name__ == "__main__":
    data_path = Path("/home/galtay/data/hack4nf-2022/synapse/syn5522627")

    df_manifest = read_manifest(data_path)
    df_manifest_sml = df_manifest[
        ["name", "specimenID", "individualID", "nf1Genotype", "isPrimaryCell"]
    ]

    dfs_hts_top = read_top_level_hts(data_path)
    dfs_hts_mprd = read_matrix_portal_raw_hts(data_path)

    df_top = pd.concat(dfs_hts_top.values())
    df_mprd = pd.concat(dfs_hts_mprd.values())

    df = pd.concat([df_top, df_mprd])

    # target/MoA is only in some source files but its a 1-1 mapping
    # so lets take it from one cell line and add it everywhere
    df_rep = df_mprd[df_mprd["NCGC protocol"] == "s-ntap-1PNO-1"].copy()
    for col in ["target", "MoA"]:
        mapper = df_rep.set_index("NCGC SID")[col].to_dict()
        df[col] = df["NCGC SID"].apply(lambda x: mapper[x])

    # set Cell line from NCGC protocol where its missing
    mapper = (
        df_top[['NCGC protocol', 'Cell line']]
        .drop_duplicates()
        .set_index("NCGC protocol")['Cell line']
        .to_dict()
    )
    df["Cell line"] = df["NCGC protocol"].apply(lambda x: mapper[x])

    # preserve the original row number
    df = df.reset_index()
    df = df.rename(columns={"index": "iorig_row"})

    df_manifest.to_csv("syn5522627-manifest.csv")
    df.to_csv("syn5522627-clean.csv", index=False)
