"""
Tools for analyzing "Drug Screenging of pNF Cell Lines"
https://www.synapse.org/#!Synapse:syn5522627

files included in the dataset above are,

 *  aucRecalculatedFromNCATSscreens_nplr.txt
 *  ipNF95_11bC_after_correction.png
 *  ipNF95_11bC_before_correction.png
 * 'matrix portal raw data'
 * 'NTAP ipNF02.3 2l MIPE qHTS.csv'
 * 'NTAP ipNF02.8 MIPE qHTS.csv'
 * 'NTAP ipNF05.5 MC MIPE qHTS.csv'
 * 'NTAP ipNF05.5 SC MIPE qHTS.csv'
 * 'NTAP ipNF06.2A MIPE qHTS.csv'
 *  NTAP_ipNF95.11bC_MIPE_qHTS.csv
 * 'NTAP ipNF95.11b C_T MIPE qHTS.csv'
 * 'NTAP ipNF95.6 MIPE qHTS.csv'
 * 'NTAP ipnNF95.11C MIPE qHTS.csv'
 *  qhts-protocol-dump-headers.txt
 *  s-ntap-HFF-1.csv
 *  s-ntap-MTC-1.csv
 *  SYNAPSE_METADATA_MANIFEST.tsv


"""
from pathlib import Path
import toml
from typing import Dict, List, Iterable
import numpy as np
import pandas as pd


FNAME_TO_CELL_LINE = {
    "NTAP ipNF02.3 2l MIPE qHTS.csv": "ipNF02.3 2l",
    "NTAP ipNF02.8 MIPE qHTS.csv": "ipNF02.8",
    "NTAP ipNF05.5 MC MIPE qHTS.csv": "ipNF05.5 Mixed Clones",
    "NTAP ipNF05.5 SC MIPE qHTS.csv": "ipNF05.5 Single Clone",
    "NTAP ipNF06.2A MIPE qHTS.csv": "ipNF06.2A",
    "NTAP ipNF95.11b C_T MIPE qHTS.csv": "ipNF95.11b C/T",
    "NTAP ipNF95.6 MIPE qHTS.csv": "ipNF95.6",
    "NTAP ipnNF95.11C MIPE qHTS.csv": "ipnNF95.11C",
    "NTAP_ipNF95.11bC_MIPE_qHTS.csv": "ipnNF95.11bC",
    "s-ntap-HFF-1.csv": "HFF-1",
    "s-ntap-MTC-1.csv": "MTC-1",
}


CELL_LINE_TO_FNAME = {val: key for key, val in FNAME_TO_CELL_LINE.items()}


COL_RENAMES = {
    "protocol": "NCGC protocol",
    "NCCGC protocol": "NCGC protocol",
    "SID": "NCGC SID",
    "Cell Line": "Cell line",
    "NCGCID": "NCGC SID",
    "SMILES": "smi",
    "Name": "name",
    "Target": "target",

}
for i in range(11):
    COL_RENAMES[f"CONC{i}"] = f"C{i}"


KEEP_COLS = [
    "NCGC protocol", "Cell line", "AC50", "LAC50", "HILL", "INF",
    "ZERO", "MAXR", "FAUC", "TAUC", "R2", "PHILL", "NPT", "DATA0", "DATA1",
    "DATA2", "DATA3", "DATA4", "DATA5", "DATA6", "DATA7", "DATA8", "DATA9",
    "DATA10", "C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9",
    "C10", "name", "target", "smi"]

SHOW_COLS = [
    "NCGC protocol",
    "Cell line",
    "LAC50",
    "AC50",
    "R2",
    "name",
    "target",
]


def get_hts_file_paths(
    base_path: Path,
    screen_cell_lines: List[str],
    norm_cell_lines: List[str],
) -> List[Path]:
    return [
        base_path / CELL_LINE_TO_FNAME[cell_line]
        for cell_line in screen_cell_lines + norm_cell_lines
    ]


def read_hts_files(file_paths: Iterable[Path]) -> Dict[str, pd.DataFrame]:
    dfs = {
        FNAME_TO_CELL_LINE[file_path.name]: pd.read_csv(file_path)
        for file_path in file_paths
    }
    for cell_line in dfs.keys():
        df = dfs[cell_line]
        df = df.rename(columns=COL_RENAMES)
        df = df.set_index("NCGC SID", verify_integrity=True).sort_index()
        if "LAC50" in df.columns and "AC50" not in df.columns:
            df["AC50"] = 10**df["LAC50"]
        if "AC50" in df.columns and "LAC50" not in df.columns:
            df["LAC50"] = np.log10(df["AC50"])
        df["Cell line"] = cell_line
        df = df[KEEP_COLS]
        dfs[cell_line] = df
    return dfs


def calculate_missingness(dfs):
    missing_rows = []
    for cell_line in dfs.keys():
        df = dfs[cell_line]
        for col in ["AC50", "LAC50", "R2", "target"]:
            num_missing = df[col].isnull().sum()
            frac_missing = num_missing / df.shape[0]
            missing_row = {
                "cell_line": cell_line,
                "variable": col,
                "num_missing": num_missing,
                "frac_missing": frac_missing,
            }
            missing_rows.append(missing_row)
    return pd.DataFrame(missing_rows)


def calculate_ac50_ratios(dfs: Dict[str, pd.DataFrame], norm_cell_lines: Iterable[str]):
    for norm_cell_line in norm_cell_lines:
        df_norm = dfs[norm_cell_line]
        for cell_line in dfs.keys():
            df = dfs[cell_line]
            df[f'resis_{norm_cell_line}'] = df['AC50'] / df_norm['AC50']
            df[f'sensi_{norm_cell_line}'] = df_norm['AC50'] / df['AC50']
            df[f'resis_log_{norm_cell_line}'] = np.log10(df['AC50'] / df_norm['AC50'])
            df[f'sensi_log_{norm_cell_line}'] = np.log10(df_norm['AC50'] / df['AC50'])
            dfs[cell_line] = df
    return dfs


def get_unique_drugs(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat([
        dfs[cl][["name", "target"]]
        for cl in dfs.keys()
    ]).drop_duplicates().sort_index()


def filter_by_ratio(
    df: pd.DataFrame,
    screen_cell_lines: Iterable[str],
    norm_cell_lines: Iterable[str],
    quantity: str,
    ac50_ratio_min: float,
    num_cell_lines_min: int,
):

    df_reports = {}
    VALID_QUANTITY = ["resis", "sensi"]
    if quantity not in VALID_QUANTITY:
        raise ValueError(f"quantity must be one of {VALID_QUANTITY}")

    ac50_log_ratio_min = np.log10(ac50_ratio_min)

    for norm_cell_line in norm_cell_lines:

        col_check = f'{quantity}_log_{norm_cell_line}'
        col_extra = f'{quantity}_{norm_cell_line}'
        report_rows = []

        for ncgc_sid, df1 in df.groupby('NCGC SID'):

            df1_check = df1[df1['Cell line'].isin(screen_cell_lines)]

            num_lines_above_thresh = (df1_check[col_check].abs() >= ac50_log_ratio_min).sum()
            if num_lines_above_thresh >= num_cell_lines_min:
                report_row = {
                    "NCGC SID": ncgc_sid,
                    "name": df1.iloc[0]['name'],
                    "target": df1.iloc[0]['target'],
                    "num_lines_above_thresh": num_lines_above_thresh,
                }
                for cell_line in screen_cell_lines:
                    df_tmp = df1_check[df1_check['Cell line'] == cell_line]
                    if df_tmp.shape[0] == 1:
                        report_row[f"log {cell_line}"] = df_tmp.iloc[0][col_check]
                        report_row[cell_line] = df_tmp.iloc[0][col_extra]
                    else:
                        report_row[f"log {cell_line}"] = np.nan
                        report_row[cell_line] = np.nan
                report_rows.append(report_row)

        df_report = pd.DataFrame(report_rows)
        df_reports[f'{quantity}_{norm_cell_line}'] = df_report

    return df_reports


def main(config):

    base_path = Path(config["paths"]["base"])
    hts_file_paths = get_hts_file_paths(
        base_path,
        config["cell_lines"]["screen"],
        config["cell_lines"]["norm"],
    )
    dfs = read_hts_files(hts_file_paths)
    df_missing = calculate_missingness(dfs)

    df_drugs = get_unique_drugs(dfs)

    dfs = calculate_ac50_ratios(dfs, config["cell_lines"]["norm"])

    show_cols = SHOW_COLS + [
        col for col in dfs[config["cell_lines"]["norm"][0]].columns
        if col.startswith('sensi') or col.startswith('resis')
    ]
    df_all = pd.concat(dfs.values())
    df = df_all[df_all["R2"] > config["params"]["r2_min"]]

    df_reports_sensi = filter_by_ratio(
        df,
        config["cell_lines"]["screen"],
        config["cell_lines"]["norm"],
        "sensi",
        config["params"]["ac50_ratio_min"],
        config["params"]["num_cell_lines_min"]
    )

    df_reports_resis = filter_by_ratio(
        df,
        config["cell_lines"]["screen"],
        config["cell_lines"]["norm"],
        "resis",
        config["params"]["ac50_ratio_min"],
        config["params"]["num_cell_lines_min"]
    )

    df_reports = {**df_reports_resis, **df_reports_sensi}

    df_report = pd.DataFrame()
    for key, df in df_reports.items():
        quantity, norm_cell_line = key.split("_")
        df["quantity"] = quantity
        df["norm_cell_line"] = norm_cell_line
        df_report = pd.concat([df_report, df])

    return df_drugs, df_missing, df_all, df_report


if __name__ == "__main__":

    config_file_path = "sas_config.toml"
    with open(config_file_path, "r") as fp:
        config = toml.load(fp)

    df_drugs, df_missing, df_all, df_report = main(config)
    df_drugs.to_csv("sas_drugs_all.csv")
    df_missing.to_csv("sas_missing.csv")
    df_all.to_csv("sas_all.csv")
    df_report.to_csv("sas_report.csv")
