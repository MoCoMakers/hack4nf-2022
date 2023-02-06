from pathlib import Path
import re

import funcy
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import toml


st.set_page_config(layout="wide")


R_COLS = [f"DATA{ii}" for ii in range(11)]
C_COLS = [f"CONC{ii}" for ii in range(11)]


VAR_DEFS = {
    "protocol": "internal protocol name",
    "NCGC SID":  "NCGC cell line ID",
    "CRC": "curve class heuristic",
    "LAC50": "log AC50 (in molar units)",
    "HILL": "hill slope from curve fit",
    "INF": "asymptote of curve at max concentration",
    "ZERO": "asymptote of curve at zero concentation",
    "MAXR":  "response at max concentration",
    "NPT": "number of points in the curve",
    "FAUC": "Area under the curve (computed from the fit)",
    "TAUC": "Area under the curve (computed from response points)",
    "name": "compound name",
    "target": "primary target",
    "smi": "smiles",
}

# these a pairs of files that represent the same cell line
# the top member is from the top level of the synapse dataset
# the bottom member is from the matrix portal raw data folder
SMM_NAME_GROUPS = [
    (
        "NTAP ipNF02.3 2l MIPE qHTS.csv",
        "s-ntap-1PNO-1",
    ),
    (
        "NTAP_ipNF95.11bC_MIPE_qHTS.csv",
        "s-ntap-IPNF95-1bc-1",
    ),
    (
        "NTAP ipNF02.8 MIPE qHTS.csv",
        "s-ntap-ipn02.8-1",
    ),
    (
        "NTAP ipNF05.5 SC MIPE qHTS.csv",
        "s-ntap-IPNF05_5_P31-1",
    ),
    (
        "NTAP ipNF05.5 MC MIPE qHTS.csv",
        "s-ntap-IPNF05_5_P30-1",
    ),
    (
        "NTAP ipNF06.2A MIPE qHTS.csv",
        "s-ntap-IPNF06_2A_P27-1",
    ),
    (
        "NTAP ipNF95.6 MIPE qHTS.csv",
        "s-ntap-IPNF95_6_P53-1",
    ),
    (
        "NTAP ipnNF95.11C MIPE qHTS.csv",
        "s-ntap-ipnNF95_11C_P31-1",
    ),
    (
        "NTAP ipNF95.11b C_T MIPE qHTS.csv",
        "s-ntap-IPNF95_11b_P53-1",
    ),
    (
        "s-ntap-HFF-1.csv",
        "s-ntap-HFF-1",
    ),
    (
        "s-ntap-MTC-1.csv",
    ),
]


BREWER_12_PAIRED = [
    "#a6cee3",
    "#1f78b4",
    "#b2df8a",
    "#33a02c",
    "#fb9a99",
    "#e31a1c",
    "#fdbf6f",
    "#ff7f00",
    "#cab2d6",
    "#6a3d9a",
    "#ffff99",
    "#b15928",
]

BREWER_12_SET3 = [
    "#8dd3c7",
    "#ffffb3",
    "#bebada",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
    "#b3de69",
    "#fccde5",
    "#d9d9d9",
    "#bc80bd",
    "#ccebc5",
    "#ffed6f",
]

IBM_COLORS = [
    "#648fff",
    "#dc267f",
    "#ffb000",
    "#fe6100",
    "#785ef0",
    "#000000",
    "#ffffff",
]


def ll4(c, h, inf, zero, ec50):
    """A copy of the LL.4 function from the R drc package with,

     - c: concentration
     - h: hill slope
     - inf: asymptote at max concentration
     - zero: asymptote at zero concentration
     - ec50: EC50
    """
    num = zero - inf
    den = 1 + np.exp(h * (np.log(c)-np.log(ec50)))
    response = inf + num / den
    return response


class DoseResponseCurve:

    def __init__(self, smm_hts, df_raw):
        self.smm_hts = smm_hts
        self.df_raw = df_raw

        self.set_concentration_multipliers()
        self.create_df()

    @property
    def raw_col_name(self):
        for col in ["Name", "name"]:
            if col in self.df_raw.columns:
                return col

    @property
    def raw_col_target(self):
        for col in ["Target", "target"]:
            if col in self.df_raw.columns:
                return col

    @property
    def raw_col_ncgc(self):
        for col in ["NCGC SID", "NCGCID", "SID"]:
            if col in self.df_raw.columns:
                return col

    @property
    def raw_col_smiles(self):
        for col in ["SMILES", "smi"]:
            if col in self.df_raw.columns:
                return col

    @property
    def raw_cols_conc(self):
        if "C1" in self.df_raw.columns:
            c_cols = [
                col for col in self.df_raw.columns
                if re.match("C\d+", col)]

        if "CONC1" in self.df_raw.columns:
            c_cols = [
                col for col in self.df_raw.columns
                if re.match("CONC\d+", col)]

        assert len(c_cols) == 11
        return c_cols

    @property
    def raw_cols_resp(self):
        if "DATA1" in self.df_raw.columns:
            r_cols = [
                col for col in self.df_raw.columns
                if re.match("DATA\d+", col)]

        assert len(r_cols) == 11
        return r_cols

    def create_df(self):

        self.df = pd.DataFrame()
        self.df["NCGC SID"] = self.df_raw[self.raw_col_ncgc]
        self.df["name"] = self.df_raw[self.raw_col_name].values
        self.df["target"] = self.df_raw[self.raw_col_target].values

        if "MoA" in self.df_raw.columns:
            self.df["MoA"] = self.df_raw["MoA"]

        if "R2" in self.df_raw.columns:
            self.df["R2"] = self.df_raw["R2"]

        if "AC50" in self.df_raw.columns:
            self.df["AC50"] = self.df_raw["AC50"].values * self.ac50_mult

        if "LAC50" in self.df_raw.columns:
            self.df["LAC50"] = self.df_raw["LAC50"].values + np.log10(self.ac50_mult)

        if "AC50" in self.df.columns and "LAC50" not in self.df.columns:
            self.df["LAC50"] = np.log10(self.df["AC50"])

        if "LAC50" in self.df.columns and "AC50" not in self.df.columns:
            self.df["AC50"] = 10**self.df["LAC50"]

        assert "AC50" in self.df.columns and "LAC50" in self.df.columns

        for col, raw_col in self.get_fit_cols().items():
            if col == "AC50":
                continue
            else:
                self.df[col] = self.df_raw[raw_col].values

        self.df["SMILES"] = self.df_raw[self.raw_col_smiles].values

        for ii, col in enumerate(self.raw_cols_conc):
            self.df[f"CONC{ii}"] = self.df_raw[col].values * self.c_mult

        for ii, col in enumerate(self.raw_cols_resp):
            self.df[f"DATA{ii}"] = self.df_raw[col].values


    def set_concentration_multipliers(self):
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

        if self.smm_hts["name"] in group_1:
            self.ac50_mult = 1e6
            self.c_mult = 1.0
        elif self.smm_hts["name"] in group_2:
            self.ac50_mult = 1.0
            self.c_mult = 1e6
        else:
            raise ValueError()

    def get_non_rc_cols(self):
        return [
            col for col in self.df.columns
            if col not in C_COLS + R_COLS
        ]

    def get_drug_cols(self):
        return ["NCGC SID", "name", "target", "SMILES"]


    def get_fit_cols(self):
        if "HILL" in self.df_raw.columns:
            hill = "HILL"
        elif "Hill" in self.df_raw.columns:
            hill = "Hill"

        if "INF" in self.df_raw.columns:
            inf = "INF"
        elif "Infinity" in self.df_raw.columns:
            inf = "Infinity"

        if "ZERO" in self.df_raw.columns:
            zero = "ZERO"
        elif "Zero" in self.df_raw.columns:
            zero = "Zero"

        return {
            "HILL": hill,
            "INF": inf,
            "ZERO": zero,
            "AC50": "AC50",
        }




def read_metadata(data_path):

    # text file with some variable definitions
    df_qhts_pdh = pd.read_csv(
        data_path / "qhts-protocol-dump-headers.txt",
        header=None,
    )

    # data manifest for matrix portal data
    df_ssm = pd.read_csv(
        data_path / "matrix portal raw data/synapse_storage_manifest.csv",
    )

    # data manifest for all files
    # might only be present when syncing with python client
    df_smm = pd.read_csv(
        data_path / "SYNAPSE_METADATA_MANIFEST.tsv",
        sep="\t",
    )
    df_smm_hts = df_smm[
        df_smm['name'].apply(lambda x: 'ntap' in x.lower())
    ].reset_index(drop=True)

    df_smm_hts = df_smm_hts.set_index("name")
    df_smm_hts = df_smm_hts.loc[funcy.flatten(SMM_NAME_GROUPS)]
    df_smm_hts = df_smm_hts.reset_index()

    proc_indxs = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    raw_indxs = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

    smm_name_to_specimen_id = {}
    for pi, ri in zip(proc_indxs, raw_indxs + [20]):
        specimen_id = df_smm_hts.iloc[pi]["specimenID"]
        name_p = df_smm_hts.iloc[pi]["name"]
        name_r = df_smm_hts.iloc[ri]["name"]
        smm_name_to_specimen_id[name_p] = specimen_id
        smm_name_to_specimen_id[name_r] = specimen_id


    specimen_ids = df_smm_hts.iloc[proc_indxs]["specimenID"].values

    df_ssm_mrgd = pd.DataFrame(index=specimen_ids)
    df_ssm_mrgd['disease'] = df_smm_hts.iloc[proc_indxs]["disease"].values
    df_ssm_mrgd['nf1Genotype'] = df_smm_hts.iloc[proc_indxs]["nf1Genotype"].values
    df_ssm_mrgd['nf2Genotype'] = df_smm_hts.iloc[proc_indxs]["nf2Genotype"].values
    df_ssm_mrgd['tissue'] = df_smm_hts.iloc[proc_indxs]["tissue"].values
    df_ssm_mrgd['organ'] = df_smm_hts.iloc[proc_indxs]["organ"].values
    df_ssm_mrgd['species'] = df_smm_hts.iloc[proc_indxs]["species"].values
    df_ssm_mrgd['cellType'] = df_smm_hts.iloc[proc_indxs]["cellType"].values

    df_ssm_mrgd['sex'] = list(df_smm_hts.iloc[raw_indxs]["sex"].values) + [None]
    df_ssm_mrgd['tumorType'] = list(df_smm_hts.iloc[raw_indxs]["tumorType"].values) + [None]
    df_ssm_mrgd['diagnosis'] = list(df_smm_hts.iloc[raw_indxs]["diagnosis"].values) + [None]
    df_ssm_mrgd['modelSystemName'] = list(df_smm_hts.iloc[raw_indxs]["modelSystemName"].values) + [None]

    # fixups for NA
    for col in ["tumorType", "diagnosis"]:
        df_ssm_mrgd[col] = df_ssm_mrgd[col].apply(lambda x: x if x != "Not Applicable" else None)

    # fixups for ipNF95.11bC processes says +/- raw says -/-
    df_ssm_mrgd.loc["ipNF95.11bC", "nf1Genotype"] = "-/-, +/-"

    return df_qhts_pdh, df_ssm, df_smm, df_smm_hts, df_ssm_mrgd, smm_name_to_specimen_id


def read_raw_hts(df_smm_hts):
    dfs_raw = {}
    for indx, row in df_smm_hts.iterrows():
        if row["name"].endswith("csv"):
            path_suffix = Path(row["name"])
        else:
            path_suffix = Path("matrix portal raw data") / row["name"]
        file_path = data_path / path_suffix
        df = pd.read_csv(file_path)
        dfs_raw[row["name"]] = df
    return dfs_raw

def make_mrgd_hts(drcs, smm_name_to_specimen_id):
    dfs_mrgd = {}
    for name, drc in drcs.items():
        specimen_id = smm_name_to_specimen_id[name]
        if specimen_id in dfs_mrgd:
            for col in drc.df.columns:
                if not col in dfs_mrgd[specimen_id]:
                    dfs_mrgd[specimen_id][col] = drc.df[col]

        else:
            dfs_mrgd[specimen_id] = drc.df.copy()

    return dfs_mrgd


data_path = Path("/home/galtay/data/hack4nf-2022/synapse/syn5522627")
df_qhts_pdh, df_ssm, df_smm, df_smm_hts, df_smm_mrgd, smm_name_to_specimen_id = read_metadata(data_path)



smm_hide_cols = [
    "index", "path", "parent", "id", "synapseStore",
    "contentType", "used", "executed", "activityName",
    "experimentalTimepoint", "timePointUnit", "accessType",
    "activityDescription", "studyId", "fundingAgency",
    "isPrimaryCell", "entityId", "Resource_id", "isCellLine",
    "resourceType", "experimentalCondition", "reporterGene",
    "accessTeam", "cellLineMetadataSynId", "specimenMetadataSynId",
    "initiative", "isMultiIndividual", "fileFormat", "assay",
    "individualMetadataSynId", "studyName", "isMultiSpecimen",
    "eTag", "reporterSubstance", "dataType", "drugScreenType",
]
smm_show_cols = [col for col in df_smm_hts.columns if col not in smm_hide_cols]


dfs_raw = read_raw_hts(df_smm_hts)


drcs = {}
for smm_name, df_raw in dfs_raw.items():
    rows = df_smm_hts[df_smm_hts["name"]==smm_name][smm_show_cols]
    assert rows.shape[0] == 1
    row = rows.iloc[0]
    drc = DoseResponseCurve(row.to_dict(), df_raw)
    drcs[smm_name] = drc

dfs_mrgd = make_mrgd_hts(drcs, smm_name_to_specimen_id)


with st.sidebar:
    st.header("Cell Line")
    specimen_id = st.selectbox("Specimen ID", df_smm_mrgd.index)
    st.write(df_smm_mrgd.loc[specimen_id].to_dict())

    st.header("Compound")
    unique_ncgc_sids = set()
    for smm_name, drc in drcs.items():
        unique_ncgc_sids.update(drc.df["NCGC SID"].unique())
    unique_ncgc_sids = sorted(list(unique_ncgc_sids))
    ncgc_sid = st.selectbox("NCGC SID", unique_ncgc_sids)

    tmp_row = dfs_mrgd[specimen_id][dfs_mrgd[specimen_id]["NCGC SID"]==ncgc_sid]
    st.write(tmp_row[["NCGC SID", "name", "target", "MoA", "SMILES"]].iloc[0].to_dict())

tab1, tab2, tab3 = st.tabs(["Raw Data", "Drug Explorer", "Cell Line Explorer"])


def show_raw_data(drcs):
    for smm_name_group in SMM_NAME_GROUPS:

        col1, col2 = st.columns(2)

        with col1:
            smm_name = smm_name_group[0]
            drc = drcs[smm_name]

            st.header(smm_name)
            st.write(f"df_raw.shape: {drc.df_raw.shape}")
            st.write(drc.smm_hts)
            st.write(drc.df_raw.head(10))

            fig = px.histogram(drc.df["LAC50"])
            c_min = np.log10(drc.df_raw[drc.raw_cols_conc[0]].min())
            c_max = np.log10(drc.df_raw[drc.raw_cols_conc[-1]].max())

            fig.add_vline(x=c_min, line_color=IBM_COLORS[1])
            fig.add_vline(x=c_max, line_color=IBM_COLORS[1])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if len(smm_name_group) > 1:
                smm_name = smm_name_group[1]
                drc = drcs[smm_name]

                st.header(smm_name)
                st.write(f"df_raw.shape: {drc.df_raw.shape}")
                st.write(drc.smm_hts)
                st.write(drc.df_raw.head(10))

                fig = px.histogram(drc.df["LAC50"])
                c_min = np.log10(drc.df_raw[drc.raw_cols_conc[0]].min())
                c_max = np.log10(drc.df_raw[drc.raw_cols_conc[-1]].max())

                fig.add_vline(x=c_min, line_color=IBM_COLORS[1])
                fig.add_vline(x=c_max, line_color=IBM_COLORS[1])
                st.plotly_chart(fig, use_container_width=True)



with tab1:
    st.header("SYNAPSE_METADATA_MANIFEST.csv")
    st.dataframe(df_smm_hts[smm_show_cols])
    st.dataframe(df_smm_mrgd)
    show_raw_data(drcs)



def get_measured_trace(row, label=None, showlegend=False, color=None):
    cs = row[C_COLS].astype(float).values
    rs = row[R_COLS].astype(float).values
    tr_measured = go.Scatter(
        x=cs,
        y=rs,
        mode='markers',
        name=label,
        showlegend=showlegend,
        marker_color=color,
    )
    return tr_measured

def get_fit_trace(row, label=None, showlegend=False, color=None):
    cs = row[C_COLS].astype(float).values
    fit_rs = ll4(
        cs,
        row["HILL"],
        row["INF"],
        row["ZERO"],
        row["AC50"],
    )
    tr_fit = go.Scatter(
        x=cs,
        y=fit_rs,
        mode='lines',
        showlegend=showlegend,
        name=label,
        line_color=color,
        line_dash="dot",
    )
    return tr_fit

def get_ac50_trace(row, label=None, showlegend=False, color=None):
    rs = row[R_COLS].astype(float).values
    tr_ac50 = go.Scatter(
        x=[row["AC50"]] * 2,
        #y=[min(rs), max(rs)],
        y=[row["ZERO"], row["INF"]],
        mode='lines',
        showlegend=showlegend,
        name=label,
        line_color=color,
    )
    return tr_ac50


with tab2:

    #specimen_ids = list(dfs_mrgd.keys())
    specimen_ids = df_smm_mrgd.sort_values("disease").index
    num_cell_lines = len(specimen_ids)

    n_cols = 4
    n_rows = 3
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_xaxes="all",
        shared_yaxes="all",
        vertical_spacing=0.05,
        start_cell="top-left",
        subplot_titles=specimen_ids,
    )

    i_row = 1
    i_col = 1
    i_tot = 0

    for specimen_id in specimen_ids:

        df = dfs_mrgd[specimen_id]

        row = df[df["NCGC SID"]==ncgc_sid].iloc[0]

        i_color = i_tot
        tr_measured = get_measured_trace(row, color=BREWER_12_SET3[i_color])
        tr_fit = get_fit_trace(row, color=BREWER_12_SET3[i_color])
        tr_ac50 = get_ac50_trace(row, color=BREWER_12_SET3[i_color])
        for tr in [tr_measured, tr_fit, tr_ac50]:
            fig.add_trace(tr, row=i_row, col=i_col)

        i_tot += 1
        i_col += 1
        if i_col > n_cols:
            i_col = 1
            i_row += 1

    fig.update_xaxes(type="log") # , gridwidth=0.1, gridcolor="white") # , title="Dose")
    fig.update_yaxes(showgrid=True) # , gridwidth=0.1, gridcolor="white") # title="Response")
    fig.update_layout(height=1100)
    st.plotly_chart(fig, use_container_width=True)



with tab3:

    st.header("Cell Lines")
    st.dataframe(df_smm_mrgd)

    st.header(specimen_id)
    st.write(dfs_mrgd[specimen_id].head(10))

    iloc = st.number_input(
        "sample index",
        min_value=0,
        max_value=dfs_mrgd[specimen_id].shape[0]-1,
        value=0,
    )

    row = dfs_mrgd[specimen_id].iloc[iloc][[
        col for col in dfs_mrgd[specimen_id].columns
        if col not in C_COLS + R_COLS
    ]]
    st.write(row.to_dict())
