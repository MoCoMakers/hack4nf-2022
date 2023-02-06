from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import toml

import single_agent_screens as sas

st.set_page_config(layout="wide")


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

IBM_COLORS = [
    "#648fff",
    "#dc267f",
    "#ffb000",
    "#fe6100",
    "#785ef0",
    "#000000",
    "#ffffff",
]
COLORS = {
    "screen": IBM_COLORS[0],
    "norm": IBM_COLORS[1],
}
DEBUG = True
C_COLS = [f"C{i}" for i in range(11)]
R_COLS = [f"DATA{i}" for i in range(11)]



def ll4(c, h, r_min, r_max, ec50):
    """A copy of the LL.4 function from the R drc package with,

     - c: concentration
     - h: hill slope
     - r_min: min response
     - r_max: max response
     - ec50: EC50
    """
    num = r_max - r_min
    den = 1 + np.exp(h * (np.log(c)-np.log(ec50)))
    response = r_min + num / den
    return response


def read_config():
    config_file_path = "sas_config.toml"
    with open(config_file_path, "r") as fp:
        config = toml.load(fp)
    return config


def read_dfs(config):
    base_path = Path(config["paths"]["base"])
    hts_file_paths = sas.get_hts_file_paths(
        base_path,
        config["cell_lines"]["screen"],
        config["cell_lines"]["norm"],
    )
    dfs = sas.read_hts_files(hts_file_paths)
    return dfs


config = read_config()
all_cell_lines = config["cell_lines"]["screen"] + config["cell_lines"]["norm"]

st.sidebar.header("Cell Lines")
screen_cell_line = st.sidebar.selectbox(
    "screen cell line",
    all_cell_lines,
    index=0,
)
norm_cell_line = st.sidebar.selectbox(
    "norm cell line",
    [None] + all_cell_lines,
    index=0,
)

dfs = read_dfs(config)
df_all = pd.concat(dfs.values())

df_screen = dfs[screen_cell_line]

st.sidebar.header("Plot Choices")
iloc = st.sidebar.number_input(
    "sample index",
    min_value=0,
    max_value=df_screen.shape[0]-1,
    value=0,
)

upper_fit_param = st.sidebar.selectbox(
    "upper fit param",
    ["INF", "MAXR"],
    index=0,
)

lower_fit_param = st.sidebar.selectbox(
    "upper fit param",
    ["ZERO"],
    index=0,
)


def get_hill_traces(row, label):

    cs = row[C_COLS].astype(float).values
    rs = row[R_COLS].astype(float).values

    tr_measured = go.Scatter(
        x=cs,
        y=rs,
        mode='markers',
        name=f'{label}',
        marker_color=COLORS[label],
    )

    fit_rs = ll4(
        cs,
        row["HILL"],
        row[upper_fit_param],
        row[lower_fit_param],
        10**row["LAC50"] * 1e6,
    )

    tr_fit = go.Scatter(
        x=cs,
        y=fit_rs,
        mode='lines',
        showlegend=False,
#        name=f'{label} ll4 fit',
        line_color=COLORS[label],
        line_dash="dot",
    )

    tr_ac50 = go.Scatter(
        x=[row["AC50"] * 1e6] * 2,
        y=[min(rs), max(rs)],
        mode='lines',
        showlegend=False,
#        name=f'{label} AC50',
        line_color=COLORS[label],
    )

    return tr_measured, tr_fit, tr_ac50



tab1, tab2, tab3 = st.tabs(["Curves", "Distros", "Owl"])


with tab1:

    screen_row = df_screen.iloc[iloc]
    screen_traces = get_hill_traces(screen_row, "screen")

    fig = go.Figure()
    for tr in screen_traces:
        fig.add_trace(tr)

    if norm_cell_line is not None:

        df_norm = dfs[norm_cell_line]
        assert df_screen.shape[0] == df_norm.shape[0]
        norm_row = df_norm.loc[screen_row.name]
        norm_traces = get_hill_traces(norm_row, "norm")
        for tr in norm_traces:
            fig.add_trace(tr)

    fig.update_xaxes(type="log", title="Concentration")
    fig.update_yaxes(title="Response")
    #fig.update_layout(showlegend=False)
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    ))

#    st.title("Hill Curve AC50 Ratios")

    col1, col2 = st.columns(2)



    col1.header("Compound")
    col1.markdown("""
      - **Name**: {}
      - **Target**: {}
      - **NCGC SID**: {}
    """.format(
        screen_row["name"].replace("\n", " "),
        screen_row["target"],
        screen_row.name,
#        screen_row["smi"],
    ))


    col2.header("Fits")

    lis = []

    def color_span(txt, color):
        return f"""<span style="color:{color}">{txt}</span>"""

    lis.append("""
    <li> <b>Screen</b>: {}={:.2f}, {}={:.2e} </li>
    """.format(
        color_span("R2", COLORS["screen"]),
        screen_row["R2"],
        color_span("AC50", COLORS["screen"]),
        screen_row["AC50"]*1e6,
    ))


    if norm_cell_line is not None:

        lis.append("""
        <li> <b>Norm</b>: {}={:.2f}, {}={:.2e} </li>
        """.format(
            color_span("R2", COLORS["norm"]),
            norm_row["R2"],
            color_span("AC50", COLORS["norm"]),
            norm_row["AC50"]*1e6,
        ))

        lis.append("""
        <li> <b>Ratio</b>: Log {} / {}={:.2f} </li>
        """.format(
            color_span("AC50", COLORS["screen"]),
            color_span("AC50", COLORS["norm"]),
            np.log10(screen_row["AC50"] / norm_row["AC50"]),
        ))


    html = """
    <ul>
    {}
    </ul>""".format("\n".join([el.strip() for el in lis]))
    col2.markdown(html, unsafe_allow_html=True)

    st.plotly_chart(fig, use_container_width=True)

    if DEBUG:
        st.dataframe(df_screen)
        st.write(screen_row)
        if norm_cell_line is not None:
            st.write(norm_row)
        st.write(config)

with tab2:

    st.header("R2 Distribution")
    fig = px.histogram(df_all['R2'])
    st.plotly_chart(fig, use_container_width=True)

    st.header("Log AC50")
    fig = px.histogram(df_all['LAC50'] + 6)
    st.plotly_chart(fig, use_container_width=True)
