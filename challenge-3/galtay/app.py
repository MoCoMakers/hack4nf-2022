from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import toml


import single_agent_screens as sas

st.set_page_config(layout="wide")

DEBUG = False
C_COLS = [f"C{i}" for i in range(11)]
R_COLS = [f"DATA{i}" for i in range(11)]


def ll4(c, h, r_min, r_max, ec50):
    """This function is basically a copy of the LL.4 function from the R drc package with
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



config_file_path = "sas_config.toml"
with open(config_file_path, "r") as fp:
    config = toml.load(fp)




base_path = Path(config["paths"]["base"])
hts_file_paths = sas.get_hts_file_paths(
    base_path,
    config["cell_lines"]["screen"],
    config["cell_lines"]["norm"],
)
dfs = sas.read_hts_files(hts_file_paths)


all_cell_lines = config["cell_lines"]["screen"] + config["cell_lines"]["norm"]

screen_cell_line = st.sidebar.selectbox(
    "screen cell line",
    all_cell_lines,
    index=0,
)
df_screen = dfs[screen_cell_line]

norm_cell_line = st.sidebar.selectbox(
    "norm cell line",
    [None] + all_cell_lines,
    index=0,
)

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



screen_row = df_screen.iloc[iloc]
screen_cs = screen_row[C_COLS].astype(float).values
screen_rs = screen_row[R_COLS].astype(float).values

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=screen_cs,
    y=screen_rs,
    mode='markers',
    name='screen measured',
    marker_color='black'
))

fit_screen_rs = ll4(
    screen_cs,
    screen_row["HILL"],
    screen_row[upper_fit_param],
    screen_row[lower_fit_param],
    10**screen_row["LAC50"] * 1e6,
)

fig.add_trace(go.Scatter(
    x=screen_cs,
    y=fit_screen_rs,
    mode='lines',
    name='screen ll4 fit',
    line_color="black",
    line_dash="dot",
))

fig.add_vline(
    x=screen_row["AC50"] * 1e6,
    line_color="black",
    line_width=1.0,
    #line_dash="dot",
)



if norm_cell_line is not None:

    df_norm = dfs[norm_cell_line]
    assert df_screen.shape[0] == df_norm.shape[0]
    norm_row = df_norm.loc[screen_row.name]

    #st.write(norm_row)
    norm_cs = norm_row[C_COLS].astype(float).values
    norm_rs = norm_row[R_COLS].astype(float).values

    fig.add_trace(go.Scatter(
        x=norm_cs,
        y=norm_rs,
        mode='markers',
        name='norm measured',
        marker_color='red'
    ))

    fit_norm_rs = ll4(
        norm_cs,
        norm_row["HILL"],
        norm_row[upper_fit_param],
        norm_row[lower_fit_param],
        10**norm_row["LAC50"] * 1e6,
    )

    fig.add_trace(go.Scatter(
        x=norm_cs,
        y=fit_norm_rs,
        mode='lines',
        name='norm ll4 fit',
        line_color="red",
        line_dash="dot",
    ))

    fig.add_vline(
        x=norm_row["AC50"] * 1e6,
        line_color="red",
        line_width=1.0,
        #line_dash="dot",
    )

fig.update_xaxes(type="log", title="Concentration")
fig.update_yaxes(title="Response")


st.title("Hill Curve AC50 Ratios")
st.sidebar.header("Compound")
st.sidebar.write("Name: {}".format(screen_row["name"]))
st.sidebar.write("Target: {}".format(screen_row["target"]))
st.sidebar.markdown("SMILE: `{}`".format(screen_row["smi"]))

st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)
col1.header("Screen Fit")
col2.header("Norm Fit")


col1, col2, col3, col4 = st.columns(4)
col1.metric(label="R2", value="{:.2f}".format(screen_row["R2"]))
col2.metric(label="AC50", value="{:.2f}".format(screen_row["AC50"]*1e6))
if norm_cell_line is not None:
    col3.metric(label="R2", value="{:.2f}".format(norm_row["R2"]))
    col4.metric(label="AC50", value="{:.2f}".format(norm_row["AC50"]*1e6))


st.header("Ratio (Screen/Norm)")
col1, col2 = st.columns(2)
col1.metric(
    label="AC50 ratio",
    value="{:.2f}".format(screen_row["AC50"] / norm_row["AC50"]),
)
col2.metric(
    label="Log AC50 ratio",
    value="{:.2f}".format(np.log10(screen_row["AC50"] / norm_row["AC50"])),
)



if DEBUG:
    st.write(screen_row)
    if norm_cell_line is not None:
        st.write(norm_row)
    st.write(config)
