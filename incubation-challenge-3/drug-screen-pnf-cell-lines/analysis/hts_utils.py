import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots


R2_THRESH = 0.80
R_COLS = [f"DATA{i}" for i in range(0, 11)]
C_COLS = [f"CONC{i}" for i in range(0, 11)]
SHOW_COLS = [
    "Cell line",
    "NCGC SID",
    "name",
    "target",
    "MoA",
    "R2",
    "AC50",
    "HILL",
    "INF",
    "ZERO",
    "MAXR",
] + [
    "EFF",
    "EFF/AC50",
    "log(EFF/AC50)",
]
FIT_COLS = [
    "R2",
    "AC50",
    "HILL",
    "INF",
    "ZERO",
    "MAXR",
] + [
    "EFF",
    "EFF/AC50",
    "log(EFF/AC50)",
]
BREWER_9_SET1 = [
    "#f781bf",
    "#a65628",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#e41a1c",
    "#377eb8",
    "#999999",
]
COLORS = BREWER_9_SET1
CELL_LINE_META = {
    "ipnNF95.11C": {"source": "peripheral nerve", "status": "-/-,+/-"},
    #    'ipNF95.11bC',
    "HFF": {"source": "foreskin fibroblast", "status": "+/+"},
    "ipNF02.3 2l": {"source": "schwann cell", "status": "+/+"},
    "ipNF02.8": {"source": "schwann cell", "status": "+/+"},
    "ipNF05.5 Mixed Clones": {"source": "plexiform neurofibroma", "status": "-/-"},
    "ipNF05.5 Single Clone": {"source": "plexiform neurofibroma", "status": "-/-"},
    "ipNF06.2A": {"source": "plexiform neurofibroma", "status": "-/-"},
    "ipNF95.11b C/T": {"source": "plexiform neurofibroma", "status": "-/-"},
    "ipNF95.6": {"source": "plexiform neurofibroma", "status": "-/-"},
}
CELL_LINES = [
    "ipnNF95.11C",
    #    'ipNF95.11bC',
    "HFF",
    "ipNF02.3 2l",
    "ipNF02.8",
    "ipNF05.5 Mixed Clones",
    "ipNF05.5 Single Clone",
    "ipNF06.2A",
    "ipNF95.11b C/T",
    "ipNF95.6",
]
CL_TO_COLOR = {cl: COLORS[ii] for ii, cl in enumerate(CELL_LINES)}



def hts_read(
    file_path,
    source_dir="top",
    filter_cell_lines=True,
    filter_curve_cols=False,
):
    if source_dir not in ["top", "matrix_portal_raw_data"]:
        raise ValueError()
    df_hts_all = pd.read_csv(file_path)
    df_hts = df_hts_all[df_hts_all["source_dir"] == source_dir]
    if filter_cell_lines:
        df_hts = df_hts[df_hts["Cell line"].isin(CELL_LINES)]
    if filter_curve_cols:
        cols = [c for c in df_hts.columns if c not in C_COLS + R_COLS]
        df_hts = df_hts[cols]
    return df_hts


def hts_add_vars(df_hts):
    # df_hts["EFF"] = df_hts["ZERO"] - df_hts["MAXR"]
    df_hts["EFF"] = df_hts["ZERO"] - df_hts["INF"]
    df_hts["EFF/AC50"] = df_hts["EFF"] / df_hts["AC50"]
    df_hts["log(EFF/AC50)"] = np.log10(df_hts["EFF/AC50"])
    return df_hts


def ll4(c, h, inf, zero, ac50):
    """A copy of the LL.4 function from the R drc package with,
    https://doseresponse.github.io/drc/reference/LL.4.html

     - c: concentration
     - h: hill slope
     - inf: asymptote at infinite concentration
     - zero: asymptote at zero concentration
     - ac50: AC50
    """
    num = zero - inf
    den = 1 + np.exp(h * (np.log(c) - np.log(ac50)))
    response = inf + num / den
    return response


def hts_compare(
    df_hts_in,
    ref_cell_line,
    tumor_cell_line,
    r2_thresh=None,
    eff_thresh=None,
):
    print("compariing ref={} and tumor={}".format(ref_cell_line, tumor_cell_line))

    df_hts = df_hts_in.copy()
    df_hts = df_hts[df_hts["Cell line"].isin([ref_cell_line, tumor_cell_line])]
    print("nrows after filtering cell lines: {}".format(df_hts.shape[0]))

    if r2_thresh is not None:
        df_hts = df_hts[df_hts["R2"] >= r2_thresh]
        print("nrows after filtering R2: {}".format(df_hts.shape[0]))

    if eff_thresh is not None:
        df_hts = df_hts[df_hts["EFF"] > eff_thresh]
        print("nrows after filtering EFF: {}".format(df_hts.shape[0]))

    # only keep compounds that were not filtered from either cell line
    keep_ncgc_sids = set(df_hts["NCGC SID"].values)
    for cl, df in df_hts.groupby("Cell line"):
        keep_ncgc_sids = keep_ncgc_sids.intersection(set(df["NCGC SID"]))
        print("cell line={}, kept compounds={}".format(cl, df.shape[0]))
    print("total kept compounds={}".format(len(keep_ncgc_sids)))

    df_hts = df_hts[df_hts["NCGC SID"].isin(keep_ncgc_sids)]

    dfs_hts = {cl: df for cl, df in df_hts.groupby("Cell line")}
    dfs_hts = {cl: df.set_index("NCGC SID") for cl, df in dfs_hts.items()}
    df_ratios = pd.DataFrame()

    df_ref = dfs_hts[ref_cell_line]
    df_tumor = dfs_hts[tumor_cell_line]

    df = df_ref[["name", "target", "MoA"]].copy()
    df["ref_line"] = ref_cell_line
    df["tumor_line"] = tumor_cell_line

    for col in ["R2", "AC50", "LAC50", "EFF", "log(EFF/AC50)"]:
        df[f"ref_{col}"] = df_ref[col]
        df[f"tumor_{col}"] = df_tumor[col]


    df["AC50_r/t"] = df_ref["AC50"] / df_tumor["AC50"]
    df["log(AC50_r/t)"] = np.log10(df["AC50_r/t"])

    df["AC50_t/r"] = df_tumor["AC50"] / df_ref["AC50"]
    df["log(AC50_t/r)"] = np.log10(df["AC50_t/r"])

    df["EFF_r/t"] = df_ref["EFF"] / df_tumor["EFF"]

    # ds = (EFF/AC50)_ref / (EFF/AC50)_tumor
    # ds = (EFF_ref / EFF_tumor) (AC50_tumor / AC50_ref)
    df["EFF/AC50_r/t"] = df_ref["EFF/AC50"] / df_tumor["EFF/AC50"]
    df["log(EFF/AC50_r/t)"] = np.log10(df["EFF/AC50_r/t"])

    return df


def get_measured_trace(
    row,
    marker_symbol="circle",
    marker_size=7,
    label=None,
    showlegend=False,
    color=None,
):
    cs = row[C_COLS].astype(float).values
    rs = row[R_COLS].astype(float).values
    tr = go.Scatter(
        x=cs,
        y=rs,
        mode="markers",
        marker_symbol=marker_symbol,
        marker_size=marker_size,
        name=label,
        showlegend=showlegend,
        marker_color=color,
    )
    return tr


def get_fit_trace(
    row, label=None, showlegend=False, color=None, line_width=2, line_dash="dot"
):
    cs = row[C_COLS].astype(float).values
    fit_rs = ll4(
        cs,
        row["HILL"],
        row["INF"],
        row["ZERO"],
        row["AC50"],
    )
    tr = go.Scatter(
        x=cs,
        y=fit_rs,
        mode="lines",
        showlegend=showlegend,
        name=label,
        line_color=color,
        line_width=line_width,
        line_dash=line_dash,
    )
    return tr


def get_vert_trace(
    row,
    key,
    label=None,
    showlegend=False,
    color=None,
    ymax=140,
    line_width=1,
    line_dash="dot",
):
    tr = go.Scatter(
        x=[row[key]] * 2,
        y=[0, ymax],
        mode="lines",
        showlegend=showlegend,
        name=label,
        line_color=color,
        line_dash=line_dash,
        line_width=line_width,
    )
    return tr


def get_horiz_trace(
    row,
    key,
    label=None,
    showlegend=False,
    color=None,
    xmax=None,
    line_width=1,
    line_dash="dot",
):
    tr = go.Scatter(
        x=[0, xmax],
        y=[row[key]] * 2,
        mode="lines",
        showlegend=showlegend,
        name=label,
        line_color=color,
        line_dash=line_dash,
        line_width=line_width,
    )
    return tr


def fig_add_compound(
    fig,
    row,
    color="white",
    params_color=None,
    measured_color=None,
    fit_color=None,
    annotations_color=None,
    symbol="circle",
    add_measured=True,
    add_fit=True,
    add_params=True,
    add_annotations=True,
    showlegend=False,
    legend_name=None,
    xmin=-3.5,
    xmax=2.2,
    ymin=0,
    ymax=140,
):
    params_color = params_color or color
    measured_color = measured_color or color
    fit_color = fit_color or color
    annotations_color = annotations_color or params_color or color

    if add_params:
        tr_ac50 = get_vert_trace(row, "AC50", color=params_color, ymax=ymax)
        tr_inf = get_horiz_trace(row, "INF", color=params_color, xmax=10**xmax)
        tr_zero = get_horiz_trace(row, "ZERO", color=params_color, xmax=10**xmax)
        for tr in [tr_ac50, tr_zero, tr_inf]:
            fig.add_trace(tr)

    if add_measured:
        tr_measured = get_measured_trace(
            row,
            marker_symbol=symbol,
            label=row["name"],
            color=measured_color,
        )
        fig.add_trace(tr_measured)

    if add_fit:
        tr_fit = get_fit_trace(
            row,
            color=fit_color,
            line_dash=None,
            showlegend=showlegend,
            label=legend_name,
        )
        fig.add_trace(tr_fit)

    if add_annotations and not np.isnan(row["R2"]):
        annotations = [
            dict(
                x=1.00,
                y=row["INF"],
                text="INF",
                xref="paper",
                xanchor="left",
                showarrow=False,
                font=dict(color=annotations_color),
            ),
            dict(
                x=1.00,
                y=row["ZERO"],
                text="ZERO",
                xref="paper",
                xanchor="left",
                showarrow=False,
                font=dict(color=annotations_color),
            ),
            dict(
                x=row["LAC50"],
                y=1.0,
                text="AC50",
                yref="paper",
                yanchor="bottom",
                showarrow=False,
                font=dict(color=annotations_color),
            ),
        ]
        for annotation in annotations:
            fig.add_annotation(annotation)

    return fig


def get_single_cellline_single_compound_title(row):
    title = "Cell Line: {} ({}, {})<br>Compound: {}<br>Target: {}<br>MoA: {}".format(
        row["Cell line"],
        CELL_LINE_META[row["Cell line"]]["source"],
        CELL_LINE_META[row["Cell line"]]["status"],
        row["name"],
        row["target"],
        row["MoA"],
    )
    return title


def fig_update_layout(
    fig,
    margin,
    title="",
    axes_color="black",
    xmin=-3.5,
    xmax=2.2,
    ymin=0,
    ymax=140,
    width=600,
    height=650,
    global_font_size=12,
    title_font_size=None,
):
    fig.update_layout(
        title=dict(
            text=title,
            font_size=title_font_size,
            x=0.5,
            y=0.95,
            xanchor="center",
            yanchor="top",
        ),
        font={"size": global_font_size},
        margin=margin,
        width=width,
        height=height,
    )

    fig.update_xaxes(
        title="Concentration [uM]",
        type="log",
        range=[xmin, xmax],
        showgrid=False,
        showline=True,
        linewidth=4,
        linecolor=axes_color,
        tickcolor=axes_color,
        tickwidth=3,
        tickvals=np.logspace(-3, 2, 6),
    )
    fig.update_yaxes(
        title="Response [% of DMSO Control]",
        range=[ymin, ymax],
        showgrid=False,
        linewidth=4,
        linecolor=axes_color,
        tickcolor=axes_color,
        tickwidth=3,
    )

    return fig
