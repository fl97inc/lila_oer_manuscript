"""Fig 3d — per-element XRF molar loading before/after SEC testing, per catalyst.

Reads the shared `data/fig_3a-d.parquet`; writes `fig_3d.{png,svg}` to `figures/`.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

DATA_PATH = DATA_DIR / "fig_3a-d.parquet"
df = pd.read_parquet(DATA_PATH)

# Shared ink colour for all type, axis lines and tick marks.
INK = '#231F20'
LEGEND_BEFORE = '#231F20'
LEGEND_AFTER = '#8A8A8A'

# One colour per catalyst, in dataset row order: front-runners first, then
# Pd/Ru benchmarks. The light shade marks the post-test bar of each pair.
COLORS = [
    '#D7AA54',  # InMnPdOx
    '#715CA1',  # NiTaPdOx
    '#8B3D2B',  # CoFeZrOx
    '#458871',  # PdOx
    '#E26F46',  # RuOx
]
COLORS_LIGHT = [
    '#EDD4AD',  # InMnPdOx
    '#B7ABCD',  # NiTaPdOx
    '#C79C91',  # CoFeZrOx
    '#A4C2B6',  # PdOx
    '#F5B9A3',  # RuOx
]


def fmt_system(name: str) -> str:
    """'InMnPdOx' -> 'InMnPdO<sub>x</sub>' for HTML-rendered labels."""
    return re.sub(r'Ox$', 'O<sub>x</sub>', str(name))


def _nan_safe(v) -> float:
    return 0.0 if v is None or pd.isna(v) else float(v)


print(f'Loaded {len(df)} rows × {len(df.columns)} columns from {DATA_PATH}')

AX_KW = dict(mirror=False, linecolor=INK, linewidth=1,
             ticks='outside', ticklen=5, tickcolor=INK,
             minor=dict(ticks='outside', ticklen=3, nticks=3, tickcolor=INK))

BAR_W = 0.35
GAP_WITHIN_ELEMENT = 0.05   # gap between the pre and post bar for one element
GAP_BETWEEN_ELEMENTS = 0.3  # gap between adjacent elements in the same system
GAP_BETWEEN_SECTIONS = 1.2  # gap between adjacent systems

fig = go.Figure()

xrf_x_pos = 0.0
xrf_major_ticks = []          # (x_center, system_label) per catalyst
element_label_positions = []  # (x_center, element_symbol, max_y, colour) per bar pair

for row_idx, r in df.reset_index(drop=True).iterrows():
    section_start_x = None
    section_end_x = None

    for element in r['target_elements']:
        pre_v  = _nan_safe(r.get(f'{element} Loading Pre Test (μmol/cm²)'))
        post_v = _nan_safe(r.get(f'{element} Loading Post Test (μmol/cm²)'))

        pre_x  = xrf_x_pos
        post_x = xrf_x_pos + BAR_W + GAP_WITHIN_ELEMENT

        if section_start_x is None:
            section_start_x = pre_x

        fig.add_trace(go.Bar(
            x=[pre_x], y=[pre_v], width=BAR_W,
            marker_color=COLORS[row_idx],
            showlegend=False,
        ))
        fig.add_trace(go.Bar(
            x=[post_x], y=[post_v], width=BAR_W,
            marker_color=COLORS_LIGHT[row_idx],
            showlegend=False,
        ))

        element_label_positions.append((
            (pre_x + post_x) / 2, element, max(pre_v, post_v, 0.01),
            COLORS[row_idx],
        ))

        section_end_x = post_x
        xrf_x_pos = post_x + BAR_W + GAP_BETWEEN_ELEMENTS

    xrf_major_ticks.append(((section_start_x + section_end_x) / 2, r['System']))
    xrf_x_pos = section_end_x + BAR_W + GAP_BETWEEN_SECTIONS

fig.add_trace(go.Bar(x=[None], y=[None], marker_color=LEGEND_BEFORE,
                     name='As-deposited'))
fig.add_trace(go.Bar(x=[None], y=[None], marker_color=LEGEND_AFTER,
                     name='After SEC Testing'))

# Detection-limit grey band (y = 0 – 0.02 μmol cm⁻², full panel width).
fig.add_shape(
    type='rect',
    xref='x domain', yref='y',
    x0=0, x1=1, y0=0, y1=0.02,
    fillcolor='grey', opacity=0.15,
    line=dict(width=0),
    layer='above',
)

# Per-element labels above each bar pair.
for x_center, label, top_val, label_color in element_label_positions:
    fig.add_annotation(
        x=x_center, y=np.log10(top_val) + 0.08,
        text=f'<b>{label}</b>',
        showarrow=False, xref='x', yref='y',
        font=dict(size=17, color=label_color),
        xanchor='center', yanchor='bottom',
    )

fig.update_xaxes(
    range=[-0.50, 15.00], showgrid=False, tickmode='array',
    tickvals=[t[0] for t in xrf_major_ticks],
    ticktext=[fmt_system(t[1]) for t in xrf_major_ticks],
    tickangle=0,
    mirror=False, linecolor=INK, linewidth=1, tickcolor=INK,
)
fig.update_yaxes(
    title_text='<b>Molar Loading (μmol cm<sup>-2</sup>)</b>',
    type='log', range=[np.log10(0.005), np.log10(20)], dtick=1,
    showgrid=False,
    **{**AX_KW, 'minor': dict(ticks='outside', ticklen=3, showgrid=False,
                              tickcolor=INK)},
)

fig.update_layout(
    template='plotly_white',
    font=dict(family='Arial', color=INK, size=18),
    margin=dict(l=95, r=10, t=25, b=60),
    width=700, height=500,
    barmode='group', bargap=0.15, bargroupgap=0,
    legend=dict(x=0.99, y=0.99, xanchor='right', yanchor='top',
                font=dict(weight='bold', size=14)),
)

png_path = OUT_DIR / "fig_3d.png"
svg_path = OUT_DIR / "fig_3d.svg"
fig.write_image(str(png_path), scale=4)
print(f"Saved {png_path}")
try:
    fig.write_image(str(svg_path))
    print(f"Saved {svg_path}")
except Exception as e:
    print(f"SVG export skipped ({e}). Install kaleido: pip install kaleido")
