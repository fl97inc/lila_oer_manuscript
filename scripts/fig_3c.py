"""Fig 3c — median OER overpotential before vs after AST, per catalyst.

Reads the shared `data/fig_3a-d.parquet`; writes `fig_3c.{png,svg}` to `figures/`.
"""
from pathlib import Path
import re

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
# Pd/Ru benchmarks. The light shade marks the post-AST bar of each pair.
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


def last_pct_median(arr, pct: float = 0.10) -> float:
    """Median over the last `pct` fraction of `arr`. NaN-safe."""
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return np.nan
    tail = a[-max(1, int(np.ceil(a.size * pct))):]
    tail = tail[np.isfinite(tail)]
    if tail.size == 0:
        return np.nan
    return float(np.median(tail))


print(f'Loaded {len(df)} rows × {len(df.columns)} columns from {DATA_PATH}')

AX_KW = dict(mirror=False, linecolor=INK, linewidth=1,
             ticks='outside', ticklen=5, tickcolor=INK,
             minor=dict(ticks='outside', ticklen=3, nticks=3, tickcolor=INK))

cp1_med = df['CP1 Overpotential (V)'].apply(last_pct_median).to_numpy()
cp2_med = df['CP2 Overpotential (V)'].apply(last_pct_median).to_numpy()
categories = df['System'].tolist()

fig = go.Figure()

fig.add_trace(go.Bar(
    x=categories, y=cp1_med,
    marker_color=COLORS,
    showlegend=False, offsetgroup='before',
))
fig.add_trace(go.Bar(
    x=categories, y=cp2_med,
    marker_color=COLORS_LIGHT,
    showlegend=False, offsetgroup='after',
))
fig.add_trace(go.Bar(x=[None], y=[None], name='Before AST',
                     marker_color=LEGEND_BEFORE, offsetgroup='before'))
fig.add_trace(go.Bar(x=[None], y=[None], name='After AST',
                     marker_color=LEGEND_AFTER, offsetgroup='after'))

fig.update_xaxes(
    showgrid=False, tickmode='array',
    tickvals=list(range(len(categories))),
    ticktext=[fmt_system(c) for c in categories],
    tickangle=0,
    mirror=False, linecolor=INK, linewidth=1, ticks='', zeroline=False,
)
fig.update_yaxes(title_text='<b>OER Overpotential (V)</b>', range=[0.37, 0.57],
                 showgrid=False, **AX_KW)

fig.update_layout(
    template='plotly_white',
    font=dict(family='Arial', color=INK, size=18),
    margin=dict(l=95, r=10, t=25, b=60),
    width=700, height=500,
    barmode='group', bargap=0.15, bargroupgap=0,
    legend=dict(x=0.99, y=0.99, xanchor='right', yanchor='top',
                font=dict(weight='bold', size=14)),
)

png_path = OUT_DIR / "fig_3c.png"
svg_path = OUT_DIR / "fig_3c.svg"
fig.write_image(str(png_path), scale=4)
print(f"Saved {png_path}")
try:
    fig.write_image(str(svg_path))
    print(f"Saved {svg_path}")
except Exception as e:
    print(f"SVG export skipped ({e}). Install kaleido: pip install kaleido")
