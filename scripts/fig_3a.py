"""Fig 3a — CP1 OER overpotential vs time, per catalyst.

Reads the shared `data/fig_3a-d.parquet`; writes `fig_3a.{png,svg}` to `figures/`.
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

# One colour per catalyst, in dataset row order: front-runners first, then
# Pd/Ru benchmarks.
COLORS = [
    '#D7AA54',  # InMnPdOx
    '#715CA1',  # NiTaPdOx
    '#8B3D2B',  # CoFeZrOx
    '#458871',  # PdOx
    '#E26F46',  # RuOx
]


def fmt_system(name: str) -> str:
    """'InMnPdOx' -> 'InMnPdO<sub>x</sub>' for HTML-rendered labels."""
    return re.sub(r'Ox$', 'O<sub>x</sub>', str(name))


print(f'Loaded {len(df)} rows × {len(df.columns)} columns from {DATA_PATH}')

# Paper-coord band reserved on the right of the plot area so the per-trace
# endpoint labels sit outside the axis frame instead of a legend box.
TRACE_LABEL_BAND = 0.20
LABEL_X = 1.0 - TRACE_LABEL_BAND + 0.01

AX_KW = dict(mirror=False, linecolor=INK, linewidth=1,
             ticks='outside', ticklen=5, tickcolor=INK,
             minor=dict(ticks='outside', ticklen=3, nticks=3, tickcolor=INK))

Y_RANGE = (0.37, 0.57)

fig = go.Figure()

for (_, r), color in zip(df.iterrows(), COLORS):
    fig.add_trace(go.Scatter(
        x=r['CP1 Time (s)'], y=r['CP1 Overpotential (V)'],
        mode='lines', name=r['System'],
        line=dict(color=color, width=3),
        showlegend=False,
    ))

fig.update_xaxes(title_text='<b>Time (s)</b>', range=[-0.1, 600],
                 domain=[0.0, 1.0 - TRACE_LABEL_BAND],
                 showgrid=False, **AX_KW)
fig.update_yaxes(title_text='<b>OER Overpotential (V)</b>', range=list(Y_RANGE),
                 showgrid=False, **AX_KW)

# Direct trace labels at each curve's last in-range point.
for trace in fig.data:
    y_arr = np.asarray(trace.y, dtype=float)
    in_range = np.isfinite(y_arr) & (y_arr >= Y_RANGE[0]) & (y_arr <= Y_RANGE[1] - 0.01)
    idx = np.where(in_range)[0]
    if idx.size == 0:
        continue
    fig.add_annotation(
        text=f'<b>{fmt_system(trace.name)}</b>',
        x=LABEL_X, y=float(y_arr[idx[-1]]),
        xref='paper', yref='y',
        xanchor='left', yanchor='middle',
        showarrow=False, font=dict(size=17, color=trace.line.color),
    )

fig.update_layout(
    template='plotly_white',
    font=dict(family='Arial', color=INK, size=18),
    margin=dict(l=95, r=10, t=25, b=80),
    width=700, height=500,
)

png_path = OUT_DIR / "fig_3a.png"
svg_path = OUT_DIR / "fig_3a.svg"
fig.write_image(str(png_path), scale=4)
print(f"Saved {png_path}")
try:
    fig.write_image(str(svg_path))
    print(f"Saved {svg_path}")
except Exception as e:
    print(f"SVG export skipped ({e}). Install kaleido: pip install kaleido")
