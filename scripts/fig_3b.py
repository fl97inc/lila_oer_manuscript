"""Fig 3b — LSV (anodic, 2nd cycle) current density vs potential, per catalyst.

Reads the shared `data/fig_3a-d.parquet`; writes `fig_3b.{png,svg}` to `figures/`.
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

Y_RANGE = (-10, 105)

# Manual label y-positions (top -> bottom). PdOx, NiTaPdOx and InMnPdOx end
# within ~6 mA cm^-2 of each other, so their natural endpoints would collide.
LABEL_Y = {
    'RuOx':     102,
    'PdOx':      83,
    'NiTaPdOx':  74,
    'InMnPdOx':  65,
    'CoFeZrOx':  46,
}

fig = go.Figure()

for (_, r), color in zip(df.iterrows(), COLORS):
    fig.add_trace(go.Scatter(
        x=r['LSV Anodic Vrhe'], y=r['LSV Anodic Current Density (mA/cm²)'],
        mode='lines', name=r['System'],
        line=dict(color=color, width=3),
        showlegend=False,
    ))

fig.update_xaxes(title_text='<b>E (V vs RHE)</b>',
                 domain=[0.0, 1.0 - TRACE_LABEL_BAND],
                 showgrid=False, **AX_KW)
fig.update_yaxes(title_text='<b>Current Density (mA cm<sup>-2</sup>)</b>',
                 range=list(Y_RANGE),
                 showgrid=False, zeroline=False, **AX_KW)

# Direct trace labels to the right of the axis frame.
for trace in fig.data:
    name = (trace.name or '').strip()
    if name in LABEL_Y:
        y_pos = LABEL_Y[name]
    else:
        y_arr = np.asarray(trace.y, dtype=float)
        in_range = np.isfinite(y_arr) & (y_arr >= Y_RANGE[0]) & (y_arr <= Y_RANGE[1])
        idx = np.where(in_range)[0]
        if idx.size == 0:
            continue
        y_pos = float(y_arr[idx[-1]])
    fig.add_annotation(
        text=f'<b>{fmt_system(name)}</b>',
        x=LABEL_X, y=y_pos,
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

png_path = OUT_DIR / "fig_3b.png"
svg_path = OUT_DIR / "fig_3b.svg"
fig.write_image(str(png_path), scale=4)
print(f"Saved {png_path}")
try:
    fig.write_image(str(svg_path))
    print(f"Saved {svg_path}")
except Exception as e:
    print(f"SVG export skipped ({e}). Install kaleido: pip install kaleido")
