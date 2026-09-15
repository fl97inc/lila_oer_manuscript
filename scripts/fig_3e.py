"""Fig 3e — iR-corrected long-term HCell OER overpotential vs time, per catalyst.

Reads `data/fig_3e.csv`; writes `fig_3e.{png,svg}` to `figures/`.
"""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

# ── Load the plot data ─────────────────────────────────────────────────────
# Fig 3E: iR-corrected long-term HCell overpotential.
CSV_PATH = DATA_DIR / "fig_3e.csv"
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows from {CSV_PATH.name}")
print(f"  material systems: {sorted(df['material_system'].unique())}")
print(f"  blue-marker dots: {int(df['has_blue_marker'].sum())} / {len(df)}")

# ── Column names in the source CSV ─────────────────────────────────────────
X_COL = 'plot_x_hr'
Y_COL = 'overpotential_corrected_avg'

# iR y-title is broken across two lines because "iR-corrected OER
# Overpotential (V)" (33 chars) does not fit as one rotated line at
# 26 pt bold on a ~5.2-in-tall canvas.
Y_AXIS_TITLE = 'iR-corrected\nOER Overpotential (V)'

# ── Color palette (one per material system) ────────────────────────────────
# Hex codes from the manuscript brand palette (Harley review, 2026-09-14;
# PdOx corrected 2026-09-15):
#     InMnPdOx  → D7AA54   (warm gold)
#     NiTaPdOx  → 715CA1   (muted purple)
#     PdOx      → 458871   (teal green)
#     RuOx      → E26F46   (burnt orange)
COLOR_BY_MATERIAL_SYSTEM = {
    'InMnPdOx': '#D7AA54',
    'NiTaPdOx': '#715CA1',
    'PdOx':     '#458871',
    'RuOx':     '#E26F46',
}
TRACE_ORDER = ['InMnPdOx', 'NiTaPdOx', 'PdOx', 'RuOx']

# Labels rendered with a proper subscript-x (Ken review, 2026-09-11:
# "change Ox to O_x subscript").  Data-side keys stay ASCII ('InMnPdOx' etc.)
# so the CSV schema and any downstream joins are untouched.
DISPLAY_LABEL_BY_MATERIAL_SYSTEM = {
    'InMnPdOx': r'InMnPdO$_{\mathrm{x}}$',
    'NiTaPdOx': r'NiTaPdO$_{\mathrm{x}}$',
    'PdOx':     r'PdO$_{\mathrm{x}}$',
    'RuOx':     r'RuO$_{\mathrm{x}}$',
}

# Event-circle styling.  Harley's 2026-09-14 review asked for white fill,
# stroke color = the composition color, thicker stroke; on 2026-09-18 the fill
# was relaxed to transparent so that when an event ring lands within a
# dot-radius of another material's dot (e.g. the InMnPdOx event at
# (659.5 hr, 0.485 V) sitting 2 mV from the RuOx endpoint at (659.5, 0.487)),
# the underlying dot still shows through instead of getting hidden by the
# ring's fill.  The thick colored stroke keeps it reading as an open circle.
EVENT_CIRCLE_FILL       = 'none'            # transparent (was 'white')
EVENT_CIRCLE_EDGE_WIDTH = 3.0
EVENT_CIRCLE_SIZE_PT    = 14                # outer diameter in points

# Axis labels / ranges.
X_LABEL = 'Time (hr)'
X_RANGE = (0, 1005)
Y_RANGE = (0.3, 0.7)

# Figure size + DPI — parameterized off the master's 2000 × 500 px PNG output.
#
# IMPORTANT — Word / PowerPoint compute the physical insert size as
# (pixels ÷ DPI metadata). Plotly writes NO DPI metadata, so Word defaults to
# 96 DPI and displays the master at 2000/96 × 500/96 ≈ 20.83 × 5.21 in.
# Setting DPI_RASTER = 96 makes Word render this figure at the exact same
# physical size.
TARGET_PNG_W  = 2000
TARGET_PNG_H  = 500
DPI_RASTER    = 96                          # matches Word's default assumption

# Font sizes (pt) — match the master Plotly numbers literally
# (title 26 / tick 22 / legend 18).  Titles stay bold.
AXIS_TITLE_FONTSIZE = 26
TICK_LABEL_FONTSIZE = 22
LEGEND_FONTSIZE     = 18

# Marker area (pt²).
SCATTER_MARKER_SIZE = 80

# Tick spacing — match the master's tick placement deterministically:
# majors at 0/200/…/1000, minors every 50 hr.
X_MAJOR_TICK_HR = 200
X_MINOR_TICK_HR = 50
Y_MAJOR_TICK_V = 0.1
Y_MINOR_TICK_V = 0.1

# ── Build the figure ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 6.5))

_FALLBACK_COLOR = (0.3, 0.3, 0.3)

for ms in TRACE_ORDER:
    # Event CPs are excluded from the scatter and re-drawn as hollow rings
    # in the block below, so the ring stays a clean open circle (no dot in
    # its center) — Ken review, 2026-09-18.
    g = df.loc[(df['material_system'] == ms) & (~df['has_blue_marker'])].sort_values(X_COL)
    if g.empty:
        continue
    color = COLOR_BY_MATERIAL_SYSTEM.get(ms, _FALLBACK_COLOR)
    ax.scatter(
        g[X_COL], g[Y_COL],
        s=SCATTER_MARKER_SIZE,
        color=color,
        edgecolors=color,
        linewidths=0.8,
        zorder=2,
    )

# ── Event-circle overlay for CPs flagged in the source pipeline ────────────
# Transparent fill, stroke color = the material's own trace color, thicker
# outline.  We loop per material system so every ring gets its own edge color
# (a single ax.plot() call can't assign per-point edgecolors).
_flagged_total = 0
for ms in TRACE_ORDER:
    flagged_ms = df[(df['material_system'] == ms) & df['has_blue_marker']]
    if flagged_ms.empty:
        continue
    edge = COLOR_BY_MATERIAL_SYSTEM.get(ms, _FALLBACK_COLOR)
    ax.plot(
        flagged_ms[X_COL],
        flagged_ms[Y_COL],
        linestyle='None',
        marker='o',
        markersize=EVENT_CIRCLE_SIZE_PT,
        markerfacecolor=EVENT_CIRCLE_FILL,
        markeredgecolor=edge,
        markeredgewidth=EVENT_CIRCLE_EDGE_WIDTH,
        zorder=3,                         # above the material-system dots
    )
    _flagged_total += len(flagged_ms)
if _flagged_total:
    print(f'Overlaid {_flagged_total} event ring(s), one color per material system.')

# ── Axes: slide-style, ticks outside, no top/right spines ─────────────────
ax.set_xlabel(X_LABEL,      fontsize=AXIS_TITLE_FONTSIZE, fontfamily='Arial', fontweight='bold', color='black', labelpad=6)
ax.set_ylabel(Y_AXIS_TITLE, fontsize=AXIS_TITLE_FONTSIZE, fontfamily='Arial', fontweight='bold', color='black', labelpad=8)
ax.set_xlim(*X_RANGE)
ax.set_ylim(*Y_RANGE)

ax.tick_params(axis='both', which='major', direction='out',
               length=6, width=1, colors='black', labelsize=TICK_LABEL_FONTSIZE)
ax.tick_params(axis='both', which='minor', direction='out',
               length=3, width=1, colors='black')
ax.xaxis.set_major_locator(MultipleLocator(X_MAJOR_TICK_HR))
ax.xaxis.set_minor_locator(MultipleLocator(X_MINOR_TICK_HR))
ax.yaxis.set_major_locator(MultipleLocator(Y_MAJOR_TICK_V))
ax.yaxis.set_minor_locator(MultipleLocator(Y_MINOR_TICK_V))
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontfamily('Arial')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)

# ── Direct labels next to each trace (per 2026-09-18 review) ────────────
# Instead of a corner legend box, each material system gets a bold in-plot
# label placed near a distinctive point of its own trace, colored to match.
# Positions are (x_hr, y_V) in data coordinates, hand-picked to sit just
# above / to-the-right of each trace's endpoint or peak so they don't collide
# with the dots.  If a label ever overlaps a dot after a data update, tweak
# the (x, y) below.
DIRECT_LABEL_XY = {
    'NiTaPdOx': (580, 0.700),   # above the sharp peak at ~575 hr
    'PdOx':     (830, 0.610),   # above the trace endpoint
    'InMnPdOx': (950, 0.500),   # above the yellow trace tail
    'RuOx':     (660, 0.500),   # above the trace endpoint
}
for ms, (lx, ly) in DIRECT_LABEL_XY.items():
    ax.text(
        lx, ly,
        DISPLAY_LABEL_BY_MATERIAL_SYSTEM.get(ms, ms),
        fontsize=LEGEND_FONTSIZE,
        color=COLOR_BY_MATERIAL_SYSTEM.get(ms, _FALLBACK_COLOR),
        fontfamily='Arial',
        fontweight='bold',
        ha='left',
        va='bottom',
        zorder=4,
    )

# ── Explicit margins so nothing gets cropped ────────────────────────────
# left is generous to give the rotated y-axis title room; bottom is 0.20 so
# the 26 pt "Time (hr)" title clears the tick labels.
fig.subplots_adjust(left=0.11, right=0.985, top=0.965, bottom=0.20)

# ── Output (PNG / SVG) ─────────────────────────────────────────────────────
png_path = OUT_DIR / "fig_3e.png"
svg_path = OUT_DIR / "fig_3e.svg"

fig.savefig(png_path, dpi=DPI_RASTER)
fig.savefig(svg_path)
print(f"Saved {png_path} ({DPI_RASTER} dpi)")
print(f"Saved {svg_path}")
