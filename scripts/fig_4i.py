"""Fig 4i — Bruker XRD waterfall (Pt/Si / PdOx / InMnPdOx / NiTaPdOx), before/after test.

Reads the shared `data/fig_4i.csv`; writes `fig_4i.{png,svg}` to `figures/`.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

# --- Load merged CSV into data_dict + define plot styling ---
MERGED_CSV = DATA_DIR / "fig_4i.csv"

_merged = pd.read_csv(MERGED_CSV)
data_dict = {
    sample: (g["two_theta"].to_numpy(), g["intensity"].to_numpy())
    for sample, g in _merged.groupby("sample")
}
print("Loaded samples:", list(data_dict.keys()))

# Phase-marker toggles (only Pt + PdO are used by this figure)
SHOW_PHASES = {'Pt': True, 'PdO': True, 'RuO2': True, 'Pd': False, 'PdO2': False}

phase_markers = {
    'Pt':  {'marker': 'o', 'color': '#005272', 'size': 8},   # Pt PDF
    'PdO': {'marker': 'v', 'color': '#231F20', 'size': 10},  # PdO PDF
}

# Color scheme
INK = '#231F20'                                 # axis labels, ticks, PdO labels
ptsi_color = '#8A8A8A'                          # Pt/Si substrate
pdox_pre_color = '#458871'                      # as-deposited PdOx
pdox_post_color = '#A4C2B6'                     # post-test PdOx
inmnpd_pre_color = '#D7AA54'                    # as-deposited InMnPdOx
inmnpd_post_color = '#EDD4AD'                   # post-test InMnPdOx
nipdta_pre_color = '#715CA1'                    # as-deposited NiTaPdOx
nipdta_post_color = '#B7ABCD'                   # post-test NiTaPdOx

# --- Waterfall figure from the merged CSV -> SVG ---
# Standalone version of panel (b): Pt/Si / PdOx / InMnPdOx / NiTaPdOx waterfall.

# Use Arial for all text (incl. mathtext subscripts)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'stixsans'

# Frame sized to the requested 2028.15 x 583 px ratio (~3.48:1)
fig_b, ax_b = plt.subplots(figsize=(2028.15 / 100, 583 / 100))
# Measured from panel I snapshot: leave room for axis titles and a top-right legend
# sitting above the top curve (not overlapping it).
fig_b.subplots_adjust(left=0.062, right=0.995, bottom=0.145, top=0.98)

# Vertical offsets measured from panel I at 2θ ≈ 28° (pixel gaps 23 / 33 / 23 / 22 / 23 / 22).
# Most adjacent baselines are ~22.5 px apart; the as-deposited→post-test PdOx gap is 33 px
# so PdO(hkl) labels fit between those two curves.
_GAP_PX = 22.5
_LEVEL_UNIT = 1.55
_level_after = {
    'PtSi':         _LEVEL_UNIT * (23 / _GAP_PX),
    'PdOx_pre':     _LEVEL_UNIT * (33 / _GAP_PX),
    'PdOx_post':    _LEVEL_UNIT * (23 / _GAP_PX),
    'InMnPdOx_pre': _LEVEL_UNIT * (22 / _GAP_PX),
    'InMnPdOx_post':_LEVEL_UNIT * (23 / _GAP_PX),
    'NiPdTaOx_pre': _LEVEL_UNIT * (22 / _GAP_PX),
}

# Curve stacking order (bottom -> top).
b_order = [
    ('PtSi',          'Pt/Si substrate',            ptsi_color,         None, 1.0),
    ('PdOx_pre',      r'As-deposited PdO$_x$',       pdox_pre_color,     None, 1.0),
    ('PdOx_post',     r'Post-test PdO$_x$',          pdox_post_color,    None, 1.0),
    ('InMnPdOx_pre',  r'As-deposited InMnPdO$_x$',   inmnpd_pre_color,   None, 1.0),
    ('InMnPdOx_post', r'Post-test InMnPdO$_x$',      inmnpd_post_color,  None, 1.0),
    ('NiPdTaOx_pre',  r'As-deposited NiTaPdO$_x$',   nipdta_pre_color,   None, 1.0),
    ('NiPdTaOx_post', r'Post-test NiTaPdO$_x$',      nipdta_post_color,  None, 1.0),
]

# Curve labels sit just above the left baseline (4 pt), matching panel I.
label_x = 21.0

b_curves = []
level = 0.0
for key, label, color, ls, alpha in b_order:
    if key not in data_dict:
        continue
    tt, inten = data_dict[key]
    y_offset = np.maximum(inten, 1e-6) * 10 ** level
    ax_b.plot(tt, y_offset, lw=1.8, color=color, linestyle=ls if ls else '-',
              label='_nolegend_', alpha=alpha)
    j0 = np.argmin(np.abs(tt - label_x))
    ax_b.annotate(
        label, xy=(label_x, y_offset[j0]), xytext=(0, 4),
        textcoords='offset points', fontsize=18, fontweight='bold',
        color=color, alpha=alpha, ha='left', va='bottom', zorder=12)
    b_curves.append({'tt': tt, 'y_offset': y_offset, 'pre': ls is None, 'key': key})
    level += _level_after.get(key, _LEVEL_UNIT)

# PdO 00-041-1107 inverted triangles on as-deposited PdOx and InMnPdOx only.
# Text labels only on as-deposited PdOx.
if SHOW_PHASES.get('PdO', True):
    pdo_peaks = [(33.84, '(101)'),
                 (45.138, '(102)'),
                 (54.72, '(112)'),
                 (60.22, '(103)')]
    pdo_style = phase_markers['PdO']
    pdo_keys = {'PdOx_pre', 'InMnPdOx_pre'}
    pdo_idxs = [i for i, c in enumerate(b_curves) if c['key'] in pdo_keys]
    label_idx = next((i for i, c in enumerate(b_curves) if c['key'] == 'PdOx_pre'), None)
    for ci in pdo_idxs:
        curve = b_curves[ci]
        for peak_2theta, hkl in pdo_peaks:
            j = np.argmin(np.abs(curve['tt'] - peak_2theta))
            marker_y = curve['y_offset'][j] * 2.0
            ax_b.plot(peak_2theta, marker_y, marker=pdo_style['marker'],
                      color=pdo_style['color'], markersize=16,
                      markeredgewidth=0, zorder=15)
            if ci == label_idx:
                ax_b.annotate(
                    f'PdO{hkl}', xy=(peak_2theta, marker_y),
                    xytext=(0, 11), textcoords='offset points',
                    fontsize=18, ha='center', va='bottom',
                    color=pdo_style['color'], fontweight='normal', zorder=16)

# Pt(111) markers on every curve; text label only on the topmost curve
if SHOW_PHASES['Pt']:
    pt_peak = 39.59
    ms = phase_markers['Pt']
    for i, curve in enumerate(b_curves):
        tt = curve['tt']
        if np.min(np.abs(tt - pt_peak)) > 0.5:
            continue
        j = np.argmin(np.abs(tt - pt_peak))
        marker_y = curve['y_offset'][j] * 1.8
        ax_b.plot(pt_peak, marker_y, marker=ms['marker'], color=ms['color'],
                  markersize=ms['size'] + 3, markeredgewidth=0, zorder=10)
        if i == len(b_curves) - 1:
            ax_b.annotate(
                "Pt(111)", xy=(pt_peak, marker_y),
                xytext=(0, 6), textcoords='offset points',
                fontsize=16, ha='center', va='bottom',
                color=ms['color'], fontweight='normal', zorder=11)

ax_b.set_yscale('log')
ax_b.set_xlabel("2θ (degrees)", fontsize=16, fontweight='bold', color=INK, labelpad=4)
ax_b.set_ylabel("Intensity (a.u., log)", fontsize=16, fontweight='bold', color=INK, labelpad=10)
ax_b.set_xlim(20, 62.5)
ax_b.tick_params(axis='x', labelsize=16, colors=INK, width=1.2, pad=3, length=4)
ax_b.yaxis.set_major_locator(LogLocator(base=10.0))
ax_b.yaxis.set_minor_locator(plt.NullLocator())
ax_b.tick_params(axis='y', which='both', left=False, labelleft=False, colors=INK)
for spine in ('bottom', 'left'):
    ax_b.spines[spine].set_color(INK)
    ax_b.spines[spine].set_linewidth(1.2)
# Headroom so the legend sits above the top curve (snapshot: ~17 px / 164 px of data span)
ax_b.set_ylim(top=ax_b.get_ylim()[1] * 2.15)

# Keep only the PDF reference entries in the boxed legend
b_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#005272',
           markersize=12, markeredgecolor='none', linestyle='None',
           label='Pt PDF 00-004-0802'),
    Line2D([0], [0], marker='v', color='w', markerfacecolor='#231F20',
           markersize=16, markeredgecolor='none', linestyle='None',
           label='PdO PDF 00-041-1107'),
]
b_leg = ax_b.legend(handles=b_handles, fontsize=18, loc='upper right',
                    bbox_to_anchor=(1.0, 1.02), frameon=False,
                    handletextpad=0.4, labelspacing=0.22,
                    borderaxespad=0.0, borderpad=0.0)
for txt, c in zip(b_leg.get_texts(), ['#005272', '#231F20']):
    txt.set_color(c)
    txt.set_fontweight('normal')
ax_b.grid(False)

ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)

# No bbox='tight' crop so the saved aspect ratio stays exactly 2028.15:583.
svg_path = OUT_DIR / "fig_4i.svg"
fig_b.savefig(svg_path, format='svg')
fig_b.savefig(svg_path.with_suffix('.png'), dpi=300, format='png')
print(f"Saved: {svg_path}")
print(f"Saved: {svg_path.with_suffix('.png')}")
