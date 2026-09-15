import os
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "oer_exp_data.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CANVAS_WIDTH = 2100
PUBLISHED_ASPECT = 4.11
VERTICAL_SQUISH = 0.90
PLOT_ASPECT = PUBLISHED_ASPECT / VERTICAL_SQUISH
MARGIN = dict(l=156, r=40, t=32, b=148)
PLOT_WIDTH = CANVAS_WIDTH - MARGIN["l"] - MARGIN["r"]
PLOT_HEIGHT = round(PLOT_WIDTH / PLOT_ASPECT)
CANVAS_HEIGHT = PLOT_HEIGHT + MARGIN["t"] + MARGIN["b"]
EXPORT_SCALE = 2
PNG_DPI = 600

FONT_FAMILY = "Arial"
CAP_HALF_EM = 0.355

AXIS_TITLE_FONT_SIZE = 31
TICK_FONT_SIZE = 29
LABEL_FONT_SIZE = 31
BODY_FONT_SIZE = TICK_FONT_SIZE

COLOR_PARETO_ALL = "#AD8536"
COLOR_PARETO_NOPD = "#231F20"
COLOR_CLOUD = "#B0B0B0"
COLOR_AXIS = "#231F20"
CLOUD_OPACITY = 0.35

CLOUD_MARKER_SIZE = 14
PARETO_MARKER_SIZE = 18
NOPD_MARKER_SIZE = 16
MARKER_EDGE_WIDTH = 2.4
PARETO_LINE_WIDTH = 4.4
NOPD_LINE_WIDTH = 3.6
AXIS_LINE_WIDTH = 2.4
TICK_LEN = 12

X_COL = "overpotential_cp1_600s_mean"
Y_COL = "xrf_total_molar_change"
X_RANGE = [0.4, 0.9]
Y_RANGE = [-0.1, 1.2]
X_DTICK = 0.1
Y_DTICK = 0.4

LABEL_ALL_TEXT = "Pareto (all)"
LABEL_NOPD_TEXT = "Pareto (without Pd)"
LABEL_ALL_DY = 0.12
LABEL_NOPD_DY = 0.10


def pareto_front_min_min(x, y):
    points = np.column_stack([x, -y])
    order = np.lexsort((points[:, 1], points[:, 0]))
    sorted_pts = points[order]
    pareto_idx = [order[0]]
    best_y = sorted_pts[0, 1]
    for i in range(1, len(sorted_pts)):
        if sorted_pts[i, 1] > best_y:
            pareto_idx.append(order[i])
            best_y = sorted_pts[i, 1]
    return np.array(pareto_idx)


STRIP_BASELINE = re.compile(
    r'\s*(?:dominant|alignment)-baseline\s*[:=]\s*"?[a-z-]+"?;?'
)


def sanitize_svg_for_figma(path, font_family=FONT_FAMILY):
    svg = open(path, encoding="utf-8").read()

    def font_size_of(tag, default=LABEL_FONT_SIZE):
        m = re.search(r'font-size[:=]"?\s*([\d.]+)', tag)
        return float(m.group(1)) if m else default

    def fix(match):
        tag = match.group(0)
        mode = re.search(r'(?:dominant|alignment)-baseline[:=]"?\s*([a-z-]+)', tag)
        y = re.search(r'\by="([-\d.]+)"', tag)
        if not (mode and y):
            return tag
        fs = font_size_of(tag)
        shift = {
            "middle": CAP_HALF_EM,
            "central": CAP_HALF_EM,
            "hanging": 2 * CAP_HALF_EM,
            "text-before-edge": 2 * CAP_HALF_EM,
        }
        if mode.group(1) in shift:
            new_y = float(y.group(1)) + shift[mode.group(1)] * fs
            tag = re.sub(r'\by="[-\d.]+"', f'y="{new_y:.2f}"', tag, count=1)
        return STRIP_BASELINE.sub("", tag)

    svg = re.sub(r"<text\b[^>]*>", fix, svg)
    svg = re.sub(r'font-family:[^;"]*', f"font-family:{font_family}", svg)
    svg = re.sub(r'font-family="[^"]*"', f'font-family="{font_family}"', svg)
    open(path, "w", encoding="utf-8").write(svg)
    return svg.count("dominant-baseline") + svg.count("alignment-baseline")


def stamp_png_dpi(path, dpi=PNG_DPI):
    from PIL import Image

    with Image.open(path) as image:
        image.save(path, dpi=(dpi, dpi))


def main():
    df = pd.read_csv(DATA_PATH)

    base_df = df.copy()
    non_pgm_df = base_df[
        ~base_df["target_elements"].str.contains("Pd", na=False)
    ].copy()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=base_df[X_COL],
            y=base_df[Y_COL],
            mode="markers",
            name="All samples",
            marker=dict(
                size=CLOUD_MARKER_SIZE,
                color=COLOR_CLOUD,
                symbol="circle",
                opacity=CLOUD_OPACITY,
                line=dict(width=0),
            ),
            customdata=base_df[["composition", "target_elements"]].values,
            hovertemplate=(
                "Overpotential: %{x:.4f} V<br>"
                "Total Molar Loss: %{y:.4f} μmol/cm²<br>"
                "Composition: %{customdata[0]}<br>"
                "Elements: %{customdata[1]}"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

    x_all = base_df[X_COL].values
    y_all = base_df[Y_COL].values
    pidx_all = pareto_front_min_min(x_all, y_all)
    pidx_all_sorted = pidx_all[np.argsort(x_all[pidx_all])]

    fig.add_trace(
        go.Scatter(
            x=x_all[pidx_all_sorted],
            y=y_all[pidx_all_sorted],
            mode="lines",
            line=dict(color=COLOR_PARETO_ALL, width=PARETO_LINE_WIDTH),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_all[pidx_all_sorted],
            y=y_all[pidx_all_sorted],
            mode="markers",
            name="Pareto (all)",
            marker=dict(
                size=PARETO_MARKER_SIZE,
                color=COLOR_PARETO_ALL,
                symbol="diamond",
                line=dict(width=MARKER_EDGE_WIDTH, color="#FFFFFF"),
            ),
            customdata=base_df.iloc[pidx_all_sorted][
                ["composition", "target_elements"]
            ].values,
            hovertemplate=(
                "<b>Pareto (all)</b><br>"
                "Overpotential: %{x:.4f} V<br>"
                "Total Molar Loss: %{y:.4f} μmol/cm²<br>"
                "Composition: %{customdata[0]}<br>"
                "Elements: %{customdata[1]}"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

    x_np = non_pgm_df[X_COL].values
    y_np = non_pgm_df[Y_COL].values
    pidx_np = pareto_front_min_min(x_np, y_np)
    pidx_np_sorted = pidx_np[np.argsort(x_np[pidx_np])]

    fig.add_trace(
        go.Scatter(
            x=x_np[pidx_np_sorted],
            y=y_np[pidx_np_sorted],
            mode="lines",
            line=dict(color=COLOR_PARETO_NOPD, width=NOPD_LINE_WIDTH, dash="dash"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    all_pareto_set = set(pidx_all)
    non_pgm_global_idx = non_pgm_df.index.values
    honorable_mask = np.array(
        [non_pgm_global_idx[i] not in all_pareto_set for i in pidx_np_sorted]
    )
    honorable_idx = pidx_np_sorted[honorable_mask]

    if len(honorable_idx) > 0:
        fig.add_trace(
            go.Scatter(
                x=x_np[honorable_idx],
                y=y_np[honorable_idx],
                mode="markers",
                name="Pareto (without Pd)",
                marker=dict(
                    size=NOPD_MARKER_SIZE,
                    color=COLOR_PARETO_NOPD,
                    symbol="square",
                    line=dict(width=MARKER_EDGE_WIDTH, color="#FFFFFF"),
                ),
                customdata=non_pgm_df.iloc[honorable_idx][
                    ["composition", "target_elements"]
                ].values,
                hovertemplate=(
                    "<b>Pareto (without Pd)</b><br>"
                    "Overpotential: %{x:.4f} V<br>"
                    "Total Molar Loss: %{y:.4f} μmol/cm²<br>"
                    "Composition: %{customdata[0]}<br>"
                    "Elements: %{customdata[1]}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    label_anchors = [
        (
            LABEL_ALL_TEXT,
            COLOR_PARETO_ALL,
            LABEL_ALL_DY,
            x_all[pidx_all_sorted[0]],
            y_all[pidx_all_sorted[0]],
        ),
    ]
    if len(pidx_np_sorted) > 0:
        label_anchors.append(
            (
                LABEL_NOPD_TEXT,
                COLOR_PARETO_NOPD,
                LABEL_NOPD_DY,
                x_np[pidx_np_sorted[0]],
                y_np[pidx_np_sorted[0]],
            )
        )
    for text, color, dy, ax, ay in label_anchors:
        fig.add_annotation(
            x=float(ax),
            y=min(float(ay) + dy, Y_RANGE[1] - 0.04),
            text=f"<b>{text}</b>",
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            font=dict(family=FONT_FAMILY, size=LABEL_FONT_SIZE, color=color),
        )
        print(f"  Label '{text}' anchored at ({float(ax):.3f}, {float(ay) + dy:.3f})")

    clean_axis = dict(
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor=COLOR_AXIS,
        linewidth=AXIS_LINE_WIDTH,
        mirror=False,
        ticks="outside",
        ticklen=TICK_LEN,
        tickwidth=AXIS_LINE_WIDTH,
        tickcolor=COLOR_AXIS,
    )

    fig.update_layout(
        width=CANVAS_WIDTH,
        height=CANVAS_HEIGHT,
        margin=MARGIN,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color=COLOR_AXIS, size=BODY_FONT_SIZE, family=FONT_FAMILY),
        showlegend=False,
    )
    fig.update_xaxes(
        title=dict(
            text="<b>Overpotential @10min (V)</b>",
            font=dict(size=AXIS_TITLE_FONT_SIZE, color=COLOR_AXIS),
        ),
        range=X_RANGE,
        dtick=X_DTICK,
        tickfont=dict(size=TICK_FONT_SIZE, color=COLOR_AXIS),
        **clean_axis,
    )
    fig.update_yaxes(
        title=dict(
            text="<b>Total Molar Loss (μmol cm<sup>-2</sup>)</b>",
            font=dict(size=AXIS_TITLE_FONT_SIZE, color=COLOR_AXIS),
        ),
        range=Y_RANGE,
        dtick=Y_DTICK,
        tickfont=dict(size=TICK_FONT_SIZE, color=COLOR_AXIS),
        **clean_axis,
    )

    png_path = os.path.join(OUTPUT_DIR, "fig_2c.png")
    svg_path = os.path.join(OUTPUT_DIR, "fig_2c.svg")

    fig.write_image(png_path, scale=EXPORT_SCALE)
    stamp_png_dpi(png_path)
    print(f"Saved {png_path}")

    try:
        fig.write_image(svg_path)
        leftover = sanitize_svg_for_figma(svg_path)
        print(f"Saved {svg_path} (baseline attributes remaining: {leftover})")
    except Exception as e:
        print(f"SVG export skipped ({e}). Install kaleido: pip install kaleido")

    print(f"\nTotal samples: {len(base_df)}")
    print(f"Pareto (all): {len(pidx_all)} points")
    print(f"Pareto (without Pd, exclusive): {len(honorable_idx)} points")
    print(
        f"  Canvas: {CANVAS_WIDTH} x {CANVAS_HEIGHT} at {EXPORT_SCALE}x "
        f"= {CANVAS_WIDTH * EXPORT_SCALE} x {CANVAS_HEIGHT * EXPORT_SCALE} px"
    )
    print(
        f"  Plot area: {PLOT_WIDTH} x {PLOT_HEIGHT} "
        f"(aspect {PLOT_WIDTH / PLOT_HEIGHT:.2f}:1, "
        f"{VERTICAL_SQUISH:.0%} of the published {PUBLISHED_ASPECT}:1)"
    )
    print(f"  Ticks: x every {X_DTICK}, y every {Y_DTICK}")
    print(
        f"  Font: {FONT_FAMILY}; axis titles {AXIS_TITLE_FONT_SIZE}, "
        f"ticks {TICK_FONT_SIZE}, in canvas units"
    )
    print(
        f"  Colors: Pareto (all) {COLOR_PARETO_ALL} gold/700, "
        f"Pareto (without Pd) {COLOR_PARETO_NOPD} neutral/ink, "
        f"cloud {COLOR_CLOUD} neutral/n400 @ {CLOUD_OPACITY:.0%}"
    )


if __name__ == "__main__":
    main()
