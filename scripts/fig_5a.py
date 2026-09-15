import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CANVAS_PX = 2100
CANVAS_PPI = 300
EXPORT_SCALE = 2
PNG_DPI = 600
FIG_WIDTH_IN = CANVAS_PX / CANVAS_PPI
HALF_WIDTH_IN = FIG_WIDTH_IN / 2
HALF_CANVAS_PX = CANVAS_PX // 2

AXIS_TITLE_PX = 31
TICK_PX = 29
LABEL_PX = 31


def px(size_px):
    return size_px / CANVAS_PPI * 72.0


COLORS = {
    "gold/wash": "#F4E6D3",
    "gold/200": "#F2D3A2",
    "gold/300": "#D7B06D",
    "gold/tint": "#EDD4AD",
    "gold/solid": "#D7AA54",
    "gold/700": "#AD8536",
    "orange/wash": "#FFE2D6",
    "orange/200": "#FFC6AA",
    "orange/300": "#FF9D78",
    "orange/tint": "#F5B9A3",
    "orange/solid": "#E26F46",
    "orange/700": "#B44D29",
    "green/wash": "#DDECE5",
    "green/200": "#AED4C5",
    "green/300": "#81B6A2",
    "green/tint": "#A4C2B6",
    "green/solid": "#458871",
    "green/700": "#236450",
    "purple/wash": "#ECE5F7",
    "purple/200": "#BEAFDA",
    "purple/300": "#A08CC7",
    "purple/tint": "#B7ABCD",
    "purple/solid": "#715CA1",
    "purple/700": "#4E3C7A",
    "rust/wash": "#F0C9BF",
    "rust/200": "#CA9486",
    "rust/300": "#B56F5D",
    "rust/tint": "#C79C91",
    "rust/solid": "#8B3D2B",
    "rust/700": "#6D2618",
    "blue/wash": "#DEEAF3",
    "blue/200": "#9FC0D6",
    "blue/300": "#70A1C0",
    "blue/tint": "#9EB7C8",
    "blue/solid": "#2A7498",
    "blue/700": "#005272",
    "n50": "#F2F2F2",
    "n200": "#D6D6D6",
    "n400": "#B0B0B0",
    "n600": "#8A8A8A",
    "n800": "#5A5A5A",
    "ink": "#231F20",
}

INK = COLORS["ink"]
BAND_ALPHA = 0.35

SERIES = {
    "seq_learning_agent": (
        "SeqLearning Agent",
        COLORS["gold/solid"],
        COLORS["gold/tint"],
        "-",
        "H",
    ),
    "active_search": (
        "Active search",
        COLORS["blue/700"],
        COLORS["blue/tint"],
        (0, (5, 1)),
        "s",
    ),
    "bo": (
        "Bayesian optimization (EHVI)",
        COLORS["rust/solid"],
        COLORS["rust/tint"],
        ":",
        "D",
    ),
    "random": ("Random", COLORS["n800"], COLORS["n200"], "--", "x"),
    "llm_fb": (
        "Claude Opus 4.6 (with feedback)",
        COLORS["green/solid"],
        COLORS["green/tint"],
        "-.",
        "o",
    ),
}

SHORT_LABELS = {
    "seq_learning_agent": "SeqLearning\nAgent",
    "active_search": "Active search",
    "bo": "Bayesian opt.\n(EHVI)",
    "random": "Random",
    "llm_fb": "Claude Opus 4.6",
}


def series_label(key, short=False):
    if short and key in SHORT_LABELS:
        return SHORT_LABELS[key]
    return SERIES[key][0]


def apply_style():
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "font.size": px(TICK_PX),
            "axes.labelsize": px(AXIS_TITLE_PX),
            "axes.labelweight": "bold",
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": px(TICK_PX),
            "ytick.labelsize": px(TICK_PX),
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "text.color": INK,
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
            "savefig.bbox": "tight",
            "figure.dpi": 150,
            "svg.fonttype": "none",
        }
    )


STRIP_BASELINE = re.compile(
    r'\s*(?:dominant|alignment)-baseline\s*[:=]\s*"?[a-z-]+"?;?'
)
CAP_HALF_EM = 0.355
CAP_CENTER_NUDGE_EM = 0.10

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def _fmt_scaled(value, k):
    return f"{float(value) * k:.6f}".rstrip("0").rstrip(".")


def _scale_path_d(d, k):
    return re.sub(_NUM, lambda m: _fmt_scaled(m.group(0), k), d)


def _scale_style(style, k):
    style = re.sub(
        r"font-size:\s*([\d.]+)px",
        lambda m: f"font-size: {_fmt_scaled(m.group(1), k)}px",
        style,
    )
    style = re.sub(
        r"stroke-width:\s*([\d.]+)",
        lambda m: f"stroke-width: {_fmt_scaled(m.group(1), k)}",
        style,
    )
    style = re.sub(
        r"stroke-dasharray:\s*([\d., ]+)",
        lambda m: (
            "stroke-dasharray: "
            + ",".join(
                _fmt_scaled(n.strip(), k) for n in m.group(1).split(",") if n.strip()
            )
        ),
        style,
    )
    return style


def _scale_transform(tr, k):
    tr = tr.strip()
    nums = [float(n) for n in re.findall(_NUM, tr)]
    if tr.startswith("rotate"):
        if not nums or abs(nums[0]) < 1e-9:
            return None
        if len(nums) == 3:
            return f"rotate({nums[0]:g} {_fmt_scaled(nums[1], k)} {_fmt_scaled(nums[2], k)})"
        return tr
    if tr.startswith("translate") and len(nums) >= 1:
        rest = nums[1] if len(nums) > 1 else 0.0
        return f"translate({_fmt_scaled(nums[0], k)} {_fmt_scaled(rest, k)})"
    if tr.startswith("matrix") and len(nums) == 6:
        a, b, c, d, e, f = nums
        return f"matrix({a} {b} {c} {d} {_fmt_scaled(e, k)} {_fmt_scaled(f, k)})"
    return tr


def _text_to_plain_attrs(tag):
    sm = re.search(r'style="([^"]*)"', tag)
    style = {}
    if sm:
        for part in sm.group(1).split(";"):
            if ":" in part:
                key, val = part.split(":", 1)
                style[key.strip()] = val.strip()
    x_m = re.search(r'\bx="([^"]+)"', tag)
    y_m = re.search(r'\by="([^"]+)"', tag)
    tr_m = re.search(r'transform="([^"]*)"', tag)
    x = x_m.group(1) if x_m else None
    y = y_m.group(1) if y_m else None
    transform = None
    if tr_m:
        t = tr_m.group(1)
        nums = re.findall(_NUM, t)
        if t.startswith("translate") and nums:
            x = nums[0]
            y = nums[1] if len(nums) > 1 else "0"
        elif t.startswith("rotate") and nums and abs(float(nums[0])) >= 1e-9:
            transform = t
            if x is None and len(nums) >= 3:
                x, y = nums[1], nums[2]
    if x is None or y is None:
        return tag
    attrs = [f'x="{x}"', f'y="{y}"']
    fs = style.get("font-size", "").replace("px", "").strip()
    if fs:
        attrs.append(f'font-size="{int(round(float(fs)))}"')
    fw = style.get("font-weight")
    if fw:
        attrs.append(f'font-weight="{fw}"')
    fill = style.get("fill")
    if fill:
        attrs.append(f'fill="{fill}"')
    ta = style.get("text-anchor")
    if ta:
        attrs.append(f'text-anchor="{ta}"')
    if transform:
        attrs.append(f'transform="{transform}"')
    return "<text " + " ".join(attrs) + ">"


def mpl_svg_to_canvas(path, width_px=None, ppi=CANVAS_PPI):
    svg = open(path, encoding="utf-8").read()
    svg = re.sub(r"<!DOCTYPE svg[^>]*>\s*", "", svg)
    svg = re.sub(r"<metadata>[\s\S]*?</metadata>\s*", "", svg)
    m = re.search(r"<svg\b[^>]*>", svg)
    if not m:
        return None, None
    tag = m.group(0)
    vb = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', tag)
    if vb:
        src_w, src_h = float(vb.group(1)), float(vb.group(2))
    else:
        wm = re.search(r'\bwidth="([\d.]+)(pt|px)?"', tag)
        hm = re.search(r'\bheight="([\d.]+)(pt|px)?"', tag)
        src_w, src_h = float(wm.group(1)), float(hm.group(1))
    if width_px is None:
        width_px = round(src_w / 72.0 * ppi)
    k = width_px / src_w
    height_px = round(src_h * k)

    body = svg[m.end() :]
    body = re.sub(
        r'\bd="([^"]*)"', lambda mm: f'd="{_scale_path_d(mm.group(1), k)}"', body
    )
    for attr in (
        "x",
        "y",
        "x1",
        "y1",
        "x2",
        "y2",
        "cx",
        "cy",
        "r",
        "rx",
        "ry",
        "width",
        "height",
    ):
        body = re.sub(
            rf'\b{attr}="(-?[\d.]+)"',
            lambda mm, a=attr: f'{a}="{_fmt_scaled(mm.group(1), k)}"',
            body,
        )
    body = re.sub(
        r'style="([^"]*)"', lambda mm: f'style="{_scale_style(mm.group(1), k)}"', body
    )

    def _tr(mm):
        scaled = _scale_transform(mm.group(1), k)
        return "" if scaled is None else f' transform="{scaled}"'

    body = re.sub(r'\s+transform="([^"]*)"', _tr, body)
    body = re.sub(r"<text\b[^>]*>", lambda mm: _text_to_plain_attrs(mm.group(0)), body)

    new_tag = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{width_px}px" height="{height_px}px" '
        f'viewBox="0 0 {width_px} {height_px}" '
        f'font-family="Arial" text-rendering="geometricPrecision">'
    )
    open(path, "w", encoding="utf-8").write(svg[: m.start()] + new_tag + body)
    return width_px, height_px


def sanitize_svg_for_figma(path, font_family="Arial"):
    svg = open(path, encoding="utf-8").read()

    def fix(match):
        tag = match.group(0)
        mode = re.search(r'(?:dominant|alignment)-baseline\s*[:=]\s*"?([a-z-]+)', tag)
        y = re.search(r'\by="([-\d.]+)"', tag)
        if mode and y:
            fs_m = re.search(r'font-size[:=]"?\s*([\d.]+)', tag)
            fs = float(fs_m.group(1)) if fs_m else LABEL_PX
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


def save_mpl(fig, stem, out_dir, canvas_width=HALF_CANVAS_PX):
    import cairosvg
    from PIL import Image

    out_dir = Path(out_dir)
    svg_path = out_dir / f"{stem}.svg"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(svg_path, format="svg", facecolor="#FFFFFF")
    w, h = mpl_svg_to_canvas(svg_path, width_px=canvas_width)
    sanitize_svg_for_figma(svg_path)
    png_w = round(w * EXPORT_SCALE)
    png_h = round(h * EXPORT_SCALE)
    cairosvg.svg2png(
        url=str(svg_path),
        write_to=str(png_path),
        output_width=png_w,
        output_height=png_h,
    )
    with Image.open(png_path) as image:
        image.save(png_path, dpi=(PNG_DPI, PNG_DPI))
    print(f"wrote {svg_path} ({w} x {h} px)")
    print(f"wrote {png_path} ({png_w} x {png_h} px at {PNG_DPI} dpi)")
    return w, h


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "figures"

MAX_STEPS = 50
FIG_HEIGHT_IN = 1.92
LINE_WIDTH = 1.1
SERIES_LABEL_PX = 28
LABEL_NUDGE = {
    "seq_learning_agent": 0.08,
    "active_search": -0.07,
    "bo": 0.05,
    "random": -0.08,
    "llm_fb": 0.0,
}
LABEL_X_PAD = 1.2

PLOT_ORDER = ["random", "bo", "active_search", "llm_fb", "seq_learning_agent"]

SOURCES = {
    "seq_learning_agent": "fig_5a_seq_learning_agent.csv",
    "bo": "fig_5a_bo.csv",
    "random": "fig_5a_random.csv",
    "active_search": "fig_5a_active_search.csv",
    "llm_fb": "fig_5a_llm.csv",
}


def prepare_curves(df, group_cols, metric_col="hypervolume", max_steps=MAX_STEPS):
    curves = []
    for _, grp in df.groupby(group_cols):
        grp = grp.sort_values("step")
        values = grp[metric_col].values
        steps = grp["step"].values

        curve = np.full(max_steps, np.nan)
        curve[steps.astype(int) - 1] = values

        last_valid = None
        for i in range(max_steps):
            if not np.isnan(curve[i]):
                last_valid = curve[i]
            elif last_valid is not None:
                curve[i] = last_valid

        curves.append(curve)
    return np.array(curves)


def load(key):
    df = pd.read_csv(DATA_DIR / SOURCES[key])
    if key == "llm_fb":
        df = df[df["llm"] == "claude_opus4.6"]
    return df


def render(stem, height=FIG_HEIGHT_IN):
    steps = np.arange(1, MAX_STEPS + 1)
    fig, ax = plt.subplots(figsize=(HALF_WIDTH_IN, height))

    finals = {}
    for key in PLOT_ORDER:
        df = load(key)
        if len(df) == 0:
            print(f"  skipping {key}: no rows")
            continue
        label, line_color, band_color, ls, _marker = SERIES[key]
        curves = prepare_curves(df, ["repeat"])
        mean = np.nanmean(curves, axis=0)
        n = np.sum(~np.isnan(curves), axis=0)
        se = np.nanstd(curves, axis=0) / np.sqrt(np.maximum(n, 1))

        ax.fill_between(
            steps,
            mean - se,
            mean + se,
            color=band_color,
            alpha=BAND_ALPHA,
            linewidth=0,
            zorder=2,
        )
        ax.plot(
            steps,
            mean,
            linestyle=ls,
            color=line_color,
            linewidth=LINE_WIDTH,
            zorder=3,
            solid_capstyle="round",
        )
        finals[key] = (mean[-1], curves)
        print(f"  {label:34s} n={curves.shape[0]:3d}  final={mean[-1]:.4f}")

    ax.set_xlabel("Iteration", fontsize=px(AXIS_TITLE_PX), fontweight="bold")
    ax.set_ylabel("Hypervolume", fontsize=px(AXIS_TITLE_PX), fontweight="bold")
    ax.set_xticks([1] + list(range(10, MAX_STEPS + 1, 10)))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.4))
    ax.set_xlim(1, MAX_STEPS)
    ax.grid(False)
    ax.tick_params(axis="x", pad=2, labelsize=px(TICK_PX))
    ax.tick_params(axis="y", pad=2, labelsize=px(TICK_PX))
    ax.xaxis.labelpad = 4
    ax.yaxis.labelpad = 4

    for key, (final, _curves) in finals.items():
        label, line_color, _band, _ls, _marker = SERIES[key]
        text = series_label(key, short=True).replace("\n", " ")
        ax.annotate(
            text,
            xy=(MAX_STEPS + LABEL_X_PAD, final + LABEL_NUDGE.get(key, 0.0)),
            xytext=(0, -CAP_CENTER_NUDGE_EM * px(SERIES_LABEL_PX)),
            textcoords="offset points",
            xycoords="data",
            ha="left",
            va="center",
            fontsize=px(SERIES_LABEL_PX),
            color=line_color,
            fontweight="bold",
            annotation_clip=False,
        )

    fig.tight_layout(pad=0.15)
    save_mpl(fig, stem, OUT_DIR)
    plt.close(fig)
    return finals


def main():
    apply_style()
    OUT_DIR.mkdir(exist_ok=True)

    finals = render("fig_5a")

    print("\nFinal hypervolume at step 50")
    for key, (final, curves) in finals.items():
        last = curves[:, -1]
        valid = last[~np.isnan(last)]
        print(
            f"  {SERIES[key][0]:34s} mean={np.mean(valid):.4f} "
            f"± {np.std(valid):.4f}  median={np.median(valid):.4f}  n={len(valid)}"
        )


if __name__ == "__main__":
    main()
