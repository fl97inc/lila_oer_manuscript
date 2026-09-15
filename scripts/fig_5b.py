import csv
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import offset_copy
from scipy.stats import beta as B

matplotlib.use("Agg")


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

SERIES = {
    "seq_learning_agent": ("SeqLearning Agent", COLORS["gold/solid"], COLORS["gold/tint"], "-", "H"),
    "active_search": (
        "Active search",
        COLORS["blue/700"],
        COLORS["blue/tint"],
        (0, (5, 1)),
        "s",
    ),
    "bo": (
        "Bayesian optimization (GP + EHVI)",
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
    "llm_no": (
        "Claude Opus 4.6 (no feedback)",
        COLORS["green/tint"],
        COLORS["green/wash"],
        (0, (1, 1.5)),
        "o",
    ),
}

SHORT_LABELS = {
    "seq_learning_agent": "SeqLearning\nAgent",
    "active_search": "Active search",
    "bo": "Bayesian opt.\n(GP + EHVI)",
    "random": "Random",
    "llm_fb": "Claude Opus 4.6",
    "llm_no": "Claude Opus 4.6\n(no feedback)",
}


def series_label(key, short=False):
    if short and key in SHORT_LABELS:
        return SHORT_LABELS[key]
    return SERIES[key][0]


def lighten(hex_color, t):
    h = hex_color.lstrip("#")[:6]
    rgb = (int(h[i: i + 2], 16) for i in (0, 2, 4))
    return "#%02X%02X%02X" % tuple(round(c + (255 - c) * t) for c in rgb)


def apply_style():
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            # Render mathtext (the italic $n$ in the n-count labels) in the
            # same sans-serif face as the rest of the figure, not the default
            # Computer Modern serif.
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
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


CAP_HALF_EM = 0.355
CAP_CENTER_NUDGE_EM = 0.10


def text_cap_center(ax, x, y, s, fontsize, **kwargs):
    kwargs.setdefault("va", "center")
    trans = offset_copy(
        ax.transData,
        fig=ax.figure,
        x=0,
        y=-CAP_CENTER_NUDGE_EM * fontsize,
        units="points",
    )
    return ax.text(x, y, s, fontsize=fontsize, transform=trans, **kwargs)


STRIP_BASELINE = re.compile(
    r'\s*(?:dominant|alignment)-baseline\s*[:=]\s*"?[a-z-]+"?;?'
)

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
        lambda m: "stroke-dasharray: "
        + ",".join(_fmt_scaled(n.strip(), k) for n in m.group(1).split(",") if n.strip()),
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


def _text_to_fig2_attrs(tag):
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

    body = svg[m.end():]
    body = re.sub(r'\bd="([^"]*)"', lambda mm: f'd="{_scale_path_d(mm.group(1), k)}"', body)
    for attr in ("x", "y", "x1", "y1", "x2", "y2", "cx", "cy", "r", "rx", "ry",
                 "width", "height"):
        body = re.sub(
            rf'\b{attr}="(-?[\d.]+)"',
            lambda mm, a=attr: f'{a}="{_fmt_scaled(mm.group(1), k)}"',
            body,
        )
    body = re.sub(r'style="([^"]*)"', lambda mm: f'style="{_scale_style(mm.group(1), k)}"', body)

    def _tr(mm):
        scaled = _scale_transform(mm.group(1), k)
        return "" if scaled is None else f' transform="{scaled}"'

    body = re.sub(r'\s+transform="([^"]*)"', _tr, body)
    body = re.sub(r"<text\b[^>]*>", lambda mm: _text_to_fig2_attrs(mm.group(0)), body)

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


apply_style()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

STEM = "fig_5b"

THR = 0.90
BUDGET = 50
PREFIX = "fig_5b_"

TEXT_INSET = 0.5
OVER_GAP = 0.7
WHISKER_WIDTH = 0.9
CAP_SIZE = 2.2

FIG_HEIGHT_IN = 2.05
BAR_HEIGHT = 0.58
ROW_PITCH = 0.90
BAR_LIGHTEN = 0.15
N_FMT = "$n$ = {n:,}"
N_FMT_FAIL = "($n$ = {n:,})"

METHOD_KEYS = ["active_search", "seq_learning_agent", "bo", "random", "llm_fb", "llm_no"]

LABEL_COLOR = {"seq_learning_agent": COLORS["gold/700"]}

EXTRA_LIGHTEN = {"active_search": 0.12, "seq_learning_agent": 0.12, "bo": 0.12}


def steps(name):
    f = {"active_search": DATA_DIR / f"{PREFIX}active_search.csv",
         "random": DATA_DIR / f"{PREFIX}random.csv",
         "bo": DATA_DIR / f"{PREFIX}bo.csv",
         "seq_learning_agent": DATA_DIR / f"{PREFIX}seq_learning_agent.csv",
         "llm_fb": DATA_DIR / f"{PREFIX}llm_with_feedback.csv",
         "llm_no": DATA_DIR / f"{PREFIX}llm_no_feedback.csv"}[name]
    return np.asarray([float(r["step"]) for r in csv.DictReader(open(f))], float)


A0 = B0 = 0.5


def cross(v, thr=THR):
    for t in range(1, BUDGET + 1):
        if (v <= t).mean() >= thr:
            return t
    return None


def cred_ci(v, thr=THR):
    n = len(v)
    lo = hi = None
    for t in range(1, BUDGET + 1):
        k = int((v <= t).sum())
        a, b = A0 + k, B0 + (n - k)
        p_lo, p_hi = B.ppf([0.05, 0.95], a, b)
        if lo is None and p_hi >= thr:
            lo = t
        if hi is None and p_lo >= thr:
            hi = t
    if hi is None:
        hi = BUDGET + 1
    return lo, hi


def bar_fill(key):
    return lighten(SERIES[key][1], BAR_LIGHTEN + EXTRA_LIGHTEN.get(key, 0))


def right_edge(ax, artist):
    bb = artist.get_window_extent(ax.figure.canvas.get_renderer())
    return ax.transData.inverted().transform((bb.x1, 0))[0]


def render(rows):
    fig, ax = plt.subplots(figsize=(HALF_WIDTH_IN, FIG_HEIGHT_IN))
    y = np.arange(len(rows))[::-1] * ROW_PITCH

    for yi, (name, key, c, ci, peak, n) in zip(y, rows):
        width = c if c is not None else BUDGET
        ax.barh(yi, width, color=bar_fill(key), edgecolor="none",
                height=BAR_HEIGHT, zorder=3)
        if c is not None and ci is not None:
            ax.errorbar(c, yi, xerr=[[c - ci[0]], [ci[1] - c]], fmt="none",
                        ecolor=INK, elinewidth=WHISKER_WIDTH,
                        capsize=CAP_SIZE, capthick=WHISKER_WIDTH, zorder=4)

    y_bottom = -(BAR_HEIGHT / 2 + 0.15)
    y_top = y[0] + BAR_HEIGHT / 2
    ax.vlines(np.arange(0, BUDGET + 1, 10), y_bottom, y_top,
              color=COLORS["n200"], lw=0.6, zorder=0)
    ax.vlines(BUDGET, y_bottom, y_top, ls=(0, (4, 3)), color=COLORS["n600"],
              lw=0.7, zorder=1)

    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_ylim(y_bottom, y_top + 0.12)
    ax.set_xlim(0, BUDGET + 5)
    ax.set_xticks(np.arange(0, BUDGET + 1, 10))
    ax.set_xlabel(
        "Material systems tested to reach ≥90% Pd-discovery probability",
        fontsize=px(AXIS_TITLE_PX),
        fontweight="bold",
    )
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", pad=2)
    ax.xaxis.labelpad = 4
    fig.tight_layout(pad=0.15)

    fig.canvas.draw()
    probe = ax.text(0, 0, ">50", fontsize=px(TICK_PX), fontweight="bold")
    w_px = probe.get_window_extent(fig.canvas.get_renderer()).width
    probe.remove()
    ax.set_xlim(0, (BUDGET + OVER_GAP) / (1 - w_px / ax.get_window_extent().width))
    ax.spines["bottom"].set_bounds(0, BUDGET)
    fig.tight_layout(pad=0.15)

    fig.canvas.draw()
    name_pt = n_pt = px(TICK_PX)

    for yi, (name, key, c, ci, peak, n) in zip(y, rows):
        col = LABEL_COLOR.get(key, SERIES[key][1])
        n_text = (N_FMT if c is not None else N_FMT_FAIL).format(n=n)
        name_text = series_label(key, short=True).replace("\n", " ")

        if c is None:
            lab = text_cap_center(ax, TEXT_INSET, yi, name_text, name_pt,
                                  ha="left", fontweight="bold", color="#FFFFFF",
                                  zorder=5)
            text_cap_center(ax, right_edge(ax, lab) + 0.8, yi, n_text, n_pt,
                            ha="left", fontweight="bold", color="#FFFFFF",
                            zorder=5)
            text_cap_center(ax, BUDGET - 1, yi, f"fail (peaks at {peak:.0%})",
                            px(TICK_PX), ha="right", fontweight="bold",
                            color="#FFFFFF", zorder=4)
            text_cap_center(ax, BUDGET + OVER_GAP, yi, ">50", px(TICK_PX),
                            ha="left", fontweight="bold", color=INK, zorder=5)
        else:
            text_cap_center(ax, (ci[1] + 1.2) if ci else c + 1.0, yi, name_text,
                            name_pt, ha="left", fontweight="bold", color=col,
                            zorder=5, clip_on=False)
            text_cap_center(ax, TEXT_INSET, yi, n_text, n_pt, ha="left",
                            fontweight="bold", color="#FFFFFF", zorder=5)

    text_cap_center(ax, BUDGET - 0.8, y[0], "budget = 50", px(TICK_PX),
                    ha="right", color=COLORS["n600"], zorder=4)

    save_mpl(fig, STEM, OUT_DIR)
    plt.close(fig)


def main():
    rows = []
    for key in METHOD_KEYS:
        v = steps(key)
        c = cross(v)
        peak = (v <= BUDGET).mean()
        ci = cred_ci(v) if c is not None else None
        rows.append((SERIES[key][0], key, c, ci, peak, len(v)))
    rows.sort(key=lambda r: (r[2] is None, r[2] if r[2] is not None else 1e9))

    render(rows)

    for name, key, c, ci, peak, n in rows:
        ci_s = f"  90%CI[{ci[0]:.0f},{ci[1]:.0f}]" if ci else ""
        value = "FAIL(peak %.0f%%)" % (peak * 100) if c is None else str(c)
        print(f"  {name:34s} {value:>16s}{ci_s}")


if __name__ == "__main__":
    main()
