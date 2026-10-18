"""Paper palette: s1 clay, voice blue, clause olive, lexical plum; lighter 0.5B and darker GPT-OSS."""

import matplotlib as mpl
from matplotlib import font_manager
import numpy as np

RULE_COLORS = {
    "s1": "#D97757",
    "voice": "#6A9BCC",
    "clause": "#788C5D",
    "lexical": "#A07CB8",
}
MODEL_LIGHT_COLORS = {
    "s1": "#E4A089",
    "voice": "#95B5DC",
    "clause": "#9DAA8A",
    "lexical": "#BC9FCA",
}
MODEL_DARK_COLORS = {
    "s1": "#8D4E38",
    "voice": "#45648A",
    "clause": "#4E5B3D",
    "lexical": "#685177",
}
BASE_COLOR = "#8C8C8C"
RANDOM_COLOR = "#BDBDBD"
CHANCE_COLOR = "#999999"
METRIC_STYLES = {
    "x": {"linestyle": "-", "marker": "x"},
    "dot": {"linestyle": "--", "marker": "o"},
}
SINGLE_COLUMN = 3.4
DOUBLE_COLUMN = 7.0
EXPORT_DPI = 300

_AVAILABLE_FONTS = {entry.name for entry in font_manager.fontManager.ttflist}
FONT_FAMILY = next(
    (family for family in ("Inter", "Helvetica") if family in _AVAILABLE_FONTS),
    "DejaVu Sans",
)


def apply() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_FAMILY],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.titleweight": "normal",
            "axes.titlelocation": "left",
            "axes.labelsize": 8,
            "axes.labelweight": "normal",
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.grid.axis": "y",
            "grid.color": "#E6E6E6",
            "grid.linewidth": 0.5,
            "grid.linestyle": "-",
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.color": "#444444",
            "ytick.color": "#444444",
            "lines.linewidth": 1.1,
            "lines.markersize": 3,
            "savefig.dpi": EXPORT_DPI,
        }
    )


def panel_title(ax, letter: str | None, title: str, **kwargs) -> None:
    prefix = rf"$\mathbf{{{letter}}}$  " if letter else ""
    ax.set_title(prefix + title, loc="left", **kwargs)


def heatmap(ax, values, **kwargs):
    ax.grid(False, which="both")
    image = ax.imshow(values, **kwargs)
    ax.grid(False, which="both")
    return image


def heatmap_text_color(image, value) -> str:
    red, green, blue, _ = image.cmap(image.norm(value))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "#111111" if luminance >= 0.52 else "#FFFFFF"


def annotate_heatmap(
    ax,
    image,
    values,
    *,
    fmt: str = ".2f",
    stars=None,
    show_values: bool = True,
    max_cells: int = 40,
    fontsize: float = 7,
) -> None:
    values = np.asarray(values)
    show_values = show_values and values.size <= max_cells
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            significant = stars is not None and bool(stars[row, column])
            if not show_values and not significant:
                continue
            text = f"{values[row, column]:{fmt}}" if show_values else "*"
            ax.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                fontsize=fontsize,
                color=heatmap_text_color(image, values[row, column]),
            )
