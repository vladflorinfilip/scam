"""Paper palette: s1 clay, voice blue, clause olive, lexical plum; lighter 0.5B and darker GPT-OSS."""

import matplotlib as mpl
from matplotlib import font_manager

RULE_COLORS = {
    "s1": "#D97757",
    "voice": "#6A9BCC",
    "clause": "#788C5D",
    "lexical": "#A07CB8",
}
MODEL_LIGHT_COLORS = {
    "s1": "#EAB49F",
    "voice": "#ADC8E3",
    "clause": "#B6C2A5",
    "lexical": "#C7B5D5",
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
