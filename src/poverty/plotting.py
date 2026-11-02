"""Shared matplotlib style: recessive axes, thin marks, fixed categorical order."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MUTED = "#8c8b86"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e0"
SEQ = "Blues"
POS, NEG = "#e34948", "#2a78d6"   # SHAP: pushes towards poor / away from poor


def setup():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelcolor": INK2, "axes.edgecolor": GRID,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": INK2, "ytick.color": INK2, "lines.linewidth": 2,
        "axes.prop_cycle": plt.cycler(color=SERIES), "legend.frameon": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
