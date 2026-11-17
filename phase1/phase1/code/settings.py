"""Shared paths, feature order and plotting settings (no model calculations)."""
from pathlib import Path
import os
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parent / "Problems" / "ProblemC_Music_data"
DATA, RESULTS, FIGURES = [ROOT / p for p in ["data", "results", "figures"]]
for p in [DATA, RESULTS, FIGURES]:
    p.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mat3300_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURES = ["danceability", "energy", "valence", "tempo", "loudness",
            "acousticness", "instrumentalness", "liveness", "speechiness", "log_duration"]
SEED = 3300
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 120, "savefig.dpi": 170})

def save_figure(fig, name):
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
