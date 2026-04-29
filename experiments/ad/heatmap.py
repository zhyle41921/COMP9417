import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.utils.plotting import plot_agop_heatmap

plot_agop_heatmap(
    ROOT / "outputs" / "ad" / "xrfm_best_agop.csv",
    output_path=ROOT / "outputs" / "ad" / "agop_heatmap.png",
    top_k=20,
)
