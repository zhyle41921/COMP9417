import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.utils.plotting import plot_agop_heatmap

plot_agop_heatmap(
    ROOT / "outputs" / "wine_quality" / "xrfm_best_agop.csv",
    output_path=ROOT / "outputs" / "wine_quality" / "agop_heatmap.png",
    top_k=20,
)