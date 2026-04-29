import sys
from pathlib import Path
from src.utils.plotting import plot_agop_heatmap

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

plot_agop_heatmap(
    ROOT / "outputs" / "insurance_company_benchmark" / "xrfm_best_agop.csv",
    output_path=ROOT / "outputs" / "insurance_company_benchmark" / "agop_heatmap.png",
    top_k=20,
)