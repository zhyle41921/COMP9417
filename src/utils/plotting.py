import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def plot_agop_heatmap(
    csv_path,
    output_path=None,
    figsize=(10, 8),
    cmap="coolwarm",
    center=0,
    top_k=None,
):
    """
    Plot heatmap from AGOP CSV.

    Args:
        csv_path (str or Path): path to AGOP csv
        output_path (str or Path, optional): save figure if provided
        figsize (tuple): figure size
        cmap (str): color map
        center (float): center value (use 0 for AGOP)
        top_k (int, optional): plot only top-k features by importance
    """

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, index_col=0)

    # optionally restrict to top-k features
    if top_k is not None:
        scores = df.abs().sum(axis=1)
        top_features = scores.sort_values(ascending=False).head(top_k).index
        df = df.loc[top_features, top_features]

    plt.figure(figsize=figsize)
    sns.heatmap(df, cmap=cmap, center=center)

    plt.title(f"AGOP Heatmap ({csv_path.name})")
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"Saved heatmap to: {output_path}")

    plt.show()