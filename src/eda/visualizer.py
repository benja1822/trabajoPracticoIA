from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from src.config import (
    SEED, KMEANS_N_CLUSTERS, TSNE_PERPLEXITY,
    UMAP_N_NEIGHBORS, UMAP_MIN_DIST, EDA_SAMPLE_SIZE, VOC_CLASSES,
)

sns.set_theme(style="whitegrid", palette="muted")


def plot_class_distribution(
    class_to_ids: Dict[str, List[str]],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    classes = [c for c in VOC_CLASSES if c in class_to_ids]
    counts  = [len(class_to_ids[c]) for c in classes]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(classes, counts, color=sns.color_palette("muted", len(classes)))
    ax.set_xlabel("Número de imágenes")
    ax.set_title("Distribución de imágenes por clase — Pascal VOC 2012")
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.invert_yaxis()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_dimension_scatter(
    stats: List[Dict],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    widths  = [r["width"]  for r in stats]
    heights = [r["height"] for r in stats]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].scatter(widths, heights, alpha=0.3, s=5, color="steelblue")
    axes[0].set_xlabel("Ancho (px)"); axes[0].set_ylabel("Alto (px)")
    axes[0].set_title("Dimensiones de imágenes")

    axes[1].hist(widths, bins=40, color="steelblue", edgecolor="white")
    axes[1].set_xlabel("Ancho (px)"); axes[1].set_title("Distribución de ancho")

    axes[2].hist(heights, bins=40, color="coral", edgecolor="white")
    axes[2].set_xlabel("Alto (px)"); axes[2].set_title("Distribución de alto")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_sample_images(
    image_ids: List[str],
    images_dir: Path,
    n: int = 16,
    title: str = "Muestra de imágenes",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    sample = random.Random(SEED).sample(image_ids, min(n, len(image_ids)))
    cols = 4
    rows = (len(sample) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()

    for ax, img_id in zip(axes, sample):
        img_path = images_dir / f"{img_id}.jpg"
        if img_path.exists():
            ax.imshow(Image.open(img_path))
        ax.set_title(img_id, fontsize=7)
        ax.axis("off")

    for ax in axes[len(sample):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def find_optimal_k(
    embeddings: np.ndarray,
    k_range: range = range(5, 25),
    save_path: Optional[Path] = None,
) -> int:
    """Find optimal K for K-Means via silhouette score (scikit-learn best practice).

    Returns the K with the highest silhouette score.
    The silhouette score measures cohesion vs separation: higher is better.
    Range: [-1, 1]. Values near 1 indicate well-separated clusters.
    """
    from sklearn.metrics import silhouette_score

    n = min(EDA_SAMPLE_SIZE, len(embeddings))
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(embeddings), size=n, replace=False)
    sub = embeddings[idx]

    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=SEED, n_init="auto")
        labels = km.fit_predict(sub)
        scores[k] = silhouette_score(sub, labels, sample_size=min(2000, n))

    best_k = max(scores, key=lambda k: scores[k])

    if save_path:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(list(scores.keys()), list(scores.values()), marker="o", color="steelblue")
        ax.axvline(best_k, color="coral", linestyle="--", label=f"Óptimo K={best_k}")
        ax.set_xlabel("K (número de clusters)"); ax.set_ylabel("Silhouette Score")
        ax.set_title("Selección de K — Silhouette Analysis")
        ax.legend(); plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return best_k


def plot_embedding_clusters(
    embeddings: np.ndarray,
    image_ids: List[str],
    method: str = "tsne",
    n_clusters: Optional[int] = None,
    save_path: Optional[Path] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Reduce embeddings to 2D and color by K-Means cluster.

    If n_clusters is None, uses KMEANS_N_CLUSTERS from config.
    Returns the figure and the cluster labels array.
    """
    n = min(EDA_SAMPLE_SIZE, len(embeddings))
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(embeddings), size=n, replace=False)
    sub = embeddings[idx]

    k = n_clusters if n_clusters is not None else KMEANS_N_CLUSTERS
    kmeans = KMeans(n_clusters=k, random_state=SEED, n_init="auto")
    labels = kmeans.fit_predict(sub)

    if method == "tsne":
        reducer = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY,
                       random_state=SEED, n_jobs=-1)
        coords = reducer.fit_transform(sub)
    elif method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST,
                                 random_state=SEED)
            coords = reducer.fit_transform(sub)
        except ImportError:
            raise ImportError("umap-learn no instalado. Usar method='tsne' o pip install umap-learn")
    else:
        raise ValueError(f"method debe ser 'tsne' o 'umap', recibido: {method!r}")

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=labels, cmap="tab20", alpha=0.6, s=8,
    )
    ax.set_title(f"Clustering de embeddings ({method.upper()}, K={KMEANS_N_CLUSTERS})")
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, labels


def show_search_results(
    query: str,
    retrieved_ids: List[str],
    images_dir: Path,
    relevant_ids: Optional[set] = None,
    title_prefix: str = "",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    n = min(10, len(retrieved_ids))
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i, img_id in enumerate(retrieved_ids[:n]):
        ax = axes[i]
        img_path = images_dir / f"{img_id}.jpg"
        if img_path.exists():
            ax.imshow(Image.open(img_path))
        is_rel = relevant_ids is not None and img_id in relevant_ids
        color  = "green" if is_rel else ("red" if relevant_ids is not None else "gray")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
        ax.set_title(f"#{i+1} {img_id}", fontsize=7)
        ax.axis("off")

    fig.suptitle(f'{title_prefix}Query: "{query}"', fontsize=11)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
