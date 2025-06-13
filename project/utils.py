from typing import Iterable
from matplotlib.figure import Figure
import csv
import numpy as np
import matplotlib.pyplot as plt

from project.model_abstractions import (
    ClusteringModel,
)
from project.clusterers import ClusterNode


# %%
def standarize_matrix(data_matrix):
    mean = np.mean(data_matrix, axis=0)
    std = np.std(data_matrix, axis=0)
    std[std == 0] = 1
    standarized = np.array((data_matrix - mean) / std)
    return standarized


def standarize_matrix_and_add_ones(data_matrix):
    standarized = standarize_matrix(data_matrix)
    return np.c_[np.ones(standarized.shape[0]), standarized]


# %%
def to_matrices(data, labels, preprocess_func=lambda x: x):
    return preprocess_func(
        data.to_numpy()
    ), labels.to_numpy() if labels is not None else None


# %%


def plot_clusters(data: np.ndarray, clusters: Iterable[ClusterNode]):
    assert data.shape[1] == 2, f"Data must be 2D but is {data.shape}"

    colors = plt.cm.get_cmap("tab10", len(list(clusters)))  # up to 10 different colors

    plt.figure(figsize=(8, 6))

    for i, cluster in enumerate(clusters):
        points = data[cluster.contents]
        plt.scatter(
            points[:, 0], points[:, 1], s=40, color=colors(i), label=f"Cluster {i}"
        )

        # Draw centroid if available
        if hasattr(cluster, "centroid"):
            centroid = getattr(cluster, "centroid")
            plt.scatter(
                centroid[0],
                centroid[1],
                marker="x",
                s=200,
                color=colors(i),
                edgecolor="black",
                linewidth=2,
            )

    plt.title("Cluster visualization")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_clusterings(
    data,
    labels,
    models: dict[str, ClusteringModel],
    colors_corresponding_to_classes: bool = False,
) -> Figure:
    fig, axes = plt.subplots(1, len(models) + 1, figsize=(5 * (len(models) + 1), 5))

    # Ground truth plot
    axes[0].scatter(data[:, 0], data[:, 1], c=labels, cmap="tab20", s=10)
    axes[0].set_title("Ground Truth")

    # Clusterer plots
    for ax, (name, model) in zip(axes[1:], models.items()):
        # Assign color to each cluster
        colors = np.zeros(len(data), dtype=int)
        if colors_corresponding_to_classes:
            colors = model.predict(data)
        else:
            clusters = model.clusters

            sizes = [len(cl.contents) for cl in clusters]
            print(
                f"Clusters count: {len(sizes)}, sizes: {sizes}, sizes standard deviation: {np.std(sizes)}"
            )

            for i, cluster in enumerate(clusters):
                for idx in cluster.contents:
                    colors[idx] = i

        centroids = np.empty((0, 2))
        for cluster in model.clusters:
            if hasattr(cluster, "centroid"):
                centroids = np.vstack(
                    [
                        centroids,
                        np.reshape(cluster.centroid, (-1, 2)),
                    ]
                )

        ax.scatter(data[:, 0], data[:, 1], c=colors, cmap="tab20", s=10)
        ax.scatter(
            centroids[:, 0],  # pyright: ignore
            centroids[:, 1],  # pyright: ignore
            marker="X",  # Big X to flex on the data points
            c="black",  # Or 'white' for lighter style
            s=100,  # Size of centroids
            edgecolor="red",  # Outline for clarity
            linewidth=1.5,
            label="Centroids",
        )
        ax.set_title(name)

    plt.tight_layout()
    plt.show()
    return fig


def latex_friendly(x):
    return x.replace("_", "-")


def write_models_data_to_csv(
    csv_name, all_models, results, preprocess_func, default_val=float("nan")
):
    with open(csv_name, "w", newline="") as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow(["dataset"] + list(map(latex_friendly, all_models)))

        # Write rows
        for dataset_name, model_results in results.items():
            result_dict = {name: preprocess_func(res) for name, res in model_results}
            row = [latex_friendly(dataset_name)] + [
                round(result_dict.get(model, default_val), 4) for model in all_models
            ]
            writer.writerow(row)
