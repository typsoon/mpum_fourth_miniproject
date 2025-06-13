# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %autosave 0
# %matplotlib inline

# %%
import sys
import os
from typing import Optional

sys.path.append(os.path.abspath(".."))  # or path to your project's root

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from project.reading_input import files_info, files_info_list, two_dim_data
from project.model_abstractions import (
    BinaryCrossEntropy,
    ClusteringModel,
    ModelResults,
    train_and_test_clustering_model,
)
from project.clusterers import (
    AvgLinking,
    HierarchicalClusterer,
    EuclideanMetric,
    CentroidalLinking,
    KMeansClusterer,
    MinLinking,
    SpectralClusterer,
)
from project.utils import (
    to_matrices,
    visualize_clusterings,
)
from tqdm.notebook import tqdm
import numpy as np

# from tqdm import tqdm


# %% [markdown]
# # Load and display data
# %%
data, labels = files_info["rp"].read_data()

# %%
data.head()

# %%
data.info()

# %%
data.describe()

# %%
unique_row_count = len(data.drop_duplicates())
print(f"Number of distinct rows: {unique_row_count}")

# %%

# rows_count = 200
# data = data[:rows_count]
# labels = labels[:rows_count]
#
print(labels)


# TODO: Test if removing this changes anything

# train_set, test_set = stratified_split_by_bool_column(df, col_idx=10, ratio=2 / 3)

# %% [markdown]
# # Analyze Data


# %%
# pd.reset_option("display.max_rows")
data.hist(bins=50, figsize=(20, 15))
plt.show()


# %% [markdown]
# ## Hierarchical clusterers


# %%
hclusterer = HierarchicalClusterer(EuclideanMetric(), CentroidalLinking())
hmodel = ClusteringModel(hclusterer, BinaryCrossEntropy(), 2)

hclusterer_min_linking = HierarchicalClusterer(EuclideanMetric(), MinLinking())
hmodel_min_linking = ClusteringModel(hclusterer_min_linking, BinaryCrossEntropy(), 2)

hclusterer_avg_linking = HierarchicalClusterer(EuclideanMetric(), AvgLinking())
hmodel_avg_linking = ClusteringModel(hclusterer_avg_linking, BinaryCrossEntropy(), 2)

h_train_data, h_labels = to_matrices(data, labels)

# %% [markdown]
# ### Hierarchical CentroidalLinking
# %%
# %%time

train_and_test_clustering_model("Hierarchical", hmodel, h_train_data, h_labels)

# %% [markdown]
# ### Hierarchical MinLinking

# %%
# %%time
train_and_test_clustering_model(
    "Hierarchical", hmodel_min_linking, h_train_data, h_labels
)

# %% [markdown]
# ### Hierarchical AvgLinking

# %%
# %%time
train_and_test_clustering_model(
    "Hierarchical", hmodel_avg_linking, h_train_data, h_labels
)

# %% [markdown]
# ## Kmeans clusterer

# %%
kclusterer = KMeansClusterer(EuclideanMetric(), 20)
kmodel = ClusteringModel(kclusterer, BinaryCrossEntropy(), 2)

# %%
# %%time
k_train_data, k_labels = to_matrices(data, labels)
train_and_test_clustering_model("Kmeans", kmodel, k_train_data, k_labels)

# %% [markdown]
# ## Spectal clustering
# %%

sclusterer = SpectralClusterer(EuclideanMetric())
smodel = ClusteringModel(sclusterer, BinaryCrossEntropy(), 2)

# %%
# %%time
s_train_data, s_labels = to_matrices(data, labels)
train_and_test_clustering_model("Spectral", smodel, s_train_data, s_labels)


# %%
def cluster_error_graph(dataset) -> dict[str, Figure]:
    models = [
        # ("hclusterer", hclusterer),
        ("kmeans", kclusterer),
        ("spectral", sclusterer),
    ]

    answer = {}

    for info in tqdm(dataset, "Evaluating data"):
        data, labels = info.read_data()
        data, labels = to_matrices(data, labels)

        if len(data) > 1000:
            continue

        fig = plt.figure(figsize=(10, 6))
        plt.title(f"Cluster loss vs. cluster count for {info.name}")
        plt.xlabel("Clusters count (val)")
        plt.ylabel("Cluster loss")

        for name, clusterer in models:
            losses = []
            vals = list(range(2, 20))  # start from 2 to avoid empty clusters

            for val in vals:
                model: ClusteringModel = ClusteringModel(
                    clusterer, BinaryCrossEntropy(), val, info.is_labeled
                )
                model.train(0, data, labels)
                loss = model.avg_inertia(data)
                losses.append(loss)

            plt.plot(vals, losses, label=name)

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        answer[info.name] = fig

    return answer


# %%
# %%time
two_dim_data_graphs = cluster_error_graph(two_dim_data)
# %%
# %%time
files_info_graphs = cluster_error_graph(files_info_list)

# %%
two_dim_data_graphs.update(files_info_graphs)

# for label, fig in two_dim_data_graphs.items():
#     fig.savefig(f"../Raport/Images/Elbow/inertion_of_{label}.png")

# %%

desired_clusters_count: dict[int, Optional[int]] = {
    0: 9,
    1: 10,
    2: 10,
    3: 8,
    4: None,
    5: 10,
    6: 8,
    7: 10,
}


def evaluate_clustering_models(
    dataset,
    name: str = "Dataset",
    try_your_guesses=False,
    colors_corresponding_to_guesses=False,
) -> tuple[dict[str, list[tuple[str, ModelResults]]], dict[str, Figure]]:
    global desired_clusters_count
    res = {}
    plots = {}

    for i, info in tqdm(enumerate(dataset), f"Evaluating {name}"):
        if not info.is_labeled:
            continue

        res[info.name] = []

        data, labels = info.read_data()
        data, labels = to_matrices(data, labels)

        assert labels is not None
        clusters_count = 2 * len(np.unique(labels))
        clusters_count = 1 * len(np.unique(labels))
        if try_your_guesses and desired_clusters_count[i] is not None:
            clusters_count = desired_clusters_count[i]
        assert clusters_count is not None
        # clusters_count = 1 * len(np.unique(labels))

        models: list[tuple[str, ClusteringModel]] = [
            (
                "hclusterer",
                ClusteringModel(hclusterer, BinaryCrossEntropy(), clusters_count),
            ),
            (
                "hclusterer_min_linking",
                ClusteringModel(
                    hclusterer_min_linking, BinaryCrossEntropy(), clusters_count
                ),
            ),
            (
                "hclusterer_avg_linking",
                ClusteringModel(
                    hclusterer_avg_linking, BinaryCrossEntropy(), clusters_count
                ),
            ),
            (
                "kmeans",
                ClusteringModel(
                    KMeansClusterer(EuclideanMetric(), 200),
                    BinaryCrossEntropy(),
                    clusters_count,
                ),
            ),
            (
                "spectral",
                ClusteringModel(sclusterer, BinaryCrossEntropy(), clusters_count),
            ),
        ]

        models_dict = dict()
        print(f"Model {info.name}, " + ", ".join(label for label, _ in models))
        for name, model in models:
            # if (
            #     name == "hclusterer"
            #     or name == "hclusterer_min_linking"
            #     or name == "hclusterer_avg_linking"
            # ) and len(data) > 1000:
            #     continue

            models_dict[name] = model

            model_res = train_and_test_clustering_model(name, model, data, labels)
            results = res[info.name]
            results.append((name, model_res))

        # Only plot if data is 2D
        if info.dimension == 2:
            # sclusters = sclusterer.get_clusters(data, clusters_count)
            # plot_clusters(data, sclusters)

            plots_grouped = visualize_clusterings(
                data,
                labels,
                models_dict,
                colors_corresponding_to_guesses,
            )

            plots[info.name] = plots_grouped

    return res, plots


# %%

files_info_results, __ = evaluate_clustering_models(
    files_info_list, name="Analyzed files"
)
print()
# %%

_, __ = evaluate_clustering_models(two_dim_data, name="Two dim files")
print()

# %%
best_results, plots = evaluate_clustering_models(
    two_dim_data, name="Two dim files", try_your_guesses=True
)
print()

# %%
best_results.update(files_info_results)

# %%
# print(best_results)

all_models = set()
for models in best_results.values():
    for model_name, _ in models:
        all_models.add(model_name)

all_models = sorted(all_models)
print(all_models)

# write_models_data_to_csv(
#     "../Raport/Data/best_results.csv",
#     all_models,
#     best_results,
#     lambda m_data: m_data.accuracy,
# )
#
# write_models_data_to_csv(
#     "../Raport/Data/times.csv",
#     all_models,
#     best_results,
#     lambda m_data: m_data.training_time,
# )


# for dataset, fig in plots.items():
#     fig.savefig(f"../Raport/Images/2D_visualizations/{dataset}.png")  # pyright: ignore

# %%
_, plots_correct_colors = evaluate_clustering_models(
    two_dim_data,
    name="Two dim files",
    try_your_guesses=True,
    colors_corresponding_to_guesses=True,
)

for dataset, fig in plots_correct_colors.items():
    fig.savefig(f"../Raport/Images/2D_visualizations/correct_colors_{dataset}.png")  # pyright: ignore
