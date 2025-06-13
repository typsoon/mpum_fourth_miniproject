from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional, Protocol, Tuple, override
import numpy as np
from scipy.spatial.distance import pdist, squareform
from tqdm.notebook import tqdm
from sklearn.cluster import KMeans
from scipy.sparse.csgraph import laplacian
# from tqdm import tqdm


# %%
class ClusterNode(ABC):
    def __init__(self, contents):
        self._contents = contents

    @property
    @abstractmethod
    def centroid(self) -> np.ndarray:
        pass

    @property
    def contents(self) -> list[int]:
        return self._contents


class Distance(Protocol):
    @abstractmethod
    def __call__(self, left, right) -> Any:
        pass


class Clusterer:
    @abstractmethod
    def get_metric(self) -> Distance:
        pass

    @abstractmethod
    def get_clusters(
        self, data: np.ndarray, clusters_count: int
    ) -> Iterable[ClusterNode]:
        pass


# %%


class HierarchicalClusterNode(ClusterNode):
    _how_many_identical = 0

    def __init__(
        self, contents: list[int], centroid: np.ndarray, children=None
    ) -> None:
        super().__init__(contents)
        self._children = children
        self._centroid = centroid

    @property
    def children(
        self,
    ) -> Optional[tuple["HierarchicalClusterNode", "HierarchicalClusterNode"]]:
        return self._children

    @property
    def centroid(self) -> np.ndarray:
        return self._centroid

    @staticmethod
    def single_element_node(el_idx: int, element: np.ndarray):
        return HierarchicalClusterNode([el_idx], centroid=np.array(element))

    @classmethod
    def merge_nodes(
        cls,
        left_child: "HierarchicalClusterNode",
        right_child: "HierarchicalClusterNode",
    ) -> "HierarchicalClusterNode":
        contents = left_child.contents + right_child.contents

        new_centroid = (
            len(left_child.contents) * left_child.centroid
            + len(right_child.contents) * right_child.centroid
        ) / len(contents)

        # cls._how_many_identical += np.array_equal(
        #     left_child.centroid, right_child.centroid
        # )
        # print(f"how many identical: {cls._how_many_identical}")
        # print(
        #     f"Centroids: {left_child.centroid}, {right_child.centroid}, Merged: {new_centroid}"
        # )

        return HierarchicalClusterNode(
            contents, new_centroid, (left_child, right_child)
        )

    # def __eq__(self, other):
    #     return isinstance(other, ClusterNode) and self._contents == other._contents
    #
    # def __hash__(self):
    #     return hash(tuple(self._contents))


class Criterium(Protocol):
    @abstractmethod
    def __call__(
        self,
        clusters: Iterable[HierarchicalClusterNode],
        dist_func: np.ndarray,
    ) -> Tuple[int, int]:
        pass

    @abstractmethod
    def init_distances(self, data: np.ndarray, distance: Distance) -> np.ndarray:
        pass

    @abstractmethod
    def update_distances(
        self,
        clusters,
        distances: np.ndarray,
        left: int,
        right: int,
        new_cluster: HierarchicalClusterNode,
        distance: Distance,
    ) -> np.ndarray:
        pass


class HierarchicalClusterer(Clusterer):
    def __init__(self, distance: Distance, connecting_criterium: Criterium):
        self.d = distance
        self.criterium = connecting_criterium

    def get_metric(self):
        return self.d

    def get_clusters(self, data, clusters_count):
        clusters = np.array(
            [
                HierarchicalClusterNode.single_element_node(i, data[i])
                for i in range(len(data))
            ]
        )

        if len(clusters) == 1:
            return clusters

        assert len(clusters) >= 1
        iter_range = tqdm(range(len(clusters) - clusters_count), "Clusters merged")

        distances = self.criterium.init_distances(data, self.d)

        # while len(clusters) > clusters_count:
        for _ in iter_range:
            # print(f"Clusters len: {len(clusters)}")
            x, y = self.criterium(clusters, distances)

            assert x != y

            merged = HierarchicalClusterNode.merge_nodes(clusters[x], clusters[y])

            clusters = np.delete(clusters, [x, y])
            clusters = np.hstack([clusters, np.array([merged])])

            # print(len(clusters), distances.shape, x, y)

            distances = self.criterium.update_distances(
                clusters, distances, x, y, merged, self.d
            )

            # print(len(clusters))

        return clusters


# %%
class EuclideanMetric(Distance):
    @override
    def __call__(self, left, right):
        return np.linalg.norm(left - right)


class CentroidalLinking(Criterium):
    @override
    def __call__(self, clusters, distances):
        i, j = np.unravel_index(np.argmin(distances), distances.shape)
        return (int(i), int(j))

    @override
    def init_distances(self, data: np.ndarray, distance: Distance):
        self._distance_matrix = squareform(pdist(data, metric=distance))
        np.fill_diagonal(self._distance_matrix, np.inf)

        return self._distance_matrix

    @override
    def update_distances(self, clusters, distances, left, right, new_cluster, distance):
        assert left != right
        distances = np.delete(distances, [left, right], axis=0)
        distances = np.delete(distances, [left, right], axis=1)

        new_row = np.array(
            [distance(cl.centroid, new_cluster.centroid) for cl in clusters[:-1]]
        )

        # Append new row and column to distance matrix
        distances = np.vstack([distances, new_row])
        new_col = np.append(new_row, [np.inf]).reshape(
            -1, 1
        )  # set self-distance to inf
        distances = np.hstack([distances, new_col])

        return distances


class MinLinking(Criterium):
    @override
    def __call__(self, clusters, distances):
        i, j = np.unravel_index(np.argmin(distances), distances.shape)
        return (int(i), int(j))

    @override
    def init_distances(self, data: np.ndarray, distance: Distance):
        self._distance_matrix = squareform(pdist(data, metric=distance))
        np.fill_diagonal(self._distance_matrix, np.inf)

        return self._distance_matrix

    @override
    def update_distances(self, clusters, distances, left, right, new_cluster, distance):
        assert left != right
        assert self._distance_matrix is not None

        distances = np.delete(distances, [left, right], axis=0)
        distances = np.delete(distances, [left, right], axis=1)

        new_dists = []
        new_points = new_cluster.contents
        B = np.array(new_points)

        for cluster in clusters[:-1]:
            A = np.array(cluster.contents)
            dists = self._distance_matrix[A[:, None], B[None, :]]
            min_dist = np.min(dists)
            new_dists.append(min_dist)

        # Add new distances
        new_row = np.array(new_dists)
        distances = np.vstack([distances, new_row])
        new_col = np.append(new_row, [np.inf]).reshape(-1, 1)
        distances = np.hstack([distances, new_col])

        return distances


class AvgLinking(Criterium):
    @override
    def __call__(self, clusters, distances):
        i, j = np.unravel_index(np.argmin(distances), distances.shape)
        return (int(i), int(j))

    @override
    def init_distances(self, data: np.ndarray, distance: Distance):
        self._distance_matrix = squareform(pdist(data, metric=distance))
        np.fill_diagonal(self._distance_matrix, np.inf)

        return self._distance_matrix

    @override
    def update_distances(self, clusters, distances, left, right, new_cluster, distance):
        assert left != right
        assert self._distance_matrix is not None

        distances = np.delete(distances, [left, right], axis=0)
        distances = np.delete(distances, [left, right], axis=1)

        new_dists = []
        new_points = new_cluster.contents
        B = np.array(new_points)

        for cluster in clusters[:-1]:
            A = np.array(cluster.contents)
            dists = self._distance_matrix[A[:, None], B[None, :]]
            min_dist = np.mean(dists)
            new_dists.append(min_dist)

        # Add new distances
        new_row = np.array(new_dists)
        distances = np.vstack([distances, new_row])
        new_col = np.append(new_row, [np.inf]).reshape(-1, 1)
        distances = np.hstack([distances, new_col])

        return distances


# %%


class KMeansClusterNode(ClusterNode):
    def __init__(self, contents: list[int], centroid: np.ndarray):
        super().__init__(contents)
        self._centroid = centroid

    @property
    def centroid(self) -> np.ndarray:
        return self._centroid

    @staticmethod
    def single_element_node(idx: int, element: np.ndarray) -> "KMeansClusterNode":
        return KMeansClusterNode([idx], element)


class KMeansClusterer(Clusterer):
    def __init__(self, distance: Distance, runs: int = 10):
        self._d = distance
        self._runs = runs

    @override
    def get_metric(self) -> Distance:
        return self._d

    @override
    def get_clusters(
        self, data: np.ndarray, clusters_count=None
    ) -> Iterable[ClusterNode]:
        k = clusters_count
        n_samples = len(data)

        best_inertia = np.inf
        best_result = None

        for _ in range(self._runs):
            random_indices = np.random.choice(n_samples, k, replace=False)
            centroids = data[random_indices]

            while True:
                dists = np.linalg.norm(
                    data[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2
                )
                assignments = np.argmin(dists, axis=1)

                new_centroids = np.zeros_like(centroids)
                for i in range(k):
                    cluster_points = data[assignments == i]
                    if len(cluster_points) > 0:
                        new_centroids[i] = np.mean(cluster_points, axis=0)
                    else:
                        new_centroids[i] = data[np.random.choice(n_samples)]

                if np.array_equal(centroids, new_centroids):
                    break

                centroids = new_centroids

            final_dists = np.linalg.norm(data - centroids[assignments], axis=1)
            inertia = np.sum(final_dists**2)

            if inertia < best_inertia:
                best_inertia = inertia
                best_result = (assignments.copy(), centroids.copy())

        assert best_result is not None
        assignments, centroids = best_result

        cluster_indices = [np.where(assignments == i)[0].tolist() for i in range(k)]

        return [
            KMeansClusterNode(indices, centroid)  # pyright: ignore
            for indices, centroid in zip(cluster_indices, centroids)
        ]


# %%


class SpectralClusterNode(ClusterNode):
    def __init__(self, contents, data):
        super().__init__(contents)
        self._data = data
        self.lazy_centroid: Optional[np.ndarray] = None

    @property
    def centroid(self) -> np.ndarray:
        if self.lazy_centroid is None:
            self.lazy_centroid = np.mean(self._data[self._contents], axis=0)

        assert self.lazy_centroid is not None
        return self.lazy_centroid


class SpectralClusterer(Clusterer):
    def __init__(self, distance: Distance):
        self._d = distance

    @override
    def get_metric(self) -> Distance:
        return self._d

    @override
    def get_clusters(
        self, data: np.ndarray, clusters_count=None
    ) -> Iterable[ClusterNode]:
        k = clusters_count

        pairwise_dists = squareform(pdist(data, metric=self._d))
        sigma = np.mean(pairwise_dists)
        similarity = np.exp(-(pairwise_dists**2) / (2 * sigma**2))

        lap = laplacian(similarity, normed=True)

        eigvals, eigvecs = np.linalg.eigh(lap)
        selected_vecs = eigvecs[:, 1 : k + 1]

        kmeans = KMeans(n_clusters=k, n_init="auto")
        labels = kmeans.fit_predict(selected_vecs)

        clusters = [[] for _ in range(k)]
        for idx, label in enumerate(labels):
            clusters[label].append(idx)

        return [SpectralClusterNode(c, data) for c in clusters]
