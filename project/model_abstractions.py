from abc import ABC, abstractmethod
from time import perf_counter
from dataclasses import dataclass
from typing import Any, Iterable, Optional, override
import numpy as np
from project.clusterers import ClusterNode, Clusterer, Distance


# %%
class LossFunc(ABC):
    @abstractmethod
    def loss_value(self, prediction, ground_truth) -> Any:
        pass

    # @abstractmethod
    # def loss_derivative(self, prediction, ground_truth, parameters=None) -> Any:
    #     pass


def avg_inertia(
    clusters: Iterable[ClusterNode],
    centroids: np.ndarray,
    data: np.ndarray,
    distance: Distance,
):
    answer = 0
    for cluster, centroid in zip(clusters, centroids):
        answer += np.sum(
            [distance(x, centroid) ** 2 for x in data[cluster.contents]]
        ) / len(data)

    return answer


class BinaryCrossEntropy(LossFunc):
    # prediction is a vector holding probabilities for each class
    # ground truth = labels one hot
    @override
    def loss_value(self, prediction, ground_truth):
        eps = 1e-15
        prediction = np.clip(prediction, eps, 1 - eps)  # stabilize log

        assert prediction.shape == ground_truth.shape, (
            f"{prediction.shape} != {ground_truth.shape}"
        )

        # BCE: -(y * log(p) + (1 - y) * log(1 - p))
        losses = -(
            ground_truth * np.log(prediction)
            + (1 - ground_truth) * np.log(1 - prediction)
        )

        return np.mean(losses)

    # @override
    # def loss_derivative(self, prediction, ground_truth, parameters=None):
    #     # return 1 / self.number_of_inputs * (prediction - ground_truth)
    #     return 1 / len(prediction) * (prediction - ground_truth)


# %%
class Model(ABC):
    @abstractmethod
    def train(self, epochs_num, data, labels, loss_at_epoch=None):
        pass

    @abstractmethod
    def test(self, data, labels) -> np.ndarray:
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def predict(self, data) -> np.ndarray:
        pass


# %%
class ClusteringModel(Model):
    def __init__(
        self,
        clusterer: Clusterer,
        loss_func: LossFunc,
        clusters_count: int,
        labeled: bool = True,
    ):
        self._clusterer = clusterer
        self._loss_func = loss_func
        self._clusters_count = clusters_count
        self.reset()
        self._labeled = labeled

    # @property
    # def labeled(self):
    #     return self._labeled

    @property
    def clusters(self):
        if self._clusters is None:
            raise Exception("Model has not been trained yet")

        return self._clusters

    @property
    def clusterer(self):
        return self._clusterer

    @override
    def train(self, epochs_num, data, labels, loss_at_epoch=None):  # pyright: ignore
        # print("Train called")
        self._clusters = self._clusterer.get_clusters(data, self._clusters_count)
        self._data = data

        self._cluster_to_class_mapping = dict()
        self._val_to_cluster_mapping = dict()
        mapping = self._cluster_to_class_mapping

        if self._labeled:
            for i, cluster in enumerate(self._clusters):
                label = np.argmax(np.bincount(labels[cluster.contents]))
                mapping[i] = int(label)

                for idx in cluster.contents:
                    self._val_to_cluster_mapping[idx] = i

    @override
    def test(self, data, labels):
        if self._clusters is None or self._cluster_to_class_mapping is None:
            raise Exception("Model has not been trained yet")

        prediction = self.predict(data)

        return self._loss_func.loss_value(prediction, labels)

    def avg_inertia(self, data):
        if self._clusters is None or self._cluster_to_class_mapping is None:
            raise Exception("Model has not been trained yet")

        centroids = np.array([node.centroid for node in self._clusters])
        return avg_inertia(
            self._clusters, centroids, data, self._clusterer.get_metric()
        )

    @override
    def reset(self):
        self._data: Optional[np.ndarray] = None
        self._clusters: Optional[Iterable[Any]] = None
        self._cluster_to_class_mapping: Optional[dict[int, int]] = None
        self._val_to_cluster_mapping: Optional[dict[int, int]] = None

    @override
    def predict(self, data):
        val_to_cluster = self._val_to_cluster_mapping
        cluster_to_class = self._cluster_to_class_mapping

        if val_to_cluster is None or cluster_to_class is None:
            raise Exception("Model has not been trained yet")

        def inner_predict(idx):
            assert idx in val_to_cluster.keys()

            cluster = val_to_cluster[idx]
            return cluster_to_class[cluster]

        answer = np.array([inner_predict(idx) for idx in range(len(data))])
        # print(val_to_cluster)
        # print(cluster_to_class)
        # print(answer.shape)
        # print(answer)
        return answer


# %%

learning_rate = 0.01
some_models = []


def get_accuracy(model: Model, data, labels) -> np.ndarray:
    prediction = model.predict(data)

    # print(prediction)
    # print(labels)
    # print(prediction == labels)
    print(np.bincount(prediction == labels))
    return np.mean(prediction == labels)


@dataclass(frozen=True)
class ModelResults:
    accuracy: Any
    avg_inertia: Any
    training_time: float


def train_and_test_clustering_model(
    name,
    model: ClusteringModel,
    train_matrix,
    train_labels_matrix,
    # skip_this_many_in_plot=0,
    # lambda_val=0.02,
) -> ModelResults:
    # loss_at_epochs = [model.test(train_matrix, train_labels_matrix)]

    start = perf_counter()
    model.train(0, train_matrix, train_labels_matrix)
    end = perf_counter()
    # print("Train set: ", model.test(train_matrix, train_labels_matrix))

    acc = get_accuracy(model, train_matrix, train_labels_matrix)
    inertia = model.avg_inertia(train_matrix)

    print(
        f"""Model: {name}, train set accuracy: {acc}, 
        avg inertia: {inertia}""",
    )

    return ModelResults(acc, inertia, end - start)
    # print("Validation set: ", model.test(validation_matrix, validation_labels_matrix))
    # print("Test set: ", model.test(test_matrix, test_labels_matrix))
