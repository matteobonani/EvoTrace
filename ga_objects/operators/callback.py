from pymoo.core.callback import Callback
import numpy as np
from scipy.spatial.distance import pdist, squareform
from typing import Dict, Any


class BaseCallback(Callback):
    def __init__(self):
        super().__init__()
        self.first_objective = []

    def notify(self, algorithm):
        population = algorithm.pop.get("X")

        # compute pairwise hamming distances (only upper triangle)
        dist_matrix = squareform(pdist(population, metric="hamming")) # 0.1s | pop = 3000 , trace_length = 50 , model1

        n = dist_matrix.shape[0]
        mean_per_trace = (np.sum(dist_matrix, axis=1) - 0) / (n - 1)

        self.update_objectives(algorithm, mean_per_trace)

    def update_objectives(self, algorithm, mean_per_trace):
        """
        Abstract method to be implemented by subclasses.
        Updates algorithm population objectives.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def get_data(self) -> Dict[str, Any]:
        """Return recorded first objective history."""
        return {"first_objective": self.first_objective}

class SingleObjectiveCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.constraint_history = []

    def update_objectives(self, algorithm, mean_per_trace):
        G = algorithm.pop.get("G")
        algorithm.pop.set("F", -mean_per_trace[:, None])
        self.first_objective.append(np.mean(mean_per_trace[:, None]))
        self.constraint_history.append(np.mean(G[:, 0]))

    def get_data(self) -> Dict[str, Any]:
        data = super().get_data()
        data.update({"constraint_history": self.constraint_history})
        return data

class MultiObjectiveCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.second_objective = []
        self.max_diversity = 0

    def update_objectives(self, algorithm, mean_per_trace):
        F = algorithm.pop.get("F")

        if self.max_diversity == 0:
            self.max_diversity = len(algorithm.pop.get("X")[0])


        F[:, 0] = -mean_per_trace

        algorithm.pop.set("F", F)

        self.first_objective.append(np.mean(mean_per_trace))
        self.second_objective.append(np.mean(F[:, 1]))

    def get_data(self) -> Dict[str, Any]:
        data = super().get_data()
        data.update({
            "second_objective": self.second_objective,
            "max_diversity": self.max_diversity
        })
        return data




