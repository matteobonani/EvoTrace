from pymoo.core.callback import Callback
import numpy as np
from scipy.spatial.distance import pdist, squareform
from typing import Dict, Any


class UpdatePopulationCallback(Callback):
    """
    A callback class for tracking the optimization process and updating the current population.

    This class collects constraint violations, objective values, and generation counts
    to monitor the progress of the optimization run.
    """
    def __init__(self):
        super().__init__()
        self.constraint_history = []
        self.second_objective = []
        self.first_objective = []


    def notify(self, algorithm):
        """Update the problem's current population and gather data for plotting."""

        population = algorithm.pop.get("X")


        # pdist is faster than cdist for symmetric pairwise distance matrices
        # storing only the upper triangular matrix (the diagonal is 0 and the lower triangular matrix has the same values as the upper one)
        dist_matrix = squareform(pdist(population, metric="hamming")) # 0.1s | pop = 3000 , trace_length = 50 , model1

        # eliminate the diagonal of 0
        n = dist_matrix.shape[0]
        mean_per_trace = (np.sum(dist_matrix, axis=1) - 0) / (n - 1)

        # diversity_weight = 0.2
        # max_diversity = 50
        #
        # normalized_diversity = mean_per_trace / max_diversity
        # weighted_diversity = diversity_weight * normalized_diversity

        self.first_objective.append(np.mean(mean_per_trace[:, None]))

        # algorithm.pop.set("F", -mean_per_trace[:, None])

        # store current population
        algorithm.problem.set_current_population(population)

        G = algorithm.pop.get("G")
        if G is not None and G.size > 0:
            algorithm.pop.set("F", -mean_per_trace[:, None])
            self.constraint_history.append(np.mean(G[:, 0]))

        F = algorithm.pop.get("F")
        # self.first_objective.append(np.mean(F[:, None]))
        if F.shape[1] > 1:  # if F[1] exists (multi obj GA)
            F[:, 0] = -mean_per_trace  # set first column
            algorithm.pop.set("F", F)
            self.second_objective.append(np.mean(F[:, 1]))

    def get_data(self) -> Dict[str, Any]:
        """Retrieve and return recorded data from the optimization process."""
        return {
            "constraint_history": self.constraint_history,
            "second_objective": self.second_objective,
            "first_objective": self.first_objective
        }





