from pymoo.core.callback import Callback
import numpy as np
from pymoo.core.population import Population
from scipy.spatial.distance import cdist, pdist
from scipy.spatial.distance import pdist, squareform
import time


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
        self.generation = 0


    def notify(self, algorithm):
        """Update the problem's current population and gather data for plotting."""

        # Get current population decision variables
        population = algorithm.pop.get("X")

        # Efficient Hamming distance computation
        # pdist is faster than cdist for symmetric pairwise distance matrices
        # storing only the upper triangular matrix (the diagonal is 0 and the lower triangular matrix has the same values as the upper one)
        dist_matrix = squareform(pdist(population, metric="hamming")) # 0.1s | pop = 3000 - trace_length = 50 - model1

        # eliminate the diagonal of 0
        n = dist_matrix.shape[0]
        mean_per_trace = (np.sum(dist_matrix, axis=1) - 0) / (n - 1)


        # Calculate mean diversity score per individual
        # Subtracting diagonal and dividing by (n-1)
        # n = dist_matrix.shape[0]
        # mean_per_trace = (np.sum(dist_matrix, axis=1) - np.diag(dist_matrix)) / (n - 1)

        # set F
        algorithm.pop.set("F", -mean_per_trace[:, None])


        # Store current population
        algorithm.problem.set_current_population(population)

        # Constraints
        G = algorithm.pop.get("G")
        # if G is not None and G.size > 0:
        self.constraint_history.append(np.mean(G[:, 0]))

        # Objective tracking
        F = algorithm.pop.get("F")
        self.first_objective.append(np.mean(F[:, 0]))

        self.generation += 1

        # if F.shape[1] > 1:  # if F[1] exists (multi obj GA)
        #     second_objective = F[:, 1]
        #     self.second_objective.append(np.mean(second_objective))





    def get_data(self):
        """Retrieve and return recorded data from the optimization process."""
        return {
            "constraint_history": self.constraint_history,
            "second_objective": self.second_objective,
            "first_objective": self.first_objective,
            "generations": self.generation,
        }





